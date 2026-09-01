#!/usr/bin/env python3
"""
GridMapExtractor - Extracts road point cloud data from ROS2 MCAP bag

Uses raw MCAP reading to extract PointCloud2 data from /road_whole topic.
"""

import os
import struct
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GridMapData:
    """Container for extracted grid map data."""
    raw_data: np.ndarray          # Raw grid data (may have NaN)
    resolution: float              # Meters per cell
    origin: Tuple[float, float]    # World origin (x, y)
    dimensions: Tuple[int, int]    # Grid dimensions (rows, cols)
    length_x: float                # World size X (meters)
    length_y: float                # World size Y (meters)
    layers: List[str]              # Available layer names
    road_layer_idx: int            # Index of road layer
    point_cloud: Optional[np.ndarray] = None  # Original XYZ points


class GridMapExtractor:
    """Extract road data from ROS2 MCAP bag file using /road_whole PointCloud2."""

    def __init__(self, bag_path: str, topic: str = "/road_whole", resolution: float = 0.5):
        """
        Args:
            bag_path: Path to bag directory or MCAP file
            topic: Topic name to read (default: /road_whole for PointCloud2)
            resolution: Grid resolution in meters per cell
        """
        self.bag_path = bag_path
        self.topic = topic
        self.resolution = resolution
        self.data: Optional[GridMapData] = None
        self.points: Optional[np.ndarray] = None

    def extract(self) -> GridMapData:
        """
        Extract road points from bag file and convert to grid.

        Returns:
            GridMapData containing the extracted road grid
        """
        try:
            from mcap.reader import make_reader
        except ImportError:
            raise ImportError(
                "Please install mcap package:\n"
                "pip install mcap"
            )

        mcap_file = self._find_mcap_file()
        print(f"Reading MCAP file: {mcap_file}")

        # Read raw data from MCAP
        with open(mcap_file, "rb") as f:
            reader = make_reader(f)

            # First, list all available topics
            print("\nAvailable topics:")
            for channel_id, channel in reader.get_summary().channels.items():
                print(f"  {channel.topic}: {channel.message_encoding}")

            # Reset and read messages
            f.seek(0)
            reader = make_reader(f)

            for schema, channel, message in reader.iter_messages(topics=[self.topic]):
                print(f"\nFound message on topic: {channel.topic}")
                print(f"  Data size: {len(message.data)} bytes")

                # Parse PointCloud2 message
                self.points = self._parse_pointcloud2(message.data)

                if self.points is not None and len(self.points) > 0:
                    print(f"  Extracted {len(self.points)} points")
                    print(f"  X range: [{self.points[:, 0].min():.1f}, {self.points[:, 0].max():.1f}]")
                    print(f"  Y range: [{self.points[:, 1].min():.1f}, {self.points[:, 1].max():.1f}]")
                    if self.points.shape[1] > 2:
                        print(f"  Z range: [{self.points[:, 2].min():.1f}, {self.points[:, 2].max():.1f}]")

                    # Convert to grid
                    self.data = self._points_to_grid(self.points)
                    return self.data

        raise ValueError(f"No valid data found on topic {self.topic}")

    def _parse_pointcloud2(self, data: bytes) -> Optional[np.ndarray]:
        """
        Parse PointCloud2 message from raw CDR bytes.
        Handles zstd compression.
        """
        try:
            # Check for zstd compression (magic number 0x28B52FFD)
            if data[:4] == b'\x28\xb5\x2f\xfd':
                import zstandard as zstd
                print("  Data is zstd compressed, decompressing...")
                dctx = zstd.ZstdDecompressor()
                data = dctx.decompress(data)

            data_len = len(data)
            print(f"  Decompressed size: {data_len} bytes ({data_len / 1e6:.1f} MB)")

            # Parse CDR PointCloud2 header
            offset = 4  # Skip CDR encapsulation header

            # Header: stamp (8 bytes)
            offset += 8

            # frame_id string
            frame_id_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + frame_id_len
            offset = (offset + 3) & ~3  # Align

            # height, width
            height = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            width = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            num_points = height * width
            print(f"  PointCloud2: {num_points} points ({width}x{height})")

            # fields array
            num_fields = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            x_offset = y_offset = z_offset = 0
            for _ in range(num_fields):
                name_len = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                name = data[offset:offset + name_len - 1].decode('utf-8', errors='ignore')
                offset += name_len
                offset = (offset + 3) & ~3

                field_offset = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                offset += 1  # datatype
                offset = (offset + 3) & ~3
                offset += 4  # count

                if name == 'x':
                    x_offset = field_offset
                elif name == 'y':
                    y_offset = field_offset
                elif name == 'z':
                    z_offset = field_offset

            print(f"  Field offsets: x={x_offset}, y={y_offset}, z={z_offset}")

            # Skip is_bigendian
            offset += 1
            offset = (offset + 3) & ~3

            # point_step, row_step
            point_step = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            offset += 4  # row_step

            print(f"  point_step={point_step} bytes")

            # data array length
            data_array_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            point_data = data[offset:offset + data_array_len]
            print(f"  Point data: {len(point_data)} bytes")

            # Parse all points using numpy for efficiency
            num_points_actual = min(num_points, len(point_data) // point_step)
            print(f"  Parsing {num_points_actual} points...")

            # Use numpy structured array for fast parsing
            # Create a dtype that matches point_step with padding
            # We only need x, y, z which are floats at offsets 0, 4, 8
            point_dtype = np.dtype([
                ('x', '<f4'),      # offset 0
                ('y', '<f4'),      # offset 4
                ('z', '<f4'),      # offset 8
                ('_pad', f'V{point_step - 12}')  # padding to match point_step
            ])

            # Truncate point_data to exact multiple of point_step
            usable_bytes = num_points_actual * point_step
            points_array = np.frombuffer(point_data[:usable_bytes], dtype=point_dtype)

            x_values = points_array['x']
            y_values = points_array['y']
            z_values = points_array['z']

            # Filter valid points
            valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
            x_valid = x_values[valid]
            y_valid = y_values[valid]
            z_valid = z_values[valid]

            print(f"  Valid points: {len(x_valid)}")
            print(f"  X range: [{x_valid.min():.1f}, {x_valid.max():.1f}]")
            print(f"  Y range: [{y_valid.min():.1f}, {y_valid.max():.1f}]")
            print(f"  Z range: [{z_valid.min():.1f}, {z_valid.max():.1f}]")

            return np.column_stack([x_valid, y_valid, z_valid]).astype(np.float32)

        except Exception as e:
            print(f"  Error parsing PointCloud2: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _points_to_grid(self, points: np.ndarray) -> GridMapData:
        """Convert point cloud to occupancy grid."""
        # Get bounds
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

        # Add padding
        padding = 2.0
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding

        length_x = x_max - x_min
        length_y = y_max - y_min

        # Calculate grid dimensions
        cols = int(np.ceil(length_x / self.resolution))
        rows = int(np.ceil(length_y / self.resolution))

        print(f"\nCreating grid:")
        print(f"  World bounds: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")
        print(f"  Resolution: {self.resolution} m/cell")
        print(f"  Dimensions: {rows} x {cols} cells")

        # Create grid (NaN for empty cells)
        grid = np.full((rows, cols), np.nan, dtype=np.float32)

        # Convert points to grid coordinates and mark cells
        px = ((points[:, 0] - x_min) / self.resolution).astype(int)
        py = ((points[:, 1] - y_min) / self.resolution).astype(int)

        # Clip to grid bounds
        px = np.clip(px, 0, cols - 1)
        py = np.clip(py, 0, rows - 1)

        # Mark cells with points (use Z value or 1.0)
        if points.shape[1] > 2:
            for i in range(len(points)):
                grid[py[i], px[i]] = points[i, 2]
        else:
            grid[py, px] = 1.0

        # Count valid cells
        valid_count = np.sum(~np.isnan(grid))
        print(f"  Valid cells: {valid_count} ({100 * valid_count / (rows * cols):.1f}%)")

        return GridMapData(
            raw_data=grid,
            resolution=self.resolution,
            origin=(x_min, y_min),
            dimensions=(rows, cols),
            length_x=length_x,
            length_y=length_y,
            layers=['road'],
            road_layer_idx=0,
            point_cloud=points
        )

    def _find_mcap_file(self) -> str:
        """Find .mcap file in bag directory or return path if it's a file."""
        if os.path.isfile(self.bag_path) and self.bag_path.endswith('.mcap'):
            return self.bag_path

        if os.path.isdir(self.bag_path):
            for f in os.listdir(self.bag_path):
                if f.endswith('.mcap'):
                    return os.path.join(self.bag_path, f)

        raise FileNotFoundError(f"No MCAP file found in {self.bag_path}")

    def _find_road_layer(self, layers: List[str]) -> int:
        """Find index of road layer."""
        # Look for "road" layer specifically
        for i, name in enumerate(layers):
            if name.lower() == 'road':
                return i

        # Fallback: look for occupancy-related
        for i, name in enumerate(layers):
            if 'occupancy' in name.lower() or 'obstacle' in name.lower():
                return i

        # Last resort: first layer
        print(f"Warning: No 'road' layer found, using first layer: {layers[0]}")
        return 0

    def get_point_coordinates(self) -> np.ndarray:
        """
        Get world coordinates of all valid (non-NaN) points.

        Returns:
            (N, 2) array of [x, y] world coordinates
        """
        if self.data is None:
            raise ValueError("No data extracted yet")

        # Find indices of valid points
        valid_mask = ~np.isnan(self.data.raw_data)
        rows, cols = np.where(valid_mask)

        # Convert to world coordinates
        # x = origin_x + col * resolution
        # y = origin_y + row * resolution (or origin_y + (max_row - row) * resolution for flipped)
        x = self.data.origin[0] + cols * self.data.resolution
        y = self.data.origin[1] + rows * self.data.resolution

        return np.column_stack([x, y])

    def get_point_values(self) -> np.ndarray:
        """
        Get values (alpha/intensity) of all valid points.

        Returns:
            (N,) array of point values
        """
        if self.data is None:
            raise ValueError("No data extracted yet")

        valid_mask = ~np.isnan(self.data.raw_data)
        return self.data.raw_data[valid_mask]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python grid_extractor.py <bag_path>")
        sys.exit(1)

    bag_path = sys.argv[1]
    extractor = GridMapExtractor(bag_path)
    data = extractor.extract()

    print(f"\nExtraction successful!")
    print(f"Shape: {data.raw_data.shape}")
