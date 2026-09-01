#!/usr/bin/env python3
"""
Lidar Edge Simulator - Simulates lidar rays detecting track boundaries.

Casts rays from the vehicle position and detects where they hit
the track edges (transitions from navigable to non-navigable).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class LidarScan:
    """Result of a simulated lidar scan."""
    left_edge: np.ndarray       # (N, 2) left boundary points [x, y] in world coords
    right_edge: np.ndarray      # (N, 2) right boundary points [x, y] in world coords
    all_points: np.ndarray      # (M, 2) all detected edge points
    ranges: np.ndarray          # (M,) distance to each point
    angles: np.ndarray          # (M,) angle of each ray (relative to car heading)
    car_position: Tuple[float, float]  # Car position when scan was taken
    car_yaw: float              # Car heading when scan was taken

    # Island detection fields
    island_points: np.ndarray = None    # (P, 2) island boundary points in world coords
    has_island_ahead: bool = False      # True if island detected in forward arc
    island_center_angle: float = 0.0    # Angle to island center (relative to heading)


class LidarEdgeSimulator:
    """
    Simulates a forward-facing lidar sensor detecting track edges.

    The simulator casts rays from the car position and detects where
    they first hit a non-navigable cell (track boundary).

    Edge classification can use either:
    - Car heading (default, but causes issues when car turns wrong direction)
    - Track direction from reference path (more robust, prevents left/right confusion)
    """

    def __init__(
        self,
        navigable_grid: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        sensor_range: float = 40.0,
        num_rays: int = 180,
        fov: float = np.pi,  # 180 degrees
        ray_step: float = 0.2,  # Step size for ray marching (meters)
        reference_path: Optional[np.ndarray] = None  # Reference centerline for track-relative classification
    ):
        """
        Initialize the lidar simulator.

        Args:
            navigable_grid: Binary grid (True = navigable/drivable)
            origin: World origin (x, y) of the grid
            resolution: Grid resolution in meters per cell
            sensor_range: Maximum detection range in meters
            num_rays: Number of rays to cast
            fov: Field of view in radians (centered on car heading)
            ray_step: Step size for ray marching
            reference_path: Optional (N, 2) array of reference centerline points.
                           If provided, edge classification uses track direction
                           instead of car heading, preventing left/right confusion.
        """
        self.grid = navigable_grid
        self.origin = origin
        self.resolution = resolution
        self.sensor_range = sensor_range
        self.num_rays = num_rays
        self.fov = fov
        self.ray_step = ray_step
        self.reference_path = reference_path

        # Grid dimensions
        self.rows, self.cols = navigable_grid.shape

        # Track sequential progress along reference path
        # This prevents jumping between different parts of the path (e.g., at roundabouts)
        self._path_progress_idx = 0
        self._last_car_pos = None

    def reset_path_progress(self):
        """Reset path progress tracking (call when starting a new lap)."""
        self._path_progress_idx = 0
        self._last_car_pos = None

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = self.origin[0] + col * self.resolution
        y = self.origin[1] + row * self.resolution
        return x, y

    def is_navigable(self, x: float, y: float) -> bool:
        """Check if a world position is navigable."""
        row, col = self.world_to_grid(x, y)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col]
        return False  # Out of bounds = not navigable

    def get_track_direction(self, car_x: float, car_y: float) -> Optional[float]:
        """
        Get the track direction (yaw) at the given position using the reference path.

        Uses SEQUENTIAL PROGRESS along the path instead of nearest-point matching.
        This prevents jumping between different parts of the path (e.g., at roundabouts
        where entry and exit are close together), which would cause left/right edge swapping.

        Args:
            car_x, car_y: Current position in world coordinates

        Returns:
            Track direction in radians, or None if no reference path available
        """
        if self.reference_path is None or len(self.reference_path) < 10:
            return None

        n = len(self.reference_path)

        # Search for the best matching point within a forward window from current progress
        # This prevents jumping backwards or to distant parts of the path
        search_start = max(0, self._path_progress_idx - 10)  # Allow small backtrack
        search_end = min(n, self._path_progress_idx + 100)   # Look ahead up to 100 points

        # If we're near the end, wrap around for closed loops
        if search_end >= n:
            # Check if path is a closed loop (start ≈ end)
            start_end_dist = np.sqrt(
                (self.reference_path[0, 0] - self.reference_path[-1, 0])**2 +
                (self.reference_path[0, 1] - self.reference_path[-1, 1])**2
            )
            if start_end_dist < 5.0:  # Closed loop
                # Search wraps around
                search_range = np.vstack([
                    self.reference_path[search_start:n],
                    self.reference_path[0:min(100, search_end - n)]
                ])
                search_indices = list(range(search_start, n)) + list(range(0, min(100, search_end - n)))
            else:
                search_range = self.reference_path[search_start:search_end]
                search_indices = list(range(search_start, search_end))
        else:
            search_range = self.reference_path[search_start:search_end]
            search_indices = list(range(search_start, search_end))

        if len(search_range) == 0:
            return None

        # Find nearest point within the search window
        dx = search_range[:, 0] - car_x
        dy = search_range[:, 1] - car_y
        distances = np.sqrt(dx * dx + dy * dy)
        local_idx = np.argmin(distances)
        nearest_idx = search_indices[local_idx]

        # Update progress if we've moved forward
        if nearest_idx >= self._path_progress_idx or (nearest_idx < 10 and self._path_progress_idx > n - 20):
            self._path_progress_idx = nearest_idx

        # Get track direction from reference path tangent
        # Use points ahead and behind for robust direction estimate
        lookahead = 10  # Points to look ahead/behind
        idx_behind = max(0, nearest_idx - lookahead)
        idx_ahead = min(n - 1, nearest_idx + lookahead)

        # Handle the case where we're near the ends of the path
        if idx_ahead - idx_behind < 5:
            # Near end of path, use what we have
            if nearest_idx < n - 1:
                direction = self.reference_path[nearest_idx + 1] - self.reference_path[nearest_idx]
            else:
                direction = self.reference_path[nearest_idx] - self.reference_path[nearest_idx - 1]
        else:
            direction = self.reference_path[idx_ahead] - self.reference_path[idx_behind]

        track_yaw = np.arctan2(direction[1], direction[0])
        return track_yaw

    def cast_ray(
        self,
        start_x: float,
        start_y: float,
        angle: float,
        detect_islands: bool = True
    ) -> Optional[Tuple[float, float, float, str]]:
        """
        Cast a single ray and find where it hits a track edge.

        Args:
            start_x, start_y: Ray origin in world coordinates
            angle: Ray direction in radians (world frame)
            detect_islands: If True, check if hit is island or outer boundary

        Returns:
            (hit_x, hit_y, distance, hit_type) where hit_type is 'boundary' or 'island'
            or None if no hit within range
        """
        dx = np.cos(angle) * self.ray_step
        dy = np.sin(angle) * self.ray_step

        x, y = start_x, start_y
        distance = 0.0

        # Start by checking if we're on navigable ground
        was_navigable = self.is_navigable(x, y)

        while distance < self.sensor_range:
            x += dx
            y += dy
            distance += self.ray_step

            is_nav = self.is_navigable(x, y)

            # Detect transition from navigable to non-navigable
            if was_navigable and not is_nav:
                # Found edge - back up slightly for more accurate position
                hit_x = x - dx/2
                hit_y = y - dy/2
                hit_dist = distance - self.ray_step/2

                # Determine if this is an island or outer boundary
                hit_type = 'boundary'
                if detect_islands:
                    # Continue ray to check if it re-enters navigable area (island)
                    temp_x, temp_y = x, y
                    temp_dist = distance
                    max_check_dist = min(self.sensor_range, distance + 20.0)  # Check up to 20m beyond

                    while temp_dist < max_check_dist:
                        temp_x += dx
                        temp_y += dy
                        temp_dist += self.ray_step

                        if self.is_navigable(temp_x, temp_y):
                            # Ray re-entered navigable area -> this was an island
                            hit_type = 'island'
                            break

                return (hit_x, hit_y, hit_dist, hit_type)

            was_navigable = is_nav

        return None  # No hit within range

    def scan(self, car_x: float, car_y: float, car_yaw: float) -> LidarScan:
        """
        Perform a lidar scan from the car's current position.

        Args:
            car_x, car_y: Car position in world coordinates
            car_yaw: Car heading in radians (0 = East, counterclockwise positive)

        Returns:
            LidarScan with detected edge points and island information
        """
        # Generate ray angles (centered on car heading)
        half_fov = self.fov / 2
        angles_relative = np.linspace(-half_fov, half_fov, self.num_rays)
        angles_world = angles_relative + car_yaw

        # Cast all rays
        all_points = []
        all_ranges = []
        all_angles = []
        island_points = []
        island_angles = []

        for i, (angle_rel, angle_world) in enumerate(zip(angles_relative, angles_world)):
            hit = self.cast_ray(car_x, car_y, angle_world)
            if hit is not None:
                hit_x, hit_y, distance, hit_type = hit
                all_points.append([hit_x, hit_y])
                all_ranges.append(distance)
                all_angles.append(angle_rel)

                if hit_type == 'island':
                    island_points.append([hit_x, hit_y])
                    island_angles.append(angle_rel)

        if len(all_points) == 0:
            # No edges detected - return empty scan
            return LidarScan(
                left_edge=np.array([]).reshape(0, 2),
                right_edge=np.array([]).reshape(0, 2),
                all_points=np.array([]).reshape(0, 2),
                ranges=np.array([]),
                angles=np.array([]),
                car_position=(car_x, car_y),
                car_yaw=car_yaw,
                island_points=np.array([]).reshape(0, 2),
                has_island_ahead=False,
                island_center_angle=0.0
            )

        all_points = np.array(all_points)
        all_ranges = np.array(all_ranges)
        all_angles = np.array(all_angles)

        # Determine the reference heading for edge classification
        # If we have a reference path, use track direction (robust to car heading errors)
        # Otherwise, fall back to car heading (original behavior)
        track_yaw = self.get_track_direction(car_x, car_y)
        classification_yaw = track_yaw if track_yaw is not None else car_yaw

        # Transform all points to track-relative frame for edge classification
        # This ensures left/right classification is consistent with track geometry
        # even when the car's heading deviates from the track direction
        all_points_track_frame = self._to_car_frame(all_points, car_x, car_y, classification_yaw)

        # Classify points as left or right edge based on LATERAL POSITION in track frame
        # Points with positive Y (to the left of track direction) = left edge
        # Points with negative Y (to the right of track direction) = right edge
        left_mask = all_points_track_frame[:, 1] > 0
        right_mask = all_points_track_frame[:, 1] <= 0

        left_edge = all_points[left_mask]
        right_edge = all_points[right_mask]

        # Sort edges by forward distance in CAR frame (for path planning)
        # We classify by track frame but sort by car frame for usability
        all_points_car_frame = self._to_car_frame(all_points, car_x, car_y, car_yaw)

        if len(left_edge) > 0:
            left_car_frame = all_points_car_frame[left_mask]
            sort_idx = np.argsort(left_car_frame[:, 0])  # Sort by forward (x in car frame)
            left_edge = left_edge[sort_idx]

        if len(right_edge) > 0:
            right_car_frame = all_points_car_frame[right_mask]
            sort_idx = np.argsort(right_car_frame[:, 0])
            right_edge = right_edge[sort_idx]

        # Process island information
        has_island_ahead = len(island_points) > 0
        island_center_angle = 0.0
        island_pts_arr = np.array([]).reshape(0, 2)

        if has_island_ahead:
            island_pts_arr = np.array(island_points)
            # Compute angle to island center (average of island point angles)
            island_center_angle = np.mean(island_angles)

            # Only consider island "ahead" if it's in the forward arc (-45 to +45 degrees)
            forward_island_mask = np.abs(np.array(island_angles)) < np.pi / 4
            has_island_ahead = np.any(forward_island_mask)

        return LidarScan(
            left_edge=left_edge,
            right_edge=right_edge,
            all_points=all_points,
            ranges=all_ranges,
            angles=all_angles,
            car_position=(car_x, car_y),
            car_yaw=car_yaw,
            island_points=island_pts_arr,
            has_island_ahead=has_island_ahead,
            island_center_angle=island_center_angle
        )

    def _to_car_frame(
        self,
        points: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float
    ) -> np.ndarray:
        """
        Transform points from world frame to car frame.

        Car frame: origin at car, X forward, Y left.
        """
        # Translate to car origin
        translated = points - np.array([car_x, car_y])

        # Rotate by -car_yaw
        cos_yaw = np.cos(-car_yaw)
        sin_yaw = np.sin(-car_yaw)
        rotation = np.array([[cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw]])

        return translated @ rotation.T

    def _from_car_frame(
        self,
        points: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float
    ) -> np.ndarray:
        """Transform points from car frame to world frame."""
        # Rotate by car_yaw
        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        rotation = np.array([[cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw]])

        rotated = points @ rotation.T

        # Translate to world
        return rotated + np.array([car_x, car_y])


if __name__ == "__main__":
    # Test the lidar simulator
    print("Testing LidarEdgeSimulator...")

    # Create a simple track (corridor)
    grid = np.zeros((100, 200), dtype=bool)
    # Make a corridor from col 50 to 150, rows 30 to 70
    grid[30:70, 50:150] = True

    simulator = LidarEdgeSimulator(
        navigable_grid=grid,
        origin=(0, 0),
        resolution=1.0,
        sensor_range=40.0,
        num_rays=90
    )

    # Car in middle of corridor, heading right (East)
    car_x, car_y = 100.0, 50.0
    car_yaw = 0.0  # Facing East

    scan = simulator.scan(car_x, car_y, car_yaw)

    print(f"Car position: ({car_x}, {car_y}), heading: {np.degrees(car_yaw)}°")
    print(f"Total edge points: {len(scan.all_points)}")
    print(f"Left edge points: {len(scan.left_edge)}")
    print(f"Right edge points: {len(scan.right_edge)}")

    if len(scan.left_edge) > 0:
        print(f"Left edge Y range: {scan.left_edge[:, 1].min():.1f} to {scan.left_edge[:, 1].max():.1f}")
    if len(scan.right_edge) > 0:
        print(f"Right edge Y range: {scan.right_edge[:, 1].min():.1f} to {scan.right_edge[:, 1].max():.1f}")
