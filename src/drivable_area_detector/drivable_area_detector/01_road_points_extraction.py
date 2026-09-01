#!/usr/bin/env python3
"""
Road Points Extraction - Simplified (Less is More)

Extracts road surface points by intensity filtering on ground points.
- Input: /patchworkpp/ground (ground-segmented points)
- Output: /road_surface (low-intensity road points)

Road surface (asphalt) has LOW intensity (0-15).
Road markings (white paint) have HIGH intensity (20-60).

Temporal Buffer: Points must appear in 3+ of last 5 frames to be confirmed.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import numpy as np
from collections import deque, defaultdict
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

class RoadPointsExtractionNode(Node):
    """Simple intensity-based road extraction from ground points."""

    def __init__(self):
        super().__init__('road_points_extraction_node')

        # Declare parameters
        self.declare_parameter('topics.ground_cloud_input', '/patchworkpp/ground')
        self.declare_parameter('topics.road_extraction_output', '/road_surface')
        self.declare_parameter('qos_depth', 1)
        self.declare_parameter('road_points_extraction_node.enable_extraction', True)
        self.declare_parameter('road_points_extraction_node.min_road_intensity', 0.0)
        self.declare_parameter('road_points_extraction_node.max_road_intensity', 4.0)
        
        # Density filter parameters
        self.declare_parameter('road_points_extraction_node.max_range', 15.0)
        self.declare_parameter('road_points_extraction_node.min_neighbors', 7)
        self.declare_parameter('road_points_extraction_node.neighbor_radius', 0.5)
        
        # Temporal buffer parameters
        self.declare_parameter('road_points_extraction_node.buffer_frames', 5)
        self.declare_parameter('road_points_extraction_node.min_confirmations', 3)
        self.declare_parameter('road_points_extraction_node.bin_size', 0.3)
        
        # DBSCAN final clustering parameters
        self.declare_parameter('road_points_extraction_node.dbscan_eps', 0.8)
        self.declare_parameter('road_points_extraction_node.dbscan_min_samples', 15)
        self.declare_parameter('road_points_extraction_node.dbscan_min_cluster_size', 30)

        # Get parameters
        ground_topic = self.get_parameter('topics.ground_cloud_input').value
        output_topic = self.get_parameter('topics.road_extraction_output').value
        qos_depth = self.get_parameter('qos_depth').value

        self.enable_extraction = self.get_parameter('road_points_extraction_node.enable_extraction').value
        self.min_intensity = self.get_parameter('road_points_extraction_node.min_road_intensity').value
        self.max_intensity = self.get_parameter('road_points_extraction_node.max_road_intensity').value

        # QoS Profile
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )

        # Subscriber and publisher
        self.ground_sub = self.create_subscription(
            PointCloud2,
            ground_topic,
            self.ground_callback,
            sensor_qos
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, sensor_qos)

        self.get_logger().info(f'Road Points Extraction (Simplified):')
        self.get_logger().info(f'  Input:  {ground_topic}')
        self.get_logger().info(f'  Output: {output_topic}')
        self.get_logger().info(f'  Intensity: [{self.min_intensity:.1f}, {self.max_intensity:.1f}]')

        # Temporal buffer: from params
        self.buffer_frames = self.get_parameter('road_points_extraction_node.buffer_frames').value
        self.min_confirmations = self.get_parameter('road_points_extraction_node.min_confirmations').value
        self.bin_size = self.get_parameter('road_points_extraction_node.bin_size').value
        self.frame_buffer: deque = deque(maxlen=self.buffer_frames)
        
        # Density filter: from params
        self.max_range = self.get_parameter('road_points_extraction_node.max_range').value
        self.min_neighbors = self.get_parameter('road_points_extraction_node.min_neighbors').value
        self.neighbor_radius = self.get_parameter('road_points_extraction_node.neighbor_radius').value
        
        # DBSCAN parameters
        self.dbscan_eps = self.get_parameter('road_points_extraction_node.dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('road_points_extraction_node.dbscan_min_samples').value
        self.dbscan_min_cluster_size = self.get_parameter('road_points_extraction_node.dbscan_min_cluster_size').value
        
        self.get_logger().info(f'  Temporal: {self.min_confirmations}/{self.buffer_frames} frames, {self.bin_size}m bins')
        self.get_logger().info(f'  Density: {self.min_neighbors} neighbors in {self.neighbor_radius}m, max range {self.max_range}m')

    def _get_bin_key(self, x: float, y: float) -> tuple:
        """Convert point to spatial bin key."""
        return (int(x / self.bin_size), int(y / self.bin_size))

    def _filter_by_density(self, points: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple:
        """Filter points by local density. Returns (filtered_points, filtered_x, filtered_y)."""
        if len(points) == 0:
            return points, x, y
        
        # 1. Distance filter: reject points beyond max_range
        distances = np.sqrt(x**2 + y**2)
        range_mask = distances <= self.max_range
        
        if not np.any(range_mask):
            return points[:0], x[:0], y[:0]
        
        points_r = points[range_mask]
        x_r = x[range_mask]
        y_r = y[range_mask]
        
        # 2. Density filter: count neighbors for each point
        coords = np.column_stack((x_r, y_r))
        tree = cKDTree(coords)
        
        # Query neighbors within radius (includes self)
        neighbor_counts = tree.query_ball_point(coords, r=self.neighbor_radius, return_length=True)
        
        # Keep points with enough neighbors (subtract 1 for self)
        density_mask = (neighbor_counts - 1) >= self.min_neighbors
        
        return points_r[density_mask], x_r[density_mask], y_r[density_mask]

    def _filter_by_dbscan(self, points: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple:
        """FINAL CLUSTERING: Remove isolated noise using DBSCAN.
        
        Returns (filtered_points, filtered_x, filtered_y, n_clusters)
        
        DBSCAN Parameters:
        - eps: Maximum distance between neighbors (0.8m = ~3 LiDAR rings spacing)
        - min_samples: Minimum points to form core point (15 = robust to outliers)
        - min_cluster_size: Reject clusters with fewer points (30 = ~1m² patch)
        
        Labels:
        - -1: Noise points (isolated, discarded)
        - 0+: Cluster IDs (only clusters >= min_cluster_size kept)
        """
        if len(points) == 0:
            return points, x, y, 0
        
        # Build XY coordinates for 2D clustering
        coords_2d = np.column_stack((x, y))
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=-1  # Use all CPU cores
        ).fit(coords_2d)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Keep only points in clusters >= min_cluster_size
        valid_mask = np.zeros(len(points), dtype=bool)
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.dbscan_min_cluster_size:
                valid_mask |= cluster_mask
        
        n_rejected_clusters = n_clusters - len([c for c in range(n_clusters) if np.sum(labels == c) >= self.dbscan_min_cluster_size])
        
        self.get_logger().info(
            f'[DBSCAN] Clusters: {n_clusters} | Noise: {n_noise} | '
            f'Rejected small clusters: {n_rejected_clusters} | '
            f'Kept: {np.sum(valid_mask)}/{len(points)}'
        )
        
        return points[valid_mask], x[valid_mask], y[valid_mask], n_clusters

    
    def ground_callback(self, ground_msg: PointCloud2) -> None:
        """Filter ground points by intensity and temporal confirmation."""
        
        # Passthrough mode
        if not self.enable_extraction:
            self.publisher.publish(ground_msg)
            return

        # Parse ground cloud
        ground_data = np.frombuffer(ground_msg.data, dtype=np.uint8)
        ground_points = ground_data.reshape(-1, ground_msg.point_step)

        n_ground = len(ground_points)
        if n_ground == 0:
            return

        # Extract XY coordinates and intensity
        ground_x = ground_points[:, 0:4].view(np.float32).ravel()
        ground_y = ground_points[:, 4:8].view(np.float32).ravel()
        ground_intensity = ground_points[:, 12:16].view(np.float32).ravel()

        # Filter by intensity: keep only low intensity (road surface)
        intensity_mask = (ground_intensity >= self.min_intensity) & \
                        (ground_intensity <= self.max_intensity)

        candidate_points = ground_points[intensity_mask]
        candidate_x = ground_x[intensity_mask]
        candidate_y = ground_y[intensity_mask]
        
        if len(candidate_points) == 0:
            return

        # Apply density filter to remove isolated noise
        candidate_points, candidate_x, candidate_y = self._filter_by_density(
            candidate_points, candidate_x, candidate_y
        )
        
        if len(candidate_points) == 0:
            return

        # Build bin set for this frame
        current_bins = set()
        for x, y in zip(candidate_x, candidate_y):
            current_bins.add(self._get_bin_key(x, y))
        
        # Add to temporal buffer
        self.frame_buffer.append(current_bins)
        
        # Count confirmations across buffered frames
        bin_counts: defaultdict = defaultdict(int)
        for frame_bins in self.frame_buffer:
            for bin_key in frame_bins:
                bin_counts[bin_key] += 1
        
        # Keep only points in confirmed bins
        confirmed_mask = np.zeros(len(candidate_points), dtype=bool)
        for i, (x, y) in enumerate(zip(candidate_x, candidate_y)):
            bin_key = self._get_bin_key(x, y)
            if bin_counts[bin_key] >= self.min_confirmations:
                confirmed_mask[i] = True
        
        road_points = candidate_points[confirmed_mask]
        n_road = len(road_points)

        if n_road == 0:
            return

        # Log statistics
        self.get_logger().info(
            f'[01] Ground: {n_ground} → Candidates: {len(candidate_points)} → Road: {n_road}'
        )

        # Create output message
        out_msg = PointCloud2()
        out_msg.header = ground_msg.header
        out_msg.height = 1
        out_msg.width = n_road
        out_msg.fields = ground_msg.fields
        out_msg.is_bigendian = ground_msg.is_bigendian
        out_msg.point_step = ground_msg.point_step
        out_msg.row_step = n_road * ground_msg.point_step
        out_msg.data = road_points.tobytes()
        out_msg.is_dense = True

        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RoadPointsExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
