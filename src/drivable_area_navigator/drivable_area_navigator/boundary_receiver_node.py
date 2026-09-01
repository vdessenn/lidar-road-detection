#!/usr/bin/env python3
"""
Boundary Receiver Node - Subscribes to /ring_boundaries and publishes centerline.

This node:
1. Subscribes to /ring_boundaries (PointCloud2 from detector)
2. Parses boundary points (left/right via side_id field)
3. Computes centerline using LocalCenterlinePlanner
4. Publishes nav_msgs/Path for visualization

Note: Boundaries are already in vehicle frame (velodyne), so car is at origin.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

from .types import (
    pointcloud2_to_boundaries,
    path_to_ros,
    BoundaryData
)
from .edge_smoother import EdgeSmoother, SmootherConfig
from .centerline_planner import LocalCenterlinePlanner, CenterlinePlannerConfig


class BoundaryReceiverNode(Node):
    """
    ROS2 node that receives boundaries and computes centerline.

    Combines boundary parsing, smoothing, and centerline planning
    in a single node for efficiency (reduces message passing overhead).
    """

    def __init__(self):
        super().__init__('boundary_receiver_node')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self.frame_id = self.get_parameter('frame_id').value

        # Initialize edge smoother
        smoother_config = SmootherConfig(
            kalman_enabled=self.get_parameter('edge_smoother.enable_kalman').value,
            kalman_process_noise=self.get_parameter('edge_smoother.kalman_process_noise').value,
            kalman_measurement_noise=self.get_parameter('edge_smoother.kalman_measurement_noise').value
        )
        self.smoother = EdgeSmoother(smoother_config)

        # Initialize centerline planner
        planner_config = CenterlinePlannerConfig(
            min_forward_dist=self.get_parameter('centerline_planner.min_forward_dist').value,
            max_forward_dist=self.get_parameter('centerline_planner.max_forward_dist').value,
            num_slices=self.get_parameter('centerline_planner.num_slices').value,
            safety_margin=self.get_parameter('centerline_planner.safety_margin').value,
            right_bias_alpha=self.get_parameter('centerline_planner.right_bias_alpha').value,
            adaptive_bias_enabled=self.get_parameter('centerline_planner.adaptive_bias_enabled').value,
            baseline_track_width=self.get_parameter('centerline_planner.baseline_track_width').value,
            track_widening_threshold=self.get_parameter('centerline_planner.track_widening_threshold').value
        )
        self.planner = LocalCenterlinePlanner(planner_config)

        # QoS profile matching detector pipeline
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber for boundary data
        self.boundary_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('topics.boundaries_input').value,
            self.boundary_callback,
            qos
        )

        # Publishers - use RELIABLE QoS for path (RViz2 compatibility)
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.path_pub = self.create_publisher(
            Path,
            self.get_parameter('topics.centerline_output').value,
            path_qos
        )

        # Visualization publishers - also use RELIABLE QoS for RViz2
        self.viz_pub = self.create_publisher(
            MarkerArray,
            '/centerline_markers',
            path_qos
        )

        self.get_logger().info('Boundary receiver node initialized')
        self.get_logger().info(f'  Input: {self.get_parameter("topics.boundaries_input").value}')
        self.get_logger().info(f'  Output: {self.get_parameter("topics.centerline_output").value}')

    def _declare_parameters(self):
        """Declare all node parameters."""
        # Topics
        self.declare_parameter('topics.boundaries_input', '/ring_boundaries')
        self.declare_parameter('topics.centerline_output', '/planned_centerline')
        self.declare_parameter('frame_id', 'velodyne')

        # Edge smoother
        self.declare_parameter('edge_smoother.enable_kalman', True)
        self.declare_parameter('edge_smoother.kalman_process_noise', 0.1)
        self.declare_parameter('edge_smoother.kalman_measurement_noise', 0.5)

        # Centerline planner
        self.declare_parameter('centerline_planner.min_forward_dist', 2.0)
        self.declare_parameter('centerline_planner.max_forward_dist', 20.0)
        self.declare_parameter('centerline_planner.num_slices', 40)
        self.declare_parameter('centerline_planner.safety_margin', 0.5)
        self.declare_parameter('centerline_planner.right_bias_alpha', 0.45)
        self.declare_parameter('centerline_planner.adaptive_bias_enabled', True)
        self.declare_parameter('centerline_planner.baseline_track_width', 6.0)
        self.declare_parameter('centerline_planner.track_widening_threshold', 1.1)

    def boundary_callback(self, msg: PointCloud2):
        """
        Process incoming boundary data and publish centerline.

        Args:
            msg: PointCloud2 message with x, y, z, side_id fields
        """
        try:
            self.get_logger().info(f'Callback triggered! Points in msg: {msg.width}')

            # Parse PointCloud2 to boundaries
            boundaries = pointcloud2_to_boundaries(msg)

            self.get_logger().info(f'Received: left={len(boundaries.left_points)}, right={len(boundaries.right_points)} pts')
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        if len(boundaries.left_points) < 2 and len(boundaries.right_points) < 2:
            self.get_logger().warn('Insufficient boundary points')
            return

        # Apply Kalman smoothing
        boundaries = self.smoother.smooth(boundaries)

        # Extract 2D points (x, y) for centerline computation
        left_2d = boundaries.left_points[:, :2] if len(boundaries.left_points) > 0 else np.empty((0, 2))
        right_2d = boundaries.right_points[:, :2] if len(boundaries.right_points) > 0 else np.empty((0, 2))

        # Compute centerline
        # Note: Since boundaries are in velodyne frame (vehicle-centered),
        # the car is at origin with yaw=0
        centerline = self.planner.compute(
            left_edge=left_2d,
            right_edge=right_2d,
            car_x=0.0,  # Vehicle at origin in velodyne frame
            car_y=0.0,
            car_yaw=0.0
        )

        self.get_logger().info(f'Centerline: {len(centerline)} points')

        if len(centerline) < 2:
            self.get_logger().warn('Centerline computation returned insufficient points')
            return

        # Publish path
        path_msg = path_to_ros(centerline, self.frame_id, self.get_clock().now())
        self.path_pub.publish(path_msg)

        # Publish visualization markers
        self._publish_visualization(centerline, boundaries)

    def _publish_visualization(self, centerline: np.ndarray, boundaries: BoundaryData):
        """Publish visualization markers for RViz."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Centerline marker (green line strip)
        centerline_marker = Marker()
        centerline_marker.header.frame_id = self.frame_id
        centerline_marker.header.stamp = stamp
        centerline_marker.ns = 'centerline'
        centerline_marker.id = 0
        centerline_marker.type = Marker.LINE_STRIP
        centerline_marker.action = Marker.ADD
        centerline_marker.scale.x = 0.1  # Line width
        centerline_marker.color.r = 0.0
        centerline_marker.color.g = 1.0
        centerline_marker.color.b = 0.0
        centerline_marker.color.a = 1.0
        centerline_marker.lifetime.nanosec = 200000000  # 200ms

        for point in centerline:
            p = PoseStamped().pose.position
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.1  # Slightly above ground
            centerline_marker.points.append(p)

        markers.markers.append(centerline_marker)

        self.viz_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
