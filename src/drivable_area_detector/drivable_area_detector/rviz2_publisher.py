"""
Smooth RViz2 Publisher for Road Boundaries with RANSAC Interpolation

Subscribes to ring boundary detections and publishes smoothed line markers.
Author: Victor Dessenne (Enhanced with interpolation)
Date: 2026-01-13
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class SmoothRoadMarkersPublisher(Node):
    """ROS2 node for publishing smoothed road boundary line markers."""
    
    def __init__(self):
        super().__init__('smooth_road_markers_publisher')
        
        # Parameters
        self.declare_parameter('publish_frame', 'velodyne')
        self.declare_parameter('marker_line_width', 0.08)
        self.declare_parameter('polynomial_degree', 2)
        self.declare_parameter('residual_threshold', 0.20)
        self.declare_parameter('resample_spacing', 0.3)  # Spacing entre points interpolés
        
        self.frame_id = self.get_parameter('publish_frame').value
        self.line_width = self.get_parameter('marker_line_width').value
        self.poly_degree = self.get_parameter('polynomial_degree').value
        self.ransac_threshold = self.get_parameter('residual_threshold').value
        self.spacing = self.get_parameter('resample_spacing').value
        
        # QoS: BEST_EFFORT to match LiDAR sensor
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/road_boundaries', qos)
        
        # Subscriber
        self.create_subscription(PointCloud2, '/ring_boundaries',
                                 self.boundaries_callback, qos)
        
        self.get_logger().info(
            f'Smooth road markers publisher ready '
            f'(frame: {self.frame_id}, degree: {self.poly_degree}, spacing: {self.spacing}m)'
        )
    
    def boundaries_callback(self, msg: PointCloud2):
        """Parse boundaries and publish smoothed line markers."""
        # Parse PointCloud2: extract left (side_id=1) and right (side_id=-1) points
        left_points, right_points = self.parse_boundaries(msg)
        
        # Check validity
        if len(left_points) < 5 and len(right_points) < 5:
            self.get_logger().warn('Insufficient boundary points', throttle_duration_sec=2.0)
            self.publish_empty()
            return
        
        # Create marker array
        markers = MarkerArray()
        
        # Left boundary (cyan) - with interpolation
        if len(left_points) >= 5:
            smooth_left = self.interpolate_boundary(left_points)
            if smooth_left is not None and len(smooth_left) >= 2:
                markers.markers.append(
                    self.create_line_marker(smooth_left, marker_id=0,
                                           color=(0.0, 1.0, 1.0), label='Left')
                )
        
        # Right boundary (magenta) - with interpolation
        if len(right_points) >= 5:
            smooth_right = self.interpolate_boundary(right_points)
            if smooth_right is not None and len(smooth_right) >= 2:
                markers.markers.append(
                    self.create_line_marker(smooth_right, marker_id=1,
                                           color=(1.0, 0.0, 1.0), label='Right')
                )
        
        # Publish
        if len(markers.markers) > 0:
            self.marker_pub.publish(markers)
        else:
            self.publish_empty()
    
    def interpolate_boundary(self, points):
        """
        Fit RANSAC polynomial and resample at regular intervals for smooth visualization.
        
        Args:
            points: List of (x, y, z) tuples
            
        Returns:
            List of interpolated (x, y, z) tuples or None if fit fails
        """
        if len(points) < 5:
            return None
        
        # Convert to numpy
        points_array = np.array(points)
        x = points_array[:, 0].reshape(-1, 1)
        y = points_array[:, 1]
        z = points_array[:, 2]
        
        # Fit polynomial using RANSAC
        try:
            base_model = make_pipeline(
                PolynomialFeatures(degree=self.poly_degree),
                RANSACRegressor(
                    residual_threshold=self.ransac_threshold,
                    max_trials=200,
                    random_state=42,
                    stop_probability=0.99
                )
            )
            base_model.fit(x, y)
            
            # Check inlier ratio
            inlier_mask = base_model.named_steps['ransacregressor'].inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(x)
            
            if inlier_ratio < 0.15:  # Moins de 15% d'inliers = mauvais fit
                self.get_logger().debug(
                    f'Low inlier ratio: {inlier_ratio:.2f}',
                    throttle_duration_sec=2.0
                )
                return None
            
            # Resample at regular intervals
            x_min, x_max = np.min(x), np.max(x)
            x_resampled = np.arange(x_min, x_max + self.spacing, self.spacing).reshape(-1, 1)
            y_resampled = base_model.predict(x_resampled)
            
            # Interpolate z linearly (hauteur peu importante)
            z_resampled = np.interp(x_resampled.flatten(), x.flatten(), z)
            
            # Convert back to list of tuples
            smooth_points = [
                (float(x_resampled[i]), float(y_resampled[i]), float(z_resampled[i]))
                for i in range(len(x_resampled))
            ]
            
            return smooth_points
            
        except Exception as e:
            self.get_logger().debug(
                f'RANSAC fit failed: {str(e)}',
                throttle_duration_sec=2.0
            )
            return None
    
    def parse_boundaries(self, msg: PointCloud2):
        """
        Extract left and right boundary points from PointCloud2.
        Expected format: x, y, z, side_id (float32, float32, float32, int32)
        Convention: side_id=1 (left/+y), side_id=-1 (right/-y)
        """
        left_pts = []
        right_pts = []
        point_step = msg.point_step
        
        for i in range(0, len(msg.data), point_step):
            if i + 16 > len(msg.data):  # Need 16 bytes minimum
                break
            
            try:
                x = struct.unpack_from('f', msg.data, i)[0]
                y = struct.unpack_from('f', msg.data, i+4)[0]
                z = struct.unpack_from('f', msg.data, i+8)[0]
                side_id = struct.unpack_from('i', msg.data, i+12)[0]
                
                if side_id == 1:
                    left_pts.append((x, y, z))
                elif side_id == -1:
                    right_pts.append((x, y, z))
            except struct.error:
                continue
        
        # Sort by x-coordinate
        left_pts.sort(key=lambda p: p[0])
        right_pts.sort(key=lambda p: p[0])
        
        return left_pts, right_pts
    
    def create_line_marker(self, points, marker_id, color, label):
        """Create LINE_STRIP marker from list of (x,y,z) tuples."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'road_boundaries'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Add points
        for x, y, z in points:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), float(z)
            marker.points.append(p)
        
        # Style
        marker.scale.x = self.line_width
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200_000_000  # 200ms
        
        return marker
    
    def publish_empty(self):
        """Clear markers from RViz."""
        markers = MarkerArray()
        for i in range(2):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'road_boundaries'
            m.id = i
            m.action = Marker.DELETE
            markers.markers.append(m)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = SmoothRoadMarkersPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
