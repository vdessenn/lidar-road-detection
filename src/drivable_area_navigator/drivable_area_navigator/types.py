#!/usr/bin/env python3
"""
Types and Message Converters for drivable_area_navigator.

Provides dataclasses, ROS2 message converters, and utility functions.
"""

import numpy as np
import struct
import math
from dataclasses import dataclass
from typing import Tuple, Optional

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion


@dataclass
class BoundaryData:
    """Parsed boundary data from PointCloud2."""
    left_points: np.ndarray      # (N, 3) xyz coordinates of left boundary
    right_points: np.ndarray     # (M, 3) xyz coordinates of right boundary
    timestamp: float             # ROS timestamp as float
    frame_id: str                # Coordinate frame


@dataclass
class VehicleState:
    """Vehicle state for simulation and control."""
    x: float = 0.0          # X position (m)
    y: float = 0.0          # Y position (m)
    yaw: float = 0.0        # Heading (rad, 0=East, counterclockwise positive)
    speed: float = 0.0      # Speed (m/s)
    steering: float = 0.0   # Steering angle (rad)


def pointcloud2_to_boundaries(msg: PointCloud2) -> BoundaryData:
    """
    Parse PointCloud2 message with side_id field into left/right boundary arrays.

    Expected fields: x, y, z (float32), side_id (int32)
    side_id: 1 = left boundary, -1 = right boundary

    Args:
        msg: PointCloud2 message from /ring_boundaries

    Returns:
        BoundaryData with separated left and right points
    """
    # Extract timestamp
    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    frame_id = msg.header.frame_id

    # Find field offsets
    field_offsets = {}
    for field in msg.fields:
        field_offsets[field.name] = field.offset

    # Verify required fields
    required_fields = ['x', 'y', 'z', 'side_id']
    for field in required_fields:
        if field not in field_offsets:
            # Return empty boundaries if fields missing
            return BoundaryData(
                left_points=np.empty((0, 3)),
                right_points=np.empty((0, 3)),
                timestamp=timestamp,
                frame_id=frame_id
            )

    x_off = field_offsets['x']
    y_off = field_offsets['y']
    z_off = field_offsets['z']
    side_off = field_offsets['side_id']
    point_step = msg.point_step

    # Parse points
    left_pts = []
    right_pts = []

    for i in range(0, len(msg.data), point_step):
        x = struct.unpack_from('f', msg.data, i + x_off)[0]
        y = struct.unpack_from('f', msg.data, i + y_off)[0]
        z = struct.unpack_from('f', msg.data, i + z_off)[0]
        side_id = struct.unpack_from('i', msg.data, i + side_off)[0]

        if side_id == 1:
            left_pts.append([x, y, z])
        elif side_id == -1:
            right_pts.append([x, y, z])

    # Convert to numpy arrays
    left_points = np.array(left_pts) if left_pts else np.empty((0, 3))
    right_points = np.array(right_pts) if right_pts else np.empty((0, 3))

    # Sort by x-coordinate (forward distance)
    if len(left_points) > 0:
        left_points = left_points[left_points[:, 0].argsort()]
    if len(right_points) > 0:
        right_points = right_points[right_points[:, 0].argsort()]

    return BoundaryData(
        left_points=left_points,
        right_points=right_points,
        timestamp=timestamp,
        frame_id=frame_id
    )


def path_to_ros(points: np.ndarray, frame_id: str, stamp) -> Path:
    """
    Convert numpy array of path points to nav_msgs/Path.

    Args:
        points: (N, 2) or (N, 3) array of path points [x, y] or [x, y, z]
        frame_id: Coordinate frame
        stamp: ROS2 timestamp (from node.get_clock().now())

    Returns:
        nav_msgs/Path message
    """
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp.to_msg()

    for i in range(len(points)):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp.to_msg()
        pose.pose.position.x = float(points[i, 0])
        pose.pose.position.y = float(points[i, 1])
        pose.pose.position.z = float(points[i, 2]) if points.shape[1] > 2 else 0.5  # Raised for visibility

        # Compute orientation from path direction
        if i < len(points) - 1:
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            yaw = math.atan2(dy, dx)
        elif i > 0:
            dx = points[i, 0] - points[i - 1, 0]
            dy = points[i, 1] - points[i - 1, 1]
            yaw = math.atan2(dy, dx)
        else:
            yaw = 0.0

        pose.pose.orientation = yaw_to_quaternion(yaw)
        path.poses.append(pose)

    return path


def ros_path_to_numpy(path: Path) -> np.ndarray:
    """
    Convert nav_msgs/Path to numpy array.

    Args:
        path: nav_msgs/Path message

    Returns:
        (N, 2) array of path points [x, y]
    """
    if not path.poses:
        return np.empty((0, 2))

    points = np.zeros((len(path.poses), 2))
    for i, pose in enumerate(path.poses):
        points[i, 0] = pose.pose.position.x
        points[i, 1] = pose.pose.position.y

    return points


def quaternion_to_yaw(q: Quaternion) -> float:
    """
    Extract yaw angle from quaternion.

    Args:
        q: geometry_msgs/Quaternion

    Returns:
        Yaw angle in radians
    """
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Convert yaw angle to quaternion (rotation about z-axis).

    Args:
        yaw: Yaw angle in radians

    Returns:
        geometry_msgs/Quaternion
    """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
