#!/usr/bin/env python3
"""
Pure Pursuit Controller

Implements the Pure Pursuit path following algorithm for steering control.
The car follows a path by computing the steering angle to reach a lookahead
point on the path.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PurePursuitConfig:
    """Configuration for Pure Pursuit controller."""
    wheelbase_m: float = 2.5         # Vehicle wheelbase
    min_lookahead_m: float = 4.0     # Minimum lookahead distance
    max_lookahead_m: float = 15.0    # Maximum lookahead distance
    lookahead_speed_gain: float = 0.5  # Lookahead = min + gain * speed
    max_steer_rad: float = 0.40      # Maximum steering angle


class PurePursuitController:
    """
    Pure Pursuit path following controller.

    The algorithm:
    1. Find the point on the path at the lookahead distance ahead
    2. Compute the steering angle to reach that point
    3. Saturate steering to physical limits
    """

    def __init__(self, config: Optional[PurePursuitConfig] = None):
        self.config = config or PurePursuitConfig()
        self._current_path_idx = 0

    def compute_steering(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        car_speed: float,
        path_points: np.ndarray
    ) -> Tuple[float, Tuple[float, float], int]:
        """
        Compute steering angle to follow the path.

        Args:
            car_x: Car X position (m)
            car_y: Car Y position (m)
            car_yaw: Car heading (rad, 0=East, counterclockwise positive)
            car_speed: Car speed (m/s)
            path_points: (N, 2) array of path points [x, y]

        Returns:
            Tuple of:
            - steering: Steering angle (rad, positive = left)
            - target_point: (x, y) of the lookahead target
            - path_idx: Current index on the path
        """
        if len(path_points) < 2:
            return 0.0, (car_x, car_y), 0

        # Compute lookahead distance based on speed
        lookahead = self.config.min_lookahead_m + self.config.lookahead_speed_gain * car_speed
        lookahead = min(lookahead, self.config.max_lookahead_m)

        # Find nearest point on path
        nearest_idx = self._find_nearest_point(car_x, car_y, path_points)
        self._current_path_idx = nearest_idx

        # Find target point at lookahead distance
        target_point, target_idx = self._find_lookahead_point(
            car_x, car_y, nearest_idx, lookahead, path_points
        )

        # Compute steering angle
        steering = self._compute_steering_angle(
            car_x, car_y, car_yaw, target_point
        )

        return steering, target_point, nearest_idx

    def _find_nearest_point(
        self,
        car_x: float,
        car_y: float,
        path_points: np.ndarray
    ) -> int:
        """Find index of nearest point on path."""
        dx = path_points[:, 0] - car_x
        dy = path_points[:, 1] - car_y
        distances = np.sqrt(dx*dx + dy*dy)

        # Search around current index first (for efficiency)
        search_radius = 50
        start_idx = max(0, self._current_path_idx - search_radius)
        end_idx = min(len(path_points), self._current_path_idx + search_radius)

        local_min_idx = start_idx + np.argmin(distances[start_idx:end_idx])

        # If local minimum is at edge, search globally
        if local_min_idx == start_idx or local_min_idx == end_idx - 1:
            return np.argmin(distances)

        return local_min_idx

    def _find_lookahead_point(
        self,
        car_x: float,
        car_y: float,
        start_idx: int,
        lookahead_dist: float,
        path_points: np.ndarray
    ) -> Tuple[Tuple[float, float], int]:
        """
        Find the point on the path at the lookahead distance ahead.

        Uses linear interpolation between path points for smooth tracking.
        """
        n = len(path_points)
        current_dist = 0.0

        # Accumulate distance along path from start_idx
        for i in range(start_idx, n - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]

            # Distance of this segment
            segment_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            # Distance from car to start of this segment
            dist_to_p1 = np.sqrt((p1[0] - car_x)**2 + (p1[1] - car_y)**2)

            # If lookahead point is within this segment
            if dist_to_p1 + segment_dist >= lookahead_dist:
                # Interpolate within segment
                remaining = lookahead_dist - dist_to_p1
                if segment_dist > 0:
                    t = min(1.0, max(0.0, remaining / segment_dist))
                else:
                    t = 0.0
                target_x = p1[0] + t * (p2[0] - p1[0])
                target_y = p1[1] + t * (p2[1] - p1[1])
                return (target_x, target_y), i

            current_dist += segment_dist

        # If we reach the end of path, return last point
        return (path_points[-1, 0], path_points[-1, 1]), n - 1

    def _compute_steering_angle(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        target_point: Tuple[float, float]
    ) -> float:
        """
        Compute steering angle using Pure Pursuit formula.

        delta = atan2(2 * L * sin(alpha), Ld)

        where:
        - L is the wheelbase
        - alpha is the angle to the target in vehicle frame
        - Ld is the lookahead distance
        """
        # Vector from car to target in world frame
        dx = target_point[0] - car_x
        dy = target_point[1] - car_y
        lookahead_dist = np.sqrt(dx*dx + dy*dy)

        if lookahead_dist < 0.1:
            return 0.0

        # Angle to target in world frame
        angle_to_target = np.arctan2(dy, dx)

        # Angle to target in vehicle frame (alpha)
        alpha = angle_to_target - car_yaw

        # Normalize to [-pi, pi]
        while alpha > np.pi:
            alpha -= 2 * np.pi
        while alpha < -np.pi:
            alpha += 2 * np.pi

        # Pure Pursuit formula
        steering = np.arctan2(2 * self.config.wheelbase_m * np.sin(alpha), lookahead_dist)

        # Saturate steering
        steering = np.clip(steering, -self.config.max_steer_rad, self.config.max_steer_rad)

        return steering

    @property
    def current_path_index(self) -> int:
        """Get current index on the path."""
        return self._current_path_idx


if __name__ == "__main__":
    # Test Pure Pursuit controller
    print("Testing Pure Pursuit Controller...")

    # Create a simple circular path
    t = np.linspace(0, 2 * np.pi, 100)
    radius = 20
    path = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t)
    ])

    controller = PurePursuitController()

    # Simulate car at center going right
    car_x, car_y = radius, 0
    car_yaw = np.pi / 2  # Facing up (along path tangent)
    car_speed = 5.0

    steering, target, idx = controller.compute_steering(
        car_x, car_y, car_yaw, car_speed, path
    )

    print(f"Car position: ({car_x:.1f}, {car_y:.1f})")
    print(f"Car heading: {np.degrees(car_yaw):.1f}°")
    print(f"Target point: ({target[0]:.1f}, {target[1]:.1f})")
    print(f"Steering: {np.degrees(steering):.1f}°")
    print(f"Path index: {idx}")
