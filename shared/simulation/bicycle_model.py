#!/usr/bin/env python3
"""
Kinematic Bicycle Model

Implements a simple kinematic bicycle model for vehicle simulation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Vehicle state."""
    x: float = 0.0          # X position (m)
    y: float = 0.0          # Y position (m)
    yaw: float = 0.0        # Heading (rad, 0=East, counterclockwise positive)
    speed: float = 0.0      # Speed (m/s)
    steering: float = 0.0   # Steering angle (rad)


class BicycleModel:
    """
    Kinematic bicycle model for vehicle simulation.

    The model assumes:
    - Front-wheel steering
    - No slip (kinematic model)
    - Constant speed over each time step
    """

    def __init__(
        self,
        wheelbase: float = 2.5,
        max_steer: float = 0.40,
        max_speed: float = 10.0,
        max_accel: float = 3.0,
        max_decel: float = 5.0
    ):
        """
        Args:
            wheelbase: Distance between front and rear axles (m)
            max_steer: Maximum steering angle (rad)
            max_speed: Maximum speed (m/s)
            max_accel: Maximum acceleration (m/s^2)
            max_decel: Maximum deceleration (m/s^2)
        """
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel

    def update(
        self,
        state: VehicleState,
        steering_cmd: float,
        speed_cmd: float,
        dt: float
    ) -> VehicleState:
        """
        Update vehicle state using kinematic bicycle model.

        Kinematic equations:
            x_dot = v * cos(yaw)
            y_dot = v * sin(yaw)
            yaw_dot = v * tan(delta) / L

        Args:
            state: Current vehicle state
            steering_cmd: Commanded steering angle (rad)
            speed_cmd: Commanded speed (m/s)
            dt: Time step (s)

        Returns:
            New vehicle state
        """
        # Apply limits
        steering = np.clip(steering_cmd, -self.max_steer, self.max_steer)
        target_speed = np.clip(speed_cmd, 0, self.max_speed)

        # Apply acceleration limits
        speed_diff = target_speed - state.speed
        if speed_diff > 0:
            # Accelerating
            max_delta = self.max_accel * dt
            speed = state.speed + min(speed_diff, max_delta)
        else:
            # Decelerating
            max_delta = self.max_decel * dt
            speed = state.speed + max(speed_diff, -max_delta)

        speed = np.clip(speed, 0, self.max_speed)

        # Kinematic bicycle model update
        if abs(steering) < 1e-6:
            # Straight line motion
            x = state.x + speed * np.cos(state.yaw) * dt
            y = state.y + speed * np.sin(state.yaw) * dt
            yaw = state.yaw
        else:
            # Curved motion
            yaw_rate = speed * np.tan(steering) / self.wheelbase
            yaw = state.yaw + yaw_rate * dt

            # Normalize yaw to [-pi, pi]
            while yaw > np.pi:
                yaw -= 2 * np.pi
            while yaw < -np.pi:
                yaw += 2 * np.pi

            # Position update using average heading
            avg_yaw = state.yaw + yaw_rate * dt / 2
            x = state.x + speed * np.cos(avg_yaw) * dt
            y = state.y + speed * np.sin(avg_yaw) * dt

        return VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            steering=steering
        )


if __name__ == "__main__":
    # Test bicycle model
    print("Testing Bicycle Model...")

    model = BicycleModel()
    state = VehicleState(x=0, y=0, yaw=0, speed=0)

    print(f"Initial: x={state.x:.2f}, y={state.y:.2f}, yaw={np.degrees(state.yaw):.1f}°")

    # Accelerate straight
    for i in range(10):
        state = model.update(state, steering_cmd=0, speed_cmd=5.0, dt=0.1)
    print(f"After 1s straight: x={state.x:.2f}, y={state.y:.2f}, speed={state.speed:.1f}")

    # Turn left
    for i in range(20):
        state = model.update(state, steering_cmd=0.2, speed_cmd=5.0, dt=0.1)
    print(f"After 2s left turn: x={state.x:.2f}, y={state.y:.2f}, yaw={np.degrees(state.yaw):.1f}°")
