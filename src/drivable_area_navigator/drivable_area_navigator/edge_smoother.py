#!/usr/bin/env python3
"""
Edge Smoother - Simplified Kalman temporal filtering for boundary data.

Since the upstream Node 02 (ring_boundary_extraction) already applies:
- RANSAC polynomial fitting
- Geometric constraints
- Temporal smoothing

This smoother only applies lightweight Kalman filtering to reduce
frame-to-frame jitter in the centerline computation.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .types import BoundaryData


@dataclass
class SmootherConfig:
    """Configuration for the edge smoother."""
    kalman_enabled: bool = True
    kalman_process_noise: float = 0.1      # Process noise covariance Q
    kalman_measurement_noise: float = 0.5  # Measurement noise covariance R
    min_points: int = 3                    # Minimum points to process
    max_tracked_points: int = 100          # Maximum points to track with Kalman


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing position over time."""

    def __init__(self, Q: float = 0.1, R: float = 0.5):
        """
        Initialize Kalman filter.

        Args:
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        self.Q = Q
        self.R = R
        self.x = None  # State estimate
        self.P = 1.0   # Error covariance

    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = 1.0

    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement.

        Args:
            measurement: New observed value

        Returns:
            Filtered estimate
        """
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


class EdgeSmoother:
    """
    Simplified edge smoothing with Kalman temporal filtering.

    Designed for boundary data that has already been filtered upstream.
    Applies only lightweight Kalman filtering to reduce jitter.
    """

    def __init__(self, config: Optional[SmootherConfig] = None):
        """
        Initialize edge smoother.

        Args:
            config: Smoother configuration
        """
        self.config = config or SmootherConfig()

        # Kalman filters for temporal smoothing
        # One filter per (x, y, z) coordinate per boundary point
        self._left_filters: List[Tuple[KalmanFilter1D, KalmanFilter1D, KalmanFilter1D]] = []
        self._right_filters: List[Tuple[KalmanFilter1D, KalmanFilter1D, KalmanFilter1D]] = []

        self._prev_left_count = 0
        self._prev_right_count = 0

    def reset(self):
        """Reset all Kalman filter states."""
        self._left_filters = []
        self._right_filters = []
        self._prev_left_count = 0
        self._prev_right_count = 0

    def smooth(self, boundaries: BoundaryData) -> BoundaryData:
        """
        Apply Kalman smoothing to boundary data.

        Args:
            boundaries: Raw boundary data from detector

        Returns:
            Smoothed boundary data
        """
        if not self.config.kalman_enabled:
            return boundaries

        left_smooth = self._smooth_edge(
            boundaries.left_points,
            self._left_filters,
            self._prev_left_count,
            'left'
        )
        self._prev_left_count = len(boundaries.left_points)

        right_smooth = self._smooth_edge(
            boundaries.right_points,
            self._right_filters,
            self._prev_right_count,
            'right'
        )
        self._prev_right_count = len(boundaries.right_points)

        return BoundaryData(
            left_points=left_smooth,
            right_points=right_smooth,
            timestamp=boundaries.timestamp,
            frame_id=boundaries.frame_id
        )

    def _smooth_edge(
        self,
        points: np.ndarray,
        filters: List[Tuple[KalmanFilter1D, KalmanFilter1D, KalmanFilter1D]],
        prev_count: int,
        side: str
    ) -> np.ndarray:
        """
        Apply Kalman smoothing to a single edge.

        Args:
            points: (N, 3) boundary points [x, y, z]
            filters: List of Kalman filter tuples for this edge
            prev_count: Previous number of points
            side: 'left' or 'right' (for logging)

        Returns:
            Smoothed points array
        """
        if len(points) < self.config.min_points:
            return points

        n_points = min(len(points), self.config.max_tracked_points)

        # Handle filter count changes
        if n_points != prev_count:
            # Reset filters when point count changes significantly
            filters.clear()

        # Initialize filters if needed
        while len(filters) < n_points:
            Q = self.config.kalman_process_noise
            R = self.config.kalman_measurement_noise
            filters.append((
                KalmanFilter1D(Q, R),
                KalmanFilter1D(Q, R),
                KalmanFilter1D(Q, R)
            ))

        # Apply Kalman filtering
        smoothed = np.zeros_like(points[:n_points])
        for i in range(n_points):
            fx, fy, fz = filters[i]
            smoothed[i, 0] = fx.update(points[i, 0])
            smoothed[i, 1] = fy.update(points[i, 1])
            smoothed[i, 2] = fz.update(points[i, 2])

        # Append any remaining points without filtering
        if len(points) > n_points:
            smoothed = np.vstack([smoothed, points[n_points:]])

        return smoothed

    def update_config(
        self,
        kalman_enabled: Optional[bool] = None,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None
    ):
        """
        Update configuration at runtime.

        Args:
            kalman_enabled: Enable/disable Kalman filtering
            process_noise: New process noise value Q
            measurement_noise: New measurement noise value R
        """
        if kalman_enabled is not None:
            self.config.kalman_enabled = kalman_enabled

        if process_noise is not None or measurement_noise is not None:
            # Update noise values and reset filters
            if process_noise is not None:
                self.config.kalman_process_noise = process_noise
            if measurement_noise is not None:
                self.config.kalman_measurement_noise = measurement_noise
            self.reset()
