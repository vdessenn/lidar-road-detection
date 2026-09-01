#!/usr/bin/env python3
"""
Edge Smoother - Multi-stage filtering pipeline for noisy lidar edge data.

Implements a research-based filtering pipeline:
1. RANSAC - Outlier rejection
2. Savitzky-Golay - Spatial smoothing (preserves shape)
3. Kalman Filter - Temporal smoothing across frames

References:
- pyRANSAC-3D: https://github.com/leomariga/pyRANSAC-3D
- Ego-Lane-fitting-Pointclouds: https://github.com/kymikim0401/Ego-Lane-fitting-Pointclouds
- Extended Kalman Filter for LiDAR: https://github.com/JunshengFu/tracking-with-Extended-Kalman-Filter
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class SmootherConfig:
    """Configuration for the edge smoother."""
    # RANSAC parameters
    ransac_enabled: bool = True
    ransac_threshold: float = 0.5      # Inlier threshold (meters)
    ransac_min_samples: int = 5        # Minimum samples for model
    ransac_max_trials: int = 100       # Maximum RANSAC iterations

    # Savitzky-Golay parameters
    savgol_enabled: bool = True
    savgol_window: int = 11            # Window size (must be odd)
    savgol_polyorder: int = 3          # Polynomial order

    # Kalman filter parameters
    kalman_enabled: bool = True
    kalman_process_noise: float = 0.1  # Process noise covariance Q
    kalman_measurement_noise: float = 0.5  # Measurement noise covariance R

    # General
    min_points: int = 5                # Minimum points to process


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
    Multi-stage edge smoothing pipeline.

    Applies RANSAC outlier rejection, Savitzky-Golay spatial smoothing,
    and Kalman temporal smoothing to noisy lidar edge points.
    """

    def __init__(self, config: Optional[SmootherConfig] = None):
        """
        Initialize edge smoother.

        Args:
            config: Smoother configuration
        """
        self.config = config or SmootherConfig()

        # Kalman filters for temporal smoothing (one per tracked point)
        self.kalman_left_x: List[KalmanFilter1D] = []
        self.kalman_left_y: List[KalmanFilter1D] = []
        self.kalman_right_x: List[KalmanFilter1D] = []
        self.kalman_right_y: List[KalmanFilter1D] = []

        # Track number of points for Kalman consistency
        self.prev_left_count = 0
        self.prev_right_count = 0

    def smooth_edges(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full smoothing pipeline to edge points.

        Args:
            left_edge: (N, 2) left edge points [x, y]
            right_edge: (M, 2) right edge points [x, y]
            car_x, car_y: Car position (for coordinate transform)
            car_yaw: Car heading

        Returns:
            (smoothed_left, smoothed_right) edge arrays
        """
        # Transform to car-relative coordinates for processing
        left_local = self._to_car_frame(left_edge, car_x, car_y, car_yaw) if len(left_edge) > 0 else left_edge
        right_local = self._to_car_frame(right_edge, car_x, car_y, car_yaw) if len(right_edge) > 0 else right_edge

        # Process each edge
        left_smooth = self._process_edge(left_local, 'left')
        right_smooth = self._process_edge(right_local, 'right')

        # Transform back to world coordinates
        left_world = self._from_car_frame(left_smooth, car_x, car_y, car_yaw) if len(left_smooth) > 0 else left_smooth
        right_world = self._from_car_frame(right_smooth, car_x, car_y, car_yaw) if len(right_smooth) > 0 else right_smooth

        return left_world, right_world

    def _process_edge(self, points: np.ndarray, side: str) -> np.ndarray:
        """
        Process a single edge through the filtering pipeline.

        Args:
            points: (N, 2) edge points in car-relative coordinates
            side: 'left' or 'right' (for Kalman filter state)

        Returns:
            Smoothed edge points
        """
        if len(points) < self.config.min_points:
            return points

        # Stage 1: RANSAC outlier rejection
        if self.config.ransac_enabled and len(points) >= self.config.ransac_min_samples:
            points = self._ransac_filter(points)

        if len(points) < self.config.min_points:
            return points

        # Stage 2: Savitzky-Golay spatial smoothing
        if self.config.savgol_enabled:
            points = self._savgol_filter(points)

        # Stage 3: Kalman temporal smoothing
        if self.config.kalman_enabled:
            points = self._kalman_filter(points, side)

        return points

    def _ransac_filter(self, points: np.ndarray) -> np.ndarray:
        """
        Apply RANSAC to reject outliers.

        Fits a polynomial to the edge points and rejects points
        that are too far from the fitted curve.
        """
        if len(points) < self.config.ransac_min_samples:
            return points

        # Use forward distance (x in car frame) as independent variable
        x = points[:, 0]
        y = points[:, 1]

        # Sort by x for polynomial fitting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        best_inliers = np.arange(len(points))
        best_inlier_count = 0

        for _ in range(self.config.ransac_max_trials):
            # Randomly sample points
            sample_idx = np.random.choice(len(x_sorted), self.config.ransac_min_samples, replace=False)

            try:
                # Fit polynomial to sample
                coeffs = np.polyfit(x_sorted[sample_idx], y_sorted[sample_idx], deg=2)
                y_pred = np.polyval(coeffs, x_sorted)

                # Find inliers
                residuals = np.abs(y_sorted - y_pred)
                inlier_mask = residuals < self.config.ransac_threshold
                inlier_count = np.sum(inlier_mask)

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    # Map back to original indices
                    best_inliers = sort_idx[inlier_mask]

            except np.linalg.LinAlgError:
                continue

        return points[best_inliers]

    def _savgol_filter(self, points: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to preserve edge shape.
        """
        if len(points) < self.config.savgol_window:
            # Reduce window size for small arrays
            window = len(points) if len(points) % 2 == 1 else len(points) - 1
            if window < 3:
                return points
        else:
            window = self.config.savgol_window

        polyorder = min(self.config.savgol_polyorder, window - 1)

        # Apply filter to x and y separately
        smoothed = np.zeros_like(points)
        smoothed[:, 0] = savgol_filter(points[:, 0], window, polyorder)
        smoothed[:, 1] = savgol_filter(points[:, 1], window, polyorder)

        return smoothed

    def _kalman_filter(self, points: np.ndarray, side: str) -> np.ndarray:
        """
        Apply Kalman filtering for temporal smoothing.

        Maintains state across frames to reduce jitter.
        """
        # Get appropriate Kalman filter lists
        if side == 'left':
            kf_x = self.kalman_left_x
            kf_y = self.kalman_left_y
            prev_count = self.prev_left_count
        else:
            kf_x = self.kalman_right_x
            kf_y = self.kalman_right_y
            prev_count = self.prev_right_count

        n_points = len(points)

        # Adjust number of Kalman filters if point count changed
        if n_points != prev_count:
            # Reset and recreate filters
            kf_x.clear()
            kf_y.clear()
            for _ in range(n_points):
                kf_x.append(KalmanFilter1D(self.config.kalman_process_noise,
                                          self.config.kalman_measurement_noise))
                kf_y.append(KalmanFilter1D(self.config.kalman_process_noise,
                                          self.config.kalman_measurement_noise))

        # Update count
        if side == 'left':
            self.prev_left_count = n_points
        else:
            self.prev_right_count = n_points

        # Apply Kalman filter to each point
        smoothed = np.zeros_like(points)
        for i in range(n_points):
            smoothed[i, 0] = kf_x[i].update(points[i, 0])
            smoothed[i, 1] = kf_y[i].update(points[i, 1])

        return smoothed

    def _to_car_frame(
        self,
        points: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float
    ) -> np.ndarray:
        """Transform points from world to car-relative frame."""
        if len(points) == 0:
            return points

        translated = points - np.array([car_x, car_y])
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
        """Transform points from car-relative to world frame."""
        if len(points) == 0:
            return points

        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        rotation = np.array([[cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw]])
        rotated = points @ rotation.T
        return rotated + np.array([car_x, car_y])

    def reset(self):
        """Reset all filter states (call when vehicle resets)."""
        self.kalman_left_x.clear()
        self.kalman_left_y.clear()
        self.kalman_right_x.clear()
        self.kalman_right_y.clear()
        self.prev_left_count = 0
        self.prev_right_count = 0


if __name__ == "__main__":
    # Test the edge smoother
    print("Testing EdgeSmoother...")

    # Create noisy edge data
    np.random.seed(42)

    # Left edge: curve with noise
    t = np.linspace(0, 30, 50)
    left_x = t
    left_y = 5 + 0.01 * t**2 + np.random.normal(0, 0.5, len(t))
    # Add some outliers
    left_y[10] += 5
    left_y[25] -= 4
    left_edge = np.column_stack([left_x, left_y])

    # Right edge
    right_x = t
    right_y = -5 - 0.01 * t**2 + np.random.normal(0, 0.5, len(t))
    right_edge = np.column_stack([right_x, right_y])

    smoother = EdgeSmoother()

    # Car at origin facing forward
    car_x, car_y, car_yaw = 0.0, 0.0, 0.0

    left_smooth, right_smooth = smoother.smooth_edges(
        left_edge, right_edge, car_x, car_y, car_yaw
    )

    print(f"Original left: {len(left_edge)} points")
    print(f"Smoothed left: {len(left_smooth)} points")
    print(f"Left Y std before: {np.std(left_edge[:, 1] - (5 + 0.01 * left_edge[:, 0]**2)):.3f}")
    print(f"Left Y std after: {np.std(left_smooth[:, 1] - (5 + 0.01 * left_smooth[:, 0]**2)):.3f}")
