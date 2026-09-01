#!/usr/bin/env python3
"""
Local Centerline Planner - Computes centerline from edge midpoints.

Adapted from circuit_navigator_realtime/planning/local_centerline.py
for use with ROS2 boundary data.

Takes left and right edge points and computes the track centerline
as the midpoint between paired left/right edges, with adaptive
right-bias for roundabout navigation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


@dataclass
class CenterlinePlannerConfig:
    """Configuration for local centerline planner."""
    min_forward_dist: float = 2.0      # Minimum forward distance to consider (m)
    max_forward_dist: float = 20.0     # Maximum forward distance (m)
    num_slices: int = 40               # Number of distance slices for pairing
    smoothing_window: int = 7          # Final path smoothing window
    min_path_points: int = 5           # Minimum points for valid path
    safety_margin: float = 0.5         # Safety margin from edges (m)

    # Roundabout navigation parameters
    right_bias_alpha: float = 0.45          # Normal driving: near center (0.5=center)
    track_widening_threshold: float = 1.1   # Width ratio triggering roundabout mode
    baseline_track_width: float = 6.0       # Normal track width (m)
    adaptive_bias_enabled: bool = True      # Enable adaptive right-bias
    min_track_width_for_bias: float = 2.5   # Minimum width before applying bias


class LocalCenterlinePlanner:
    """
    Computes local centerline from detected track edges.

    Algorithm:
    1. Pair left/right edge points at same forward distances
    2. Compute midpoint of each pair with adaptive bias
    3. Smooth and return path

    Note: Unlike the original, this version works in vehicle frame
    (boundaries already relative to vehicle).
    """

    def __init__(self, config: Optional[CenterlinePlannerConfig] = None):
        """Initialize the planner."""
        self.config = config or CenterlinePlannerConfig()

    def _detect_gaps(self, x_values: np.ndarray, gap_threshold: float = 2.0) -> list:
        """
        Détecte les discontinuités dans une frontière.

        Args:
            x_values: Valeurs X triées de la frontière
            gap_threshold: Seuil de distance pour considérer un gap (m)

        Returns:
            Liste de tuples (x_start_gap, x_end_gap) pour chaque gap détecté
        """
        if len(x_values) < 2:
            return []
        x_sorted = np.sort(x_values)
        gaps = np.diff(x_sorted)
        gap_indices = np.where(gaps > gap_threshold)[0]
        return [(x_sorted[i], x_sorted[i+1]) for i in gap_indices]

    def compute(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        car_x: float = 0.0,
        car_y: float = 0.0,
        car_yaw: float = 0.0
    ) -> np.ndarray:
        """
        Compute centerline from left and right edges.

        Args:
            left_edge: (N, 2) left edge points [x, y]
            right_edge: (M, 2) right edge points [x, y]
            car_x, car_y: Car position (typically 0,0 for vehicle frame)
            car_yaw: Car heading (typically 0 for vehicle frame)

        Returns:
            (K, 2) centerline points [x, y], ordered near to far
        """
        # Handle empty inputs
        if len(left_edge) < 2 or len(right_edge) < 2:
            return self._fallback_path(left_edge, right_edge, car_x, car_y, car_yaw)

        # Transform to car frame if needed
        if car_x != 0 or car_y != 0 or car_yaw != 0:
            left_local = self._to_car_frame(left_edge, car_x, car_y, car_yaw)
            right_local = self._to_car_frame(right_edge, car_x, car_y, car_yaw)
        else:
            left_local = left_edge
            right_local = right_edge

        # Compute centerline in car frame with adaptive right-bias
        centerline_local = self._compute_centerline_local(left_local, right_local)

        if len(centerline_local) < self.config.min_path_points:
            return self._fallback_path(left_edge, right_edge, car_x, car_y, car_yaw)

        # Smooth the centerline
        centerline_local = self._smooth_path(centerline_local)

        # Transform back to world frame if needed
        if car_x != 0 or car_y != 0 or car_yaw != 0:
            centerline = self._from_car_frame(centerline_local, car_x, car_y, car_yaw)
        else:
            centerline = centerline_local

        return centerline

    def _compute_centerline_local(
        self,
        left_local: np.ndarray,
        right_local: np.ndarray
    ) -> np.ndarray:
        """
        Compute centerline in car-relative coordinates with adaptive right-bias.

        Car frame: origin at car, X forward, Y left.
        In car frame: left edge has positive Y, right edge has negative Y.
        For counterclockwise roundabout navigation, we bias toward right (negative Y).
        """
        # Get forward distance range
        left_x_min = left_local[:, 0].min()
        left_x_max = left_local[:, 0].max()
        right_x_min = right_local[:, 0].min()
        right_x_max = right_local[:, 0].max()

        # Check for overlapping forward range
        x_min_strict = max(left_x_min, right_x_min, self.config.min_forward_dist)
        x_max_strict = min(left_x_max, right_x_max, self.config.max_forward_dist)

        if x_max_strict <= x_min_strict:
            # No overlap - use extended approach with single-edge fallback
            x_min = max(min(left_x_min, right_x_min), self.config.min_forward_dist)
            x_max = min(max(left_x_max, right_x_max), self.config.max_forward_dist)

            if x_max <= x_min:
                return np.array([]).reshape(0, 2)

            return self._compute_centerline_with_gaps(
                left_local, right_local, x_min, x_max,
                left_x_min, left_x_max, right_x_min, right_x_max
            )

        # Normal case: edges overlap
        x_min = x_min_strict
        x_max = x_max_strict

        # Create interpolators for left and right edges
        left_sorted_idx = np.argsort(left_local[:, 0])
        right_sorted_idx = np.argsort(right_local[:, 0])

        left_x = left_local[left_sorted_idx, 0]
        left_y = left_local[left_sorted_idx, 1]
        right_x = right_local[right_sorted_idx, 0]
        right_y = right_local[right_sorted_idx, 1]

        # Handle duplicate x values
        left_x, left_y = self._remove_duplicate_x(left_x, left_y)
        right_x, right_y = self._remove_duplicate_x(right_x, right_y)

        if len(left_x) < 2 or len(right_x) < 2:
            return np.array([]).reshape(0, 2)

        # Detect gaps in boundaries (roundabout island detection)
        left_gaps = self._detect_gaps(left_x, gap_threshold=2.0)
        right_gaps = self._detect_gaps(right_x, gap_threshold=2.0)

        # If gap detected on left side, limit x_max to gap start
        if left_gaps:
            first_gap_start = left_gaps[0][0]
            if first_gap_start > x_min and first_gap_start < x_max:
                x_max = first_gap_start

        # If gap detected on right side, also limit
        if right_gaps:
            first_gap_start = right_gaps[0][0]
            if first_gap_start > x_min and first_gap_start < x_max:
                x_max = min(x_max, first_gap_start)

        if x_max <= x_min:
            return np.array([]).reshape(0, 2)

        try:
            # Use np.nan instead of 'extrapolate' to avoid invalid extrapolation
            left_interp = interp1d(left_x, left_y, kind='linear',
                                   bounds_error=False, fill_value=np.nan)
            right_interp = interp1d(right_x, right_y, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
        except ValueError:
            return np.array([]).reshape(0, 2)

        # Sample at regular forward distances
        forward_distances = np.linspace(x_min, x_max, self.config.num_slices)

        centerline_points = []
        for x_forward in forward_distances:
            # Interpolate lateral positions
            y_left = left_interp(x_forward)
            y_right = right_interp(x_forward)

            # Skip if interpolation returned NaN (outside boundary range)
            if np.isnan(y_left) or np.isnan(y_right):
                continue

            track_width = abs(y_left - y_right)

            # Check for asymmetric widening
            left_distance = abs(y_left)
            right_distance = abs(y_right)
            asymmetry = left_distance / max(right_distance, 0.1)

            # Compute adaptive bias factor
            if self.config.adaptive_bias_enabled:
                width_factor = track_width / self.config.baseline_track_width

                # Detect roundabout approach
                is_roundabout_approach = (
                    width_factor > self.config.track_widening_threshold and
                    asymmetry > 1.2
                )

                if is_roundabout_approach:
                    adaptive_alpha = 0.15  # Strong right bias
                else:
                    adaptive_alpha = self.config.right_bias_alpha  # Normal bias
            else:
                adaptive_alpha = 0.5  # Exact center

            y_center = y_right + adaptive_alpha * (y_left - y_right)

            # Apply safety margin check
            if track_width > 2 * self.config.safety_margin:
                centerline_points.append([x_forward, y_center])

        if len(centerline_points) == 0:
            return np.array([]).reshape(0, 2)

        return np.array(centerline_points)

    def _compute_centerline_with_gaps(
        self,
        left_local: np.ndarray,
        right_local: np.ndarray,
        x_min: float,
        x_max: float,
        left_x_min: float,
        left_x_max: float,
        right_x_min: float,
        right_x_max: float
    ) -> np.ndarray:
        """Compute centerline when left and right edges don't overlap."""
        # Sort edges
        left_sorted_idx = np.argsort(left_local[:, 0])
        right_sorted_idx = np.argsort(right_local[:, 0])

        left_x = left_local[left_sorted_idx, 0]
        left_y = left_local[left_sorted_idx, 1]
        right_x = right_local[right_sorted_idx, 0]
        right_y = right_local[right_sorted_idx, 1]

        # Remove duplicates
        left_x, left_y = self._remove_duplicate_x(left_x, left_y)
        right_x, right_y = self._remove_duplicate_x(right_x, right_y)

        if len(left_x) < 2 and len(right_x) < 2:
            return np.array([]).reshape(0, 2)

        # Create interpolators with extrapolation
        left_interp = None
        right_interp = None

        if len(left_x) >= 2:
            left_interp = interp1d(left_x, left_y, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')

        if len(right_x) >= 2:
            right_interp = interp1d(right_x, right_y, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')

        # Sample at regular forward distances
        forward_distances = np.linspace(x_min, x_max, self.config.num_slices)

        centerline_points = []
        for x_forward in forward_distances:
            # Determine which edges are available at this distance
            left_available = left_interp is not None and left_x_min <= x_forward <= left_x_max
            right_available = right_interp is not None and right_x_min <= x_forward <= right_x_max

            if left_available and right_available:
                # Both edges available - normal midpoint
                y_left = left_interp(x_forward)
                y_right = right_interp(x_forward)
                y_center = y_right + self.config.right_bias_alpha * (y_left - y_right)
            elif left_available:
                # Only left edge - offset right
                y_left = left_interp(x_forward)
                offset = (1 - self.config.right_bias_alpha) * self.config.baseline_track_width
                y_center = y_left - offset
            elif right_available:
                # Only right edge - offset left
                y_right = right_interp(x_forward)
                offset = self.config.right_bias_alpha * self.config.baseline_track_width
                y_center = y_right + offset
            else:
                continue

            centerline_points.append([x_forward, y_center])

        if len(centerline_points) == 0:
            return np.array([]).reshape(0, 2)

        return np.array(centerline_points)

    def _remove_duplicate_x(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicate x values by averaging corresponding y values."""
        if len(x) == 0:
            return x, y

        unique_x = []
        unique_y = []
        processed = np.zeros(len(x), dtype=bool)

        for i in range(len(x)):
            if processed[i]:
                continue

            current_x = x[i]
            same_x_mask = np.abs(x - current_x) < 0.01
            avg_y = np.mean(y[same_x_mask])

            unique_x.append(current_x)
            unique_y.append(avg_y)

            processed[same_x_mask] = True

        return np.array(unique_x), np.array(unique_y)

    def _smooth_path(self, path: np.ndarray) -> np.ndarray:
        """Apply smoothing to the computed path."""
        if len(path) < self.config.smoothing_window:
            return path

        window = self.config.smoothing_window
        if window % 2 == 0:
            window -= 1

        if window < 3:
            return path

        smoothed = np.zeros_like(path)
        smoothed[:, 0] = savgol_filter(path[:, 0], window, 2)
        smoothed[:, 1] = savgol_filter(path[:, 1], window, 2)

        return smoothed

    def _fallback_path(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float
    ) -> np.ndarray:
        """Generate fallback path when edges are insufficient."""
        track_width = self.config.baseline_track_width
        alpha = self.config.right_bias_alpha

        # Transform to car frame if needed
        if car_x != 0 or car_y != 0 or car_yaw != 0:
            left_local = self._to_car_frame(left_edge, car_x, car_y, car_yaw) if len(left_edge) > 0 else left_edge
            right_local = self._to_car_frame(right_edge, car_x, car_y, car_yaw) if len(right_edge) > 0 else right_edge
        else:
            left_local = left_edge
            right_local = right_edge

        # If only one edge, offset from it
        if len(left_local) >= 2 and len(right_local) < 2:
            forward_mask = left_local[:, 0] > self.config.min_forward_dist
            if np.sum(forward_mask) < 2:
                forward_mask = left_local[:, 0] > 0

            if np.sum(forward_mask) >= 2:
                left_filtered = left_local[forward_mask]
                sort_idx = np.argsort(left_filtered[:, 0])
                left_filtered = left_filtered[sort_idx]

                offset_distance = (1 - alpha) * track_width
                offset = np.array([0, -offset_distance])
                path_local = left_filtered + offset

                if car_x != 0 or car_y != 0 or car_yaw != 0:
                    return self._from_car_frame(path_local, car_x, car_y, car_yaw)
                return path_local

        if len(right_local) >= 2 and len(left_local) < 2:
            forward_mask = right_local[:, 0] > self.config.min_forward_dist
            if np.sum(forward_mask) < 2:
                forward_mask = right_local[:, 0] > 0

            if np.sum(forward_mask) >= 2:
                right_filtered = right_local[forward_mask]
                sort_idx = np.argsort(right_filtered[:, 0])
                right_filtered = right_filtered[sort_idx]

                offset_distance = alpha * track_width
                offset = np.array([0, offset_distance])
                path_local = right_filtered + offset

                if car_x != 0 or car_y != 0 or car_yaw != 0:
                    return self._from_car_frame(path_local, car_x, car_y, car_yaw)
                return path_local

        # No usable edges - project straight ahead
        distances = np.linspace(2.0, min(15.0, self.config.max_forward_dist), 10)
        path_local = np.column_stack([distances, np.zeros_like(distances)])

        if car_x != 0 or car_y != 0 or car_yaw != 0:
            return self._from_car_frame(path_local, car_x, car_y, car_yaw)
        return path_local

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

    def compute_curvature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute curvature at each point of the path.

        Curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

        Args:
            path: (N, 2) array of path points [x, y]

        Returns:
            (N,) array of curvature values (1/m)
        """
        if len(path) < 3:
            return np.zeros(len(path))

        # Compute first derivatives (velocity)
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])

        # Compute second derivatives (acceleration)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5

        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        valid = denominator > 1e-6
        curvature[valid] = numerator[valid] / denominator[valid]

        return curvature
