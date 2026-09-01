#!/usr/bin/env python3
"""
Local Centerline Planner - Computes centerline from edge midpoints.

Takes left and right edge points from lidar and computes the track
centerline as the midpoint between paired left/right edges.
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
    max_forward_dist: float = 40.0     # Maximum forward distance (m)
    num_slices: int = 40               # Number of distance slices for pairing
    smoothing_window: int = 7          # Final path smoothing window
    min_path_points: int = 5           # Minimum points for valid path
    safety_margin: float = 0.5         # Safety margin from edges (m)

    # Roundabout navigation parameters (optimized for Renault Zoe)
    right_bias_alpha: float = 0.45          # Normal driving: near center (0.5=center)
    track_widening_threshold: float = 1.1   # Width ratio triggering roundabout mode
    baseline_track_width: float = 6.0       # Normal track width (m)
    adaptive_bias_enabled: bool = True      # Enable adaptive right-bias
    min_track_width_for_bias: float = 2.5   # Minimum width before applying bias


@dataclass
class TrackAnalysis:
    """Analysis of current track conditions for roundabout detection."""
    track_widths: np.ndarray           # Width at each forward distance slice
    avg_width: float                   # Average track width in view
    max_width: float                   # Maximum track width detected
    is_widening: bool                  # True if track significantly widening
    widening_start_dist: float         # Forward distance where widening begins
    probable_roundabout: bool          # True if characteristics match roundabout


class LocalCenterlinePlanner:
    """
    Computes local centerline from detected track edges.

    Algorithm:
    1. Transform edges to car-relative coordinates
    2. Pair left/right edge points at same forward distances
    3. Compute midpoint of each pair
    4. Smooth and return path in world coordinates
    """

    def __init__(self, config: Optional[CenterlinePlannerConfig] = None):
        """
        Initialize the planner.

        Args:
            config: Planner configuration
        """
        self.config = config or CenterlinePlannerConfig()

    def compute(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        car_x: float,
        car_y: float,
        car_yaw: float,
        island_points: Optional[np.ndarray] = None,
        has_island_ahead: bool = False
    ) -> np.ndarray:
        """
        Compute centerline from left and right edges.

        Args:
            left_edge: (N, 2) left edge points [x, y] in world coordinates
            right_edge: (M, 2) right edge points [x, y] in world coordinates
            car_x, car_y: Car position
            car_yaw: Car heading (radians)
            island_points: (P, 2) island boundary points (optional)
            has_island_ahead: True if island detected ahead

        Returns:
            (K, 2) centerline points [x, y] in world coordinates, ordered near to far
        """
        # Handle empty inputs
        if len(left_edge) < 2 or len(right_edge) < 2:
            return self._fallback_path(left_edge, right_edge, car_x, car_y, car_yaw, has_island_ahead)

        # Transform to car frame
        left_local = self._to_car_frame(left_edge, car_x, car_y, car_yaw)
        right_local = self._to_car_frame(right_edge, car_x, car_y, car_yaw)

        # If island detected, validate it's a genuine island before using island avoidance
        use_island_avoidance = False
        if has_island_ahead and island_points is not None and len(island_points) > 0:
            island_local = self._to_car_frame(island_points, car_x, car_y, car_yaw)

            # Filter to island points ahead of car
            island_ahead = island_local[island_local[:, 0] > 2.0]

            if len(island_ahead) > 3:
                island_y_min = np.min(island_ahead[:, 1])
                island_y_max = np.max(island_ahead[:, 1])
                island_y_span = island_y_max - island_y_min

                # Get track width from edges for comparison
                left_y_mean = np.mean(left_local[:, 1]) if len(left_local) > 0 else 3.0
                right_y_mean = np.mean(right_local[:, 1]) if len(right_local) > 0 else -3.0
                track_width = left_y_mean - right_y_mean

                # Island is valid if it doesn't span full track width and has reasonable size
                is_not_full_width = island_y_span < 0.8 * track_width
                is_substantial = len(island_ahead) >= 4

                if is_not_full_width and is_substantial:
                    use_island_avoidance = True

        if use_island_avoidance:
            centerline_local = self._compute_centerline_avoiding_island(
                left_local, right_local, island_local
            )
        else:
            # Compute centerline in car frame with adaptive right-bias
            centerline_local = self._compute_centerline_local(left_local, right_local)

        if len(centerline_local) < self.config.min_path_points:
            return self._fallback_path(left_edge, right_edge, car_x, car_y, car_yaw, has_island_ahead)

        # Smooth the centerline
        centerline_local = self._smooth_path(centerline_local)

        # Transform back to world frame
        centerline_world = self._from_car_frame(centerline_local, car_x, car_y, car_yaw)

        return centerline_world

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
            # Use the union of ranges instead of intersection
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
        # Sort by x (forward distance)
        left_sorted_idx = np.argsort(left_local[:, 0])
        right_sorted_idx = np.argsort(right_local[:, 0])

        left_x = left_local[left_sorted_idx, 0]
        left_y = left_local[left_sorted_idx, 1]
        right_x = right_local[right_sorted_idx, 0]
        right_y = right_local[right_sorted_idx, 1]

        # Handle duplicate x values by averaging
        left_x, left_y = self._remove_duplicate_x(left_x, left_y)
        right_x, right_y = self._remove_duplicate_x(right_x, right_y)

        if len(left_x) < 2 or len(right_x) < 2:
            return np.array([]).reshape(0, 2)

        try:
            left_interp = interp1d(left_x, left_y, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            right_interp = interp1d(right_x, right_y, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        except ValueError:
            return np.array([]).reshape(0, 2)

        # Sample at regular forward distances
        forward_distances = np.linspace(x_min, x_max, self.config.num_slices)

        centerline_points = []
        for x_forward in forward_distances:
            # Interpolate lateral positions
            y_left = left_interp(x_forward)
            y_right = right_interp(x_forward)

            track_width = abs(y_left - y_right)

            # Check for asymmetric widening (left edge far, right edge close)
            # This is a strong indicator of roundabout approach
            left_distance = abs(y_left)  # Distance from center to left edge
            right_distance = abs(y_right)  # Distance from center to right edge
            asymmetry = left_distance / max(right_distance, 0.1)  # >1 means left is farther

            # Compute adaptive bias factor for counterclockwise navigation
            if self.config.adaptive_bias_enabled:
                # Compute width ratio relative to baseline
                width_factor = track_width / self.config.baseline_track_width

                # Detect roundabout approach: BOTH conditions required
                # - Track significantly wider than baseline (width_factor > 1.1)
                # - AND left edge farther than right (asymmetry > 1.2)
                is_roundabout_approach = (width_factor > self.config.track_widening_threshold and
                                          asymmetry > 1.2)

                if is_roundabout_approach:
                    # Roundabout approach: strong right bias
                    adaptive_alpha = 0.15
                else:
                    # Normal driving: slight right bias
                    adaptive_alpha = 0.45
            else:
                # Bias disabled: use exact center
                adaptive_alpha = 0.5

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
        """
        Compute centerline when left and right edges don't overlap in forward distance.

        Uses single-edge offset where only one edge is available, with appropriate
        bias for counterclockwise navigation.
        """
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

        # Estimate typical track width from available data
        # If we have some overlap points, use those; otherwise estimate
        estimated_track_width = self.config.baseline_track_width

        forward_distances = np.linspace(x_min, x_max, self.config.num_slices)
        centerline_points = []

        for x_forward in forward_distances:
            have_left = left_x_min <= x_forward <= left_x_max and left_interp is not None
            have_right = right_x_min <= x_forward <= right_x_max and right_interp is not None

            if have_left and have_right:
                # Both edges available
                y_left = left_interp(x_forward)
                y_right = right_interp(x_forward)
                track_width = abs(y_left - y_right)

                # Use right bias
                y_center = y_right + self.config.right_bias_alpha * (y_left - y_right)

            elif have_left:
                # Only left edge - offset to the right
                y_left = left_interp(x_forward)
                # Assume track extends to the right by estimated width
                offset_distance = estimated_track_width * (1 - self.config.right_bias_alpha)
                y_center = y_left - offset_distance
                track_width = estimated_track_width

            elif have_right:
                # Only right edge - offset toward center using configured bias
                y_right = right_interp(x_forward)
                # Use right_bias_alpha to determine position (0.35 = 35% from right toward left)
                offset_distance = estimated_track_width * self.config.right_bias_alpha
                y_center = y_right + offset_distance
                track_width = estimated_track_width

            else:
                # Neither edge - use extrapolation from nearest available data
                if left_interp is not None:
                    y_left = left_interp(x_forward)
                    y_center = y_left - estimated_track_width * (1 - self.config.right_bias_alpha)
                elif right_interp is not None:
                    y_right = right_interp(x_forward)
                    y_center = y_right + estimated_track_width * self.config.right_bias_alpha
                else:
                    continue
                track_width = estimated_track_width

            if track_width > 2 * self.config.safety_margin:
                centerline_points.append([x_forward, y_center])

        if len(centerline_points) == 0:
            return np.array([]).reshape(0, 2)

        return np.array(centerline_points)

    def _compute_centerline_avoiding_island(
        self,
        left_local: np.ndarray,
        right_local: np.ndarray,
        island_local: np.ndarray
    ) -> np.ndarray:
        """
        Compute centerline ensuring path goes RIGHT of detected island.

        For counterclockwise roundabout navigation, the centerline must
        stay between the right track edge and the island.

        Args:
            left_local: Left edge in car frame
            right_local: Right edge in car frame
            island_local: Island boundary points in car frame

        Returns:
            Centerline points biased to avoid island
        """
        # Get island boundary - find the rightmost (most negative Y) points per forward distance
        # Only consider island points that are BETWEEN left and right edges (positive Y = left side)
        # Island in roundabout should be on the LEFT (positive Y in car frame)
        island_right_boundary = {}
        for point in island_local:
            x_fwd = point[0]
            y_lat = point[1]
            # Only consider island points that are:
            # 1. Ahead of the car
            # 2. On the LEFT side (positive Y) - island should be to the left in counterclockwise navigation
            if x_fwd > self.config.min_forward_dist and y_lat > -1.0:
                x_bucket = round(x_fwd, 0)  # Bucket by 1m
                if x_bucket not in island_right_boundary:
                    island_right_boundary[x_bucket] = y_lat
                else:
                    # Keep the most negative Y (rightmost point of island) but still on left side
                    island_right_boundary[x_bucket] = min(island_right_boundary[x_bucket], y_lat)

        # Now compute centerline with very strong right bias when near island
        # Get forward distance range
        left_x_min = left_local[:, 0].min()
        left_x_max = left_local[:, 0].max()
        right_x_min = right_local[:, 0].min()
        right_x_max = right_local[:, 0].max()

        # Use UNION of ranges (not intersection) to handle non-overlapping edges
        x_min = max(min(left_x_min, right_x_min), self.config.min_forward_dist)
        x_max = min(max(left_x_max, right_x_max), self.config.max_forward_dist)

        if x_max <= x_min:
            return np.array([]).reshape(0, 2)

        # Create interpolators
        left_sorted_idx = np.argsort(left_local[:, 0])
        right_sorted_idx = np.argsort(right_local[:, 0])

        left_x = left_local[left_sorted_idx, 0]
        left_y = left_local[left_sorted_idx, 1]
        right_x = right_local[right_sorted_idx, 0]
        right_y = right_local[right_sorted_idx, 1]

        left_x, left_y = self._remove_duplicate_x(left_x, left_y)
        right_x, right_y = self._remove_duplicate_x(right_x, right_y)

        # Create interpolators with extrapolation support
        left_interp = None
        right_interp = None

        if len(left_x) >= 2:
            try:
                left_interp = interp1d(left_x, left_y, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
            except ValueError:
                # Interpolation failed - likely duplicate x values, skip left edge
                left_interp = None

        if len(right_x) >= 2:
            try:
                right_interp = interp1d(right_x, right_y, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
            except ValueError:
                # Interpolation failed - likely duplicate x values, skip right edge
                right_interp = None

        if left_interp is None and right_interp is None:
            return np.array([]).reshape(0, 2)

        forward_distances = np.linspace(x_min, x_max, self.config.num_slices)

        centerline_points = []
        for x_forward in forward_distances:
            # Check which edges are directly available (not extrapolated)
            have_left = left_x_min <= x_forward <= left_x_max and left_interp is not None
            have_right = right_x_min <= x_forward <= right_x_max and right_interp is not None

            if have_left and have_right:
                y_left = left_interp(x_forward)
                y_right = right_interp(x_forward)
                track_width = abs(y_left - y_right)
            elif have_left:
                y_left = left_interp(x_forward)
                y_right = y_left - self.config.baseline_track_width
                track_width = self.config.baseline_track_width
            elif have_right:
                y_right = right_interp(x_forward)
                y_left = y_right + self.config.baseline_track_width
                track_width = self.config.baseline_track_width
            else:
                # Extrapolate from available data
                if left_interp is not None:
                    y_left = left_interp(x_forward)
                    y_right = y_left - self.config.baseline_track_width
                elif right_interp is not None:
                    y_right = right_interp(x_forward)
                    y_left = y_right + self.config.baseline_track_width
                else:
                    continue
                track_width = self.config.baseline_track_width

            # Check if there's island boundary at this forward distance
            x_bucket = round(x_forward, 0)
            island_y = island_right_boundary.get(x_bucket, None)

            # Also check nearby buckets (island detection can be sparse)
            if island_y is None:
                for offset in [-1, 1, -2, 2]:
                    island_y = island_right_boundary.get(x_bucket + offset, None)
                    if island_y is not None:
                        break

            if island_y is not None:
                # Island detected at this distance
                # Path must stay to the RIGHT of island (more negative Y than island)
                # Stay EXTREMELY close to right edge for proper roundabout entry
                # Use 5% from right edge toward island (maximum right bias)
                y_center = y_right + 0.05 * (island_y - y_right)
            elif len(island_right_boundary) > 0:
                # Island detected ahead but not at this exact distance
                # Maximum right bias to prepare for roundabout entry
                # Use 5% from right toward left (maximum right bias)
                y_center = y_right + 0.05 * (y_left - y_right)
            else:
                # No island here: use configured right bias
                y_center = y_right + self.config.right_bias_alpha * (y_left - y_right)

            if track_width > 2 * self.config.safety_margin:
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

        # Find unique x values and average y
        unique_x = []
        unique_y = []
        processed = np.zeros(len(x), dtype=bool)

        for i in range(len(x)):
            if processed[i]:
                continue

            current_x = x[i]
            # Find all points with same x (within tolerance)
            same_x_mask = np.abs(x - current_x) < 0.01
            avg_y = np.mean(y[same_x_mask])

            unique_x.append(current_x)
            unique_y.append(avg_y)

            # Mark all matching points as processed
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
        car_yaw: float,
        has_island_ahead: bool = False
    ) -> np.ndarray:
        """
        Generate fallback path when edges are insufficient.

        Uses available edge or projects ahead. Offsets are based on
        baseline track width and configured bias.
        """
        track_width = self.config.baseline_track_width
        alpha = self.config.right_bias_alpha

        # If we have only one edge, offset from it
        if len(left_edge) >= 2 and len(right_edge) < 2:
            # Only left edge visible
            left_local = self._to_car_frame(left_edge, car_x, car_y, car_yaw)

            # Filter to only points ahead of the car
            forward_mask = left_local[:, 0] > self.config.min_forward_dist
            if np.sum(forward_mask) < 2:
                forward_mask = left_local[:, 0] > 0  # Fallback to any forward point

            if np.sum(forward_mask) >= 2:
                left_local = left_local[forward_mask]
                # Sort by forward distance
                sort_idx = np.argsort(left_local[:, 0])
                left_local = left_local[sort_idx]

                # Offset to the right (negative Y in car frame)
                # Use (1 - alpha) * track_width to stay on the right-biased side
                offset_distance = (1 - alpha) * track_width
                offset = np.array([0, -offset_distance])
                path_local = left_local + offset
                return self._from_car_frame(path_local, car_x, car_y, car_yaw)

        if len(right_edge) >= 2 and len(left_edge) < 2:
            # Only right edge visible
            right_local = self._to_car_frame(right_edge, car_x, car_y, car_yaw)

            # Filter to only points ahead of the car
            forward_mask = right_local[:, 0] > self.config.min_forward_dist
            if np.sum(forward_mask) < 2:
                forward_mask = right_local[:, 0] > 0

            if np.sum(forward_mask) >= 2:
                right_local = right_local[forward_mask]
                # Sort by forward distance
                sort_idx = np.argsort(right_local[:, 0])
                right_local = right_local[sort_idx]

                # Offset to the left (positive Y in car frame)
                # Use alpha * track_width to maintain the right bias
                offset_distance = alpha * track_width
                offset = np.array([0, offset_distance])
                path_local = right_local + offset
                return self._from_car_frame(path_local, car_x, car_y, car_yaw)

        # No usable edges - project straight ahead
        distances = np.linspace(2.0, min(15.0, self.config.max_forward_dist), 10)
        path_local = np.column_stack([distances, np.zeros_like(distances)])
        return self._from_car_frame(path_local, car_x, car_y, car_yaw)

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


if __name__ == "__main__":
    # Test the centerline planner
    print("Testing LocalCenterlinePlanner...")

    # Create sample edge data
    t = np.linspace(5, 35, 30)

    # Left edge (5m to the left, slight curve)
    left_edge = np.column_stack([
        t,
        5 + 0.005 * t**2 + np.random.normal(0, 0.2, len(t))
    ])

    # Right edge (5m to the right, slight curve)
    right_edge = np.column_stack([
        t,
        -5 - 0.005 * t**2 + np.random.normal(0, 0.2, len(t))
    ])

    # Transform to world coordinates (car facing east at origin)
    car_x, car_y, car_yaw = 0.0, 0.0, 0.0

    planner = LocalCenterlinePlanner()
    centerline = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)

    print(f"Left edge: {len(left_edge)} points")
    print(f"Right edge: {len(right_edge)} points")
    print(f"Centerline: {len(centerline)} points")

    if len(centerline) > 0:
        print(f"Centerline X range: {centerline[:, 0].min():.1f} to {centerline[:, 0].max():.1f}")
        print(f"Centerline Y range: {centerline[:, 1].min():.3f} to {centerline[:, 1].max():.3f}")
        print(f"Expected Y ≈ 0 (midpoint between +5 and -5)")
