#!/usr/bin/env python3
"""
Centerline Evaluator - Compares real-time centerline against reference.

Provides quality metrics to assess how well the real-time computed
centerline matches the pre-computed reference centerline.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Metrics from centerline comparison."""
    mean_lateral_error: float     # Average distance from reference (m)
    max_lateral_error: float      # Maximum deviation (m)
    min_lateral_error: float      # Minimum deviation (m)
    std_lateral_error: float      # Standard deviation of error
    rms_error: float              # Root mean square error
    coverage: float               # Fraction of computed points matched
    smoothness: float             # Path curvature variance (lower = smoother)
    num_computed_points: int      # Number of points in computed path
    num_matched_points: int       # Number of points successfully matched


class CenterlineEvaluator:
    """
    Evaluates real-time centerline quality against reference.

    For each point on the computed centerline, finds the nearest point
    on the reference and measures the lateral error.
    """

    def __init__(
        self,
        reference_centerline: np.ndarray,
        match_radius: float = 5.0
    ):
        """
        Initialize evaluator with reference centerline.

        Args:
            reference_centerline: (N, 2) reference path [x, y]
            match_radius: Maximum distance to consider a match (m)
        """
        self.reference = reference_centerline
        self.match_radius = match_radius

        # Pre-compute for faster lookup
        self._ref_cumulative_dist = self._compute_cumulative_distance(reference_centerline)

    def evaluate(
        self,
        computed_path: np.ndarray,
        car_x: float,
        car_y: float
    ) -> EvaluationMetrics:
        """
        Evaluate computed centerline against reference.

        Args:
            computed_path: (M, 2) computed centerline [x, y]
            car_x, car_y: Current car position (for local evaluation)

        Returns:
            EvaluationMetrics with quality measurements
        """
        if len(computed_path) < 2:
            return EvaluationMetrics(
                mean_lateral_error=float('inf'),
                max_lateral_error=float('inf'),
                min_lateral_error=float('inf'),
                std_lateral_error=float('inf'),
                rms_error=float('inf'),
                coverage=0.0,
                smoothness=float('inf'),
                num_computed_points=len(computed_path),
                num_matched_points=0
            )

        # Find lateral errors for each computed point
        lateral_errors = []
        matched_count = 0

        for point in computed_path:
            error, matched = self._find_lateral_error(point)
            if matched:
                lateral_errors.append(error)
                matched_count += 1

        if len(lateral_errors) == 0:
            return EvaluationMetrics(
                mean_lateral_error=float('inf'),
                max_lateral_error=float('inf'),
                min_lateral_error=float('inf'),
                std_lateral_error=float('inf'),
                rms_error=float('inf'),
                coverage=0.0,
                smoothness=self._compute_smoothness(computed_path),
                num_computed_points=len(computed_path),
                num_matched_points=0
            )

        errors = np.array(lateral_errors)

        return EvaluationMetrics(
            mean_lateral_error=np.mean(errors),
            max_lateral_error=np.max(errors),
            min_lateral_error=np.min(errors),
            std_lateral_error=np.std(errors),
            rms_error=np.sqrt(np.mean(errors**2)),
            coverage=matched_count / len(computed_path),
            smoothness=self._compute_smoothness(computed_path),
            num_computed_points=len(computed_path),
            num_matched_points=matched_count
        )

    def _find_lateral_error(self, point: np.ndarray) -> Tuple[float, bool]:
        """
        Find lateral error between a point and the reference path.

        Args:
            point: (2,) point [x, y]

        Returns:
            (error, matched) - lateral distance and whether match was found
        """
        # Find nearest point on reference
        distances = np.sqrt(
            (self.reference[:, 0] - point[0])**2 +
            (self.reference[:, 1] - point[1])**2
        )

        min_dist = np.min(distances)
        nearest_idx = np.argmin(distances)

        if min_dist > self.match_radius:
            return min_dist, False

        # Compute more precise lateral error using path tangent
        # Get tangent at nearest point
        if nearest_idx == 0:
            tangent = self.reference[1] - self.reference[0]
        elif nearest_idx == len(self.reference) - 1:
            tangent = self.reference[-1] - self.reference[-2]
        else:
            tangent = self.reference[nearest_idx + 1] - self.reference[nearest_idx - 1]

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            return min_dist, True

        tangent = tangent / tangent_norm

        # Vector from reference point to query point
        to_point = point - self.reference[nearest_idx]

        # Lateral error = component perpendicular to tangent
        lateral_error = abs(to_point[0] * (-tangent[1]) + to_point[1] * tangent[0])

        return lateral_error, True

    def _compute_smoothness(self, path: np.ndarray) -> float:
        """
        Compute path smoothness as curvature variance.

        Lower values indicate smoother paths.
        """
        if len(path) < 3:
            return 0.0

        # Compute curvature at each point
        curvatures = []

        for i in range(1, len(path) - 1):
            p0 = path[i - 1]
            p1 = path[i]
            p2 = path[i + 1]

            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1

            # Lengths
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)

            if l1 < 1e-6 or l2 < 1e-6:
                continue

            # Angle change
            cos_angle = np.dot(v1, v2) / (l1 * l2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            # Curvature approximation
            arc_length = (l1 + l2) / 2
            if arc_length > 1e-6:
                curvature = angle / arc_length
                curvatures.append(curvature)

        if len(curvatures) == 0:
            return 0.0

        # Smoothness = variance of curvature
        return float(np.var(curvatures))

    def _compute_cumulative_distance(self, path: np.ndarray) -> np.ndarray:
        """Compute cumulative distance along path."""
        if len(path) < 2:
            return np.array([0.0])

        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        return cumulative

    def get_reference_segment(
        self,
        car_x: float,
        car_y: float,
        forward_dist: float = 40.0
    ) -> np.ndarray:
        """
        Get reference centerline segment near the car.

        Useful for visualization comparison.

        Args:
            car_x, car_y: Car position
            forward_dist: How far ahead to include

        Returns:
            Reference centerline segment
        """
        # Find nearest point on reference
        distances = np.sqrt(
            (self.reference[:, 0] - car_x)**2 +
            (self.reference[:, 1] - car_y)**2
        )
        nearest_idx = np.argmin(distances)

        # Get segment around this point
        start_idx = max(0, nearest_idx - 10)

        # Find end index based on path distance
        accumulated = 0.0
        end_idx = nearest_idx
        for i in range(nearest_idx, len(self.reference) - 1):
            segment_len = np.linalg.norm(
                self.reference[i + 1] - self.reference[i]
            )
            accumulated += segment_len
            end_idx = i + 1
            if accumulated > forward_dist:
                break

        return self.reference[start_idx:end_idx + 1]


def format_metrics(metrics: EvaluationMetrics) -> str:
    """Format metrics as a readable string."""
    return (
        f"Lateral Error: {metrics.mean_lateral_error:.3f}m (mean), "
        f"{metrics.max_lateral_error:.3f}m (max)\n"
        f"RMS Error: {metrics.rms_error:.3f}m\n"
        f"Coverage: {metrics.coverage*100:.1f}%\n"
        f"Smoothness: {metrics.smoothness:.6f}\n"
        f"Points: {metrics.num_matched_points}/{metrics.num_computed_points} matched"
    )


if __name__ == "__main__":
    # Test the evaluator
    print("Testing CenterlineEvaluator...")

    # Create reference path (straight line)
    t = np.linspace(0, 100, 200)
    reference = np.column_stack([t, np.zeros(len(t))])

    evaluator = CenterlineEvaluator(reference)

    # Test 1: Perfect match
    computed1 = np.column_stack([t[50:150], np.zeros(100)])
    metrics1 = evaluator.evaluate(computed1, 50, 0)
    print("\nTest 1: Perfect match")
    print(format_metrics(metrics1))

    # Test 2: Small offset
    computed2 = np.column_stack([t[50:150], np.ones(100) * 0.3])
    metrics2 = evaluator.evaluate(computed2, 50, 0)
    print("\nTest 2: 0.3m offset")
    print(format_metrics(metrics2))

    # Test 3: Noisy path
    noise = np.random.normal(0, 0.5, 100)
    computed3 = np.column_stack([t[50:150], noise])
    metrics3 = evaluator.evaluate(computed3, 50, 0)
    print("\nTest 3: Noisy (std=0.5m)")
    print(format_metrics(metrics3))
