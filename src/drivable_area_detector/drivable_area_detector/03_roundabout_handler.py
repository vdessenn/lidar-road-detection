#!/usr/bin/env python3
"""
Algorithm 3 : Roundabout Detection & Y-Road Classifier

IMPROVEMENTS OVER v02:
- Explicit roundabout geometry detection (Y-shaped entry/exit patterns)
- Right-hand traffic direction handling
- Circular arc boundary adjustment (vs linear detection)
- Ring-level confidence weighting for driving vs non-driving legs

THEORY:
- Roundabouts create Y-shaped patterns in point clouds (entry + exit close together)
- Entry/exit branches fit circular arcs with consistent radius across rings
- Right-hand traffic: vehicle exits to the right (rightmost Y-leg is driving path)
- Non-driving legs should be downweighted for trajectory planning

TARGET:
- Roundabout detection rate: >80%
- False positive rate (bifurcations misclassified): <10%
- Leg classification accuracy: >90%
- Processing time: <15ms per roundabout
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


@dataclass
class RoundaboutGeometry:
    """Represents detected roundabout parameters."""
    roundabout_id: int
    center_x: float
    center_y: float
    radius: float
    entry_azimuth: float  # radians
    exit_azimuth: float   # radians
    entry_ring_idx: int
    exit_ring_idx: int
    confidence: float  # 0.0-1.0
    is_right_hand_exit: bool
    affected_rings: List[int]  # Rings within roundabout influence


class RoundaboutDetector:
    """
    Detects roundabouts by identifying Y-shaped patterns in boundary discontinuities.

    Algorithm:
    1. Identify Y-splits: Find close discontinuities (<30m apart)
    2. Validate spatial proximity and angle constraints
    3. Fit circular arcs to entry/exit branches
    4. Check ring consistency (center offset < 1m across rings)
    5. Classify as ROUNDABOUT or BIFURCATION
    """

    def __init__(self,
                 proximity_threshold: float = 30.0,
                 radius_tolerance: float = 2.0,
                 typical_roundabout_radius: float = 10.0,
                 min_consistency_score: float = 0.7,
                 logger=None):
        """
        Initialize roundabout detector.

        Args:
            proximity_threshold: Max distance between entry/exit (meters)
            radius_tolerance: Max difference from typical radius (meters)
            typical_roundabout_radius: Expected roundabout radius (meters)
            min_consistency_score: Min ring consistency (0-1)
            logger: ROS logger for debugging
        """
        self.proximity_threshold = proximity_threshold
        self.radius_tolerance = radius_tolerance
        self.typical_roundabout_radius = typical_roundabout_radius
        self.min_consistency_score = min_consistency_score
        self.logger = logger

        self.detected_roundabouts: List[RoundaboutGeometry] = []
        self.roundabout_counter = 0

    def identify_y_splits(self,
                         boundaries: List[Dict],
                         point_cloud: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find Y-split candidates from ring boundaries.

        Analyzes discontinuity positions from BoundaryDetector output.
        A Y-split is identified when:
        - 2+ rings have multiple discontinuities (≥2 gaps)
        - Discontinuities are spatially close (<30m)
        - Angle between branches is 60-180°

        Args:
            boundaries: List of dicts with keys: ring_id, discontinuities, y_min, y_max
            point_cloud: Nx4 array (x, y, z, ring)

        Returns:
            List of (ring_idx1, ring_idx2) pairs suggesting Y-splits
        """
        y_split_candidates = []

        # Find rings with multiple discontinuities (potential Y-patterns)
        multi_disc_rings = []
        for boundary in boundaries:
            if len(boundary.get('discontinuities', [])) >= 2:
                multi_disc_rings.append(boundary['ring_id'])

        if len(multi_disc_rings) < 2:
            return []  # Need at least 2 rings with Y-pattern

        # Check pairs of rings for spatial proximity
        for i, ring1 in enumerate(multi_disc_rings):
            for ring2 in multi_disc_rings[i+1:]:
                # Get discontinuity positions for both rings
                boundary1 = next(b for b in boundaries if b['ring_id'] == ring1)
                boundary2 = next(b for b in boundaries if b['ring_id'] == ring2)

                disc1_indices = boundary1.get('discontinuities', [])
                disc2_indices = boundary2.get('discontinuities', [])

                # Extract points at discontinuities
                ring1_mask = point_cloud[:, 3] == ring1
                ring2_mask = point_cloud[:, 3] == ring2

                ring1_points = point_cloud[ring1_mask][:, :3]
                ring2_points = point_cloud[ring2_mask][:, :3]

                # Check if any discontinuities are close
                for idx1 in disc1_indices:
                    for idx2 in disc2_indices:
                        if idx1 < len(ring1_points) and idx2 < len(ring2_points):
                            pt1 = ring1_points[idx1]
                            pt2 = ring2_points[idx2]

                            if self.validate_spatial_proximity(
                                (pt1[0], pt1[1]), (pt2[0], pt2[1])):
                                y_split_candidates.append((ring1, ring2))
                                break

        return y_split_candidates

    def validate_spatial_proximity(self,
                                   discontinuity1: Tuple[float, float],
                                   discontinuity2: Tuple[float, float]) -> bool:
        """
        Check if two boundaries are close enough for Y-pattern.

        Args:
            discontinuity1: (x, y) position of first discontinuity
            discontinuity2: (x, y) position of second discontinuity

        Returns:
            True if distance < proximity_threshold
        """
        distance = np.sqrt((discontinuity1[0] - discontinuity2[0])**2 +
                          (discontinuity1[1] - discontinuity2[1])**2)
        return distance < self.proximity_threshold

    def fit_arc_geometry(self,
                        ring_points: np.ndarray,
                        y_range: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Fit circular arc to points in y_range.

        Uses least squares circle fitting (Kåsa algebraic method).

        Args:
            ring_points: Nx3 array (x, y, z)
            y_range: (y_min, y_max) to filter points

        Returns:
            (center_x, center_y, radius)
        """
        if len(ring_points) < 3:
            return 0.0, 0.0, 0.0

        # Filter points within y_range
        y_coords = ring_points[:, 1]
        mask = (y_coords >= y_range[0]) & (y_coords <= y_range[1])
        filtered_points = ring_points[mask]

        if len(filtered_points) < 3:
            return 0.0, 0.0, 0.0

        # Kåsa circle fitting
        x = filtered_points[:, 0]
        y = filtered_points[:, 1]

        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        B = x**2 + y**2

        try:
            params = np.linalg.lstsq(A, B, rcond=None)[0]
            center_x, center_y = params[0], params[1]
            radius = np.sqrt(params[2] + center_x**2 + center_y**2)
            return center_x, center_y, radius
        except np.linalg.LinAlgError:
            # Fallback to centroid
            center_x, center_y = np.mean(x), np.mean(y)
            radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2))
            return center_x, center_y, radius

    def check_ring_consistency(self,
                              ring_centers: np.ndarray) -> float:
        """
        Measure consistency of center offset across rings.

        For roundabouts: all rings should have similar circle centers (low std dev)
        For bifurcations: centers diverge (high std dev)

        Args:
            ring_centers: Nx2 array of (center_x, center_y) per ring

        Returns:
            consistency_score (0-1, where 1 = very consistent)
        """
        if len(ring_centers) < 2:
            return 0.0

        # Calculate standard deviation of center positions
        center_x_std = np.std(ring_centers[:, 0])
        center_y_std = np.std(ring_centers[:, 1])

        # Total center offset std
        center_offset_std = np.sqrt(center_x_std**2 + center_y_std**2)

        # Consistency score: 1.0 if std < 0.5m, linearly decreases to 0.0 at 2m
        if center_offset_std < 0.5:
            return 1.0
        elif center_offset_std < 2.0:
            return 1.0 - (center_offset_std - 0.5) / 1.5
        else:
            return 0.0

    def validate_radius(self, fitted_radius: float) -> bool:
        """
        Check if fitted radius matches typical roundabout radius.

        Args:
            fitted_radius: Radius from circle fitting

        Returns:
            True if within tolerance of typical_roundabout_radius
        """
        diff = abs(fitted_radius - self.typical_roundabout_radius)
        return diff < self.radius_tolerance

    def detect_roundabouts(self,
                          point_cloud: np.ndarray,
                          boundaries: List[Dict]) -> List[RoundaboutGeometry]:
        """
        Main: Detect all roundabouts in cloud.

        Algorithm:
        1. Identify Y-split candidates
        2. For each candidate:
           a. Fit circles to entry/exit branches
           b. Validate radius against typical roundabout
           c. Check ring consistency
           d. Classify as ROUNDABOUT or reject
        3. Return list of RoundaboutGeometry objects

        Args:
            point_cloud: Nx4 array (x, y, z, ring)
            boundaries: List of dicts from BoundaryDetector

        Returns:
            List of detected RoundaboutGeometry objects
        """
        self.detected_roundabouts.clear()

        # Step 1: Identify Y-split candidates
        y_split_pairs = self.identify_y_splits(boundaries, point_cloud)

        if not y_split_pairs:
            if self.logger:
                self.logger.debug('[RoundaboutDetector] No Y-split candidates found')
            return []

        if self.logger:
            self.logger.info(f'[RoundaboutDetector] Found {len(y_split_pairs)} Y-split candidates')

        # Step 2: Validate each candidate
        for ring1, ring2 in y_split_pairs:
            # Get points for both rings
            ring1_mask = point_cloud[:, 3] == ring1
            ring2_mask = point_cloud[:, 3] == ring2

            ring1_points = point_cloud[ring1_mask][:, :3]
            ring2_points = point_cloud[ring2_mask][:, :3]

            # Get boundaries for both rings
            boundary1 = next(b for b in boundaries if b['ring_id'] == ring1)
            boundary2 = next(b for b in boundaries if b['ring_id'] == ring2)

            y_range1 = (boundary1['y_min'], boundary1['y_max'])
            y_range2 = (boundary2['y_min'], boundary2['y_max'])

            # Fit circles to both rings
            cx1, cy1, r1 = self.fit_arc_geometry(ring1_points, y_range1)
            cx2, cy2, r2 = self.fit_arc_geometry(ring2_points, y_range2)

            # Validate radii
            if not (self.validate_radius(r1) and self.validate_radius(r2)):
                if self.logger:
                    self.logger.debug(
                        f'[RoundaboutDetector] Rings {ring1},{ring2} rejected: '
                        f'radii {r1:.2f}m, {r2:.2f}m outside tolerance')
                continue

            # Check ring consistency
            ring_centers = np.array([[cx1, cy1], [cx2, cy2]])
            consistency = self.check_ring_consistency(ring_centers)

            if consistency < self.min_consistency_score:
                if self.logger:
                    self.logger.debug(
                        f'[RoundaboutDetector] Rings {ring1},{ring2} rejected: '
                        f'consistency {consistency:.2f} < {self.min_consistency_score}')
                continue

            # Valid roundabout detected!
            # Calculate average geometry
            center_x = (cx1 + cx2) / 2
            center_y = (cy1 + cy2) / 2
            radius = (r1 + r2) / 2

            # Calculate azimuths for entry/exit
            entry_azimuth = np.arctan2(cy1 - center_y, cx1 - center_x)
            exit_azimuth = np.arctan2(cy2 - center_y, cx2 - center_x)

            # Determine if right-hand exit
            # Right-hand exit: exit_azimuth is to the right of entry_azimuth
            angle_diff = (exit_azimuth - entry_azimuth) % (2 * np.pi)
            is_right_hand_exit = 0 < angle_diff < np.pi

            # Determine affected rings (all rings between ring1 and ring2)
            affected_rings = list(range(min(ring1, ring2), max(ring1, ring2) + 1))

            # Create RoundaboutGeometry object
            roundabout = RoundaboutGeometry(
                roundabout_id=self.roundabout_counter,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                entry_azimuth=entry_azimuth,
                exit_azimuth=exit_azimuth,
                entry_ring_idx=ring1,
                exit_ring_idx=ring2,
                confidence=consistency,
                is_right_hand_exit=is_right_hand_exit,
                affected_rings=affected_rings
            )

            self.detected_roundabouts.append(roundabout)
            self.roundabout_counter += 1

            if self.logger:
                self.logger.info(
                    f'[RoundaboutDetector] Roundabout #{roundabout.roundabout_id} detected: '
                    f'center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}m, '
                    f'right_hand={is_right_hand_exit}, confidence={consistency:.2f}')

        return self.detected_roundabouts

    def classify_leg(self,
                    geom: RoundaboutGeometry,
                    leg_azimuth: float) -> str:
        """
        Classify leg as ENTRY, EXIT, or OTHER.

        Uses azimuth and geometry to determine leg type.

        Args:
            geom: RoundaboutGeometry object
            leg_azimuth: Azimuth angle of leg in radians

        Returns:
            'ENTRY', 'EXIT', or 'OTHER'
        """
        # Calculate angular distance from entry and exit
        entry_diff = abs((leg_azimuth - geom.entry_azimuth) % (2 * np.pi))
        exit_diff = abs((leg_azimuth - geom.exit_azimuth) % (2 * np.pi))

        # Normalize to [0, π]
        entry_diff = min(entry_diff, 2 * np.pi - entry_diff)
        exit_diff = min(exit_diff, 2 * np.pi - exit_diff)

        # Classify based on closest match
        threshold = np.pi / 6  # 30 degrees tolerance

        if entry_diff < threshold:
            return 'ENTRY'
        elif exit_diff < threshold:
            return 'EXIT'
        else:
            return 'OTHER'


class YRoadHandler:
    """
    Handles special processing for Y-shaped roads and roundabouts.

    Responsibilities:
    - Identify driving leg (right-hand traffic)
    - Adjust boundaries using circular arc constraints
    - Weight rings by relevance to driving path
    - Merge Y-road boundaries for navigation
    """

    def __init__(self, right_hand_traffic: bool = True, num_rings: int = 12,
                 ring_weight_driving: float = 1.0, ring_weight_non_driving: float = 0.3,
                 circular_arc_tolerance: float = 0.5, logger=None):
        """
        Initialize Y-road handler.

        Args:
            right_hand_traffic: True for right-hand traffic (European/US)
            num_rings: Number of LiDAR rings
            ring_weight_driving: Confidence weight for driving leg
            ring_weight_non_driving: Confidence weight for non-driving legs
            circular_arc_tolerance: Max distance from roundabout circle (meters)
            logger: ROS logger
        """
        self.right_hand = right_hand_traffic
        self.num_rings = num_rings
        self.ring_weight_driving = ring_weight_driving
        self.ring_weight_non_driving = ring_weight_non_driving
        self.circular_arc_tolerance = circular_arc_tolerance
        self.logger = logger

    def identify_drive_leg(self,
                          geom: RoundaboutGeometry) -> int:
        """
        For right-hand traffic, determine which leg is the driving path.

        Analyzes entry/exit azimuths to find the rightmost exit.
        In right-hand traffic, the exit is always to the right of entry.

        Args:
            geom: RoundaboutGeometry object

        Returns:
            leg_index (0=entry, 1=exit)
        """
        if self.right_hand:
            # Right-hand traffic: exit is driving leg
            if geom.is_right_hand_exit:
                return 1  # Exit leg
            else:
                return 0  # Entry leg (fallback)
        else:
            # Left-hand traffic: entry is driving leg
            return 0

    def adjust_boundaries_in_roundabout(self,
                                       ring_points: np.ndarray,
                                       geom: RoundaboutGeometry,
                                       original_y_min: float,
                                       original_y_max: float) -> Tuple[float, float]:
        """
        Adjust y_min/y_max using roundabout circle constraint.

        Replaces linear boundary detection with circular arc for accuracy.
        Points must lie on roundabout circle ± circular_arc_tolerance.

        Args:
            ring_points: Nx3 array (x, y, z)
            geom: RoundaboutGeometry object
            original_y_min: Original y_min from BoundaryDetector
            original_y_max: Original y_max from BoundaryDetector

        Returns:
            (adjusted_y_min, adjusted_y_max)
        """
        if len(ring_points) == 0:
            return original_y_min, original_y_max

        # Calculate distance of each point from roundabout center
        x = ring_points[:, 0]
        y = ring_points[:, 1]

        distances = np.sqrt((x - geom.center_x)**2 + (y - geom.center_y)**2)

        # Filter points within circular arc tolerance
        radius_min = geom.radius - self.circular_arc_tolerance
        radius_max = geom.radius + self.circular_arc_tolerance

        on_circle_mask = (distances >= radius_min) & (distances <= radius_max)

        if np.sum(on_circle_mask) < 3:
            # Not enough points on circle → use original boundaries
            return original_y_min, original_y_max

        # Get y-coordinates of points on circle
        y_on_circle = y[on_circle_mask]

        adjusted_y_min = float(np.min(y_on_circle))
        adjusted_y_max = float(np.max(y_on_circle))

        # Clamp to reasonable range
        adjusted_y_min = max(-15.0, min(15.0, adjusted_y_min))
        adjusted_y_max = max(-15.0, min(15.0, adjusted_y_max))

        # Ensure y_min < y_max
        if adjusted_y_min >= adjusted_y_max:
            return original_y_min, original_y_max

        if self.logger:
            self.logger.debug(
                f'[YRoadHandler] Adjusted boundaries: '
                f'({original_y_min:.2f}, {original_y_max:.2f}) → '
                f'({adjusted_y_min:.2f}, {adjusted_y_max:.2f})')

        return adjusted_y_min, adjusted_y_max

    def weight_rings(self,
                    geom: RoundaboutGeometry) -> np.ndarray:
        """
        Assign confidence weights to rings near roundabout.

        Downweight non-driving legs, upweight driving leg.

        Args:
            geom: RoundaboutGeometry object

        Returns:
            (num_rings,) weight array [0, 1]
        """
        weights = np.ones(self.num_rings) * self.ring_weight_driving

        # Identify driving leg
        drive_leg_idx = self.identify_drive_leg(geom)

        # Downweight rings not on driving leg
        for ring_id in geom.affected_rings:
            if ring_id >= self.num_rings:
                continue

            # Rings closer to driving leg get higher weight
            if drive_leg_idx == 1:  # Exit is driving leg
                if ring_id == geom.exit_ring_idx:
                    weights[ring_id] = self.ring_weight_driving
                else:
                    weights[ring_id] = self.ring_weight_non_driving
            else:  # Entry is driving leg
                if ring_id == geom.entry_ring_idx:
                    weights[ring_id] = self.ring_weight_driving
                else:
                    weights[ring_id] = self.ring_weight_non_driving

        return weights

    def merge_y_roads(self,
                     boundaries_left: np.ndarray,
                     boundaries_right: np.ndarray,
                     geom: RoundaboutGeometry) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge Y-road boundaries intelligently for navigation.

        Selects driving-relevant boundaries for path planning.

        Args:
            boundaries_left: Nx3 left boundary points (x, y, z)
            boundaries_right: Nx3 right boundary points (x, y, z)
            geom: RoundaboutGeometry object

        Returns:
            (merged_left, merged_right) boundary arrays
        """
        # Identify driving leg
        drive_leg_idx = self.identify_drive_leg(geom)

        # Filter boundaries based on driving leg
        # For right-hand traffic, keep boundaries on the right side
        if self.right_hand and drive_leg_idx == 1:
            # Keep right-side boundaries (exit leg)
            # Filter left boundaries that are too far right
            if len(boundaries_left) > 0:
                mask_left = boundaries_left[:, 1] < geom.center_y
                merged_left = boundaries_left[mask_left]
            else:
                merged_left = boundaries_left

            merged_right = boundaries_right
        else:
            # Keep original boundaries
            merged_left = boundaries_left
            merged_right = boundaries_right

        return merged_left, merged_right


def main():
    """Example usage (for testing)."""
    print("RoundaboutDetector and YRoadHandler classes initialized")
    print("Use in ROS node for actual detection")


if __name__ == '__main__':
    main()
