#!/usr/bin/env python3
"""
Comprehensive test suite for RoundaboutDetector and YRoadHandler classes
Tests roundabout detection, Y-split classification, and boundary adjustment

Test cases:
1. test_simple_roundabout_entry_exit() - Canonical Y-shape
2. test_no_roundabout_true_bifurcation() - Should reject Y-shapes that are real forks
3. test_drive_leg_identification() - Right-hand traffic direction (exit to right)
4. test_circle_fitting_roundabout() - Circle fitting on entry/exit arcs
5. test_ring_consistency_check() - Center offset validation across rings
6. test_multiple_sequential_roundabouts() - Handling 2+ roundabouts in sequence
7. test_y_road_boundary_adjustment() - Circular vs linear boundary detection
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'drivable_area_detector'))

# Import roundabout handler module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "roundabout_handler",
    Path(__file__).parent.parent / 'drivable_area_detector' / '03_roundabout_handler.py'
)
roundabout_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roundabout_module)

RoundaboutDetector = roundabout_module.RoundaboutDetector
YRoadHandler = roundabout_module.YRoadHandler
RoundaboutGeometry = roundabout_module.RoundaboutGeometry


class TestRoundaboutHandler(unittest.TestCase):
    """Test suite for RoundaboutDetector and YRoadHandler classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = RoundaboutDetector(
            proximity_threshold=30.0,
            radius_tolerance=2.0,
            typical_roundabout_radius=10.0,
            min_consistency_score=0.7,
            logger=None
        )

        self.handler = YRoadHandler(
            right_hand_traffic=True,
            num_rings=12,
            ring_weight_driving=1.0,
            ring_weight_non_driving=0.3,
            circular_arc_tolerance=0.5,
            logger=None
        )

    # =========================================================================
    # Test 1: Simple Roundabout Entry/Exit (Canonical Y-Shape)
    # =========================================================================

    def test_simple_roundabout_entry_exit(self):
        """Test canonical Y-shaped roundabout with entry and exit."""
        # Generate synthetic roundabout point cloud
        # Center at (10, 0), radius = 10m
        # Entry at 180° (straight ahead), Exit at 90° (right)

        center_x, center_y = 10.0, 0.0
        radius = 10.0

        # Ring 5: Entry leg (180° ± 30°)
        theta_entry = np.linspace(np.pi - np.pi/6, np.pi + np.pi/6, 40)
        x_entry = center_x + radius * np.cos(theta_entry)
        y_entry = center_y + radius * np.sin(theta_entry)
        z_entry = np.zeros(len(x_entry))
        ring_entry = np.full(len(x_entry), 5)

        # Ring 6: Exit leg (90° ± 30°)
        theta_exit = np.linspace(np.pi/2 - np.pi/6, np.pi/2 + np.pi/6, 40)
        x_exit = center_x + radius * np.cos(theta_exit)
        y_exit = center_y + radius * np.sin(theta_exit)
        z_exit = np.zeros(len(x_exit))
        ring_exit = np.full(len(x_exit), 6)

        # Combine into point cloud
        x = np.concatenate([x_entry, x_exit])
        y = np.concatenate([y_entry, y_exit])
        z = np.concatenate([z_entry, z_exit])
        ring = np.concatenate([ring_entry, ring_exit])

        point_cloud = np.column_stack([x, y, z, ring])

        # Create boundary detections (simulating PROMPT-2 output)
        # Both rings have multiple discontinuities (gaps in the arc)
        boundaries = [
            {
                'ring_id': 5,
                'y_min': -5.0,
                'y_max': 5.0,
                'discontinuities': [0, 20],  # 2 gaps
                'confidence': 0.8
            },
            {
                'ring_id': 6,
                'y_min': 5.0,
                'y_max': 15.0,
                'discontinuities': [0, 20],  # 2 gaps
                'confidence': 0.8
            }
        ]

        # Detect roundabouts
        roundabouts = self.detector.detect_roundabouts(point_cloud, boundaries)

        # Assertions
        self.assertGreater(len(roundabouts), 0,
                          "Should detect at least one roundabout")

        geom = roundabouts[0]
        self.assertAlmostEqual(geom.center_x, center_x, delta=2.0,
                              msg=f"Center X should be ~{center_x}m")
        self.assertAlmostEqual(geom.center_y, center_y, delta=2.0,
                              msg=f"Center Y should be ~{center_y}m")
        self.assertAlmostEqual(geom.radius, radius, delta=2.0,
                              msg=f"Radius should be ~{radius}m")
        self.assertTrue(geom.is_right_hand_exit,
                       msg="Exit should be classified as right-hand")
        self.assertGreaterEqual(geom.confidence, 0.7,
                               msg="Confidence should be ≥0.7")

        print(f"✓ Test 1 passed: Roundabout detected at ({geom.center_x:.2f}, {geom.center_y:.2f}), "
              f"radius={geom.radius:.2f}m, confidence={geom.confidence:.2f}")

    # =========================================================================
    # Test 2: True Bifurcation (Should Reject)
    # =========================================================================

    def test_no_roundabout_true_bifurcation(self):
        """Test that true bifurcations are NOT classified as roundabouts."""
        # Generate two diverging straight roads (bifurcation)
        # Road 1: Straight ahead (y=0 to y=5)
        # Road 2: Left fork (y=5 to y=15)
        # Distance between endpoints > 30m → not a roundabout

        # Ring 5: Road 1 (straight)
        x_road1 = np.linspace(0, 20, 40)
        y_road1 = np.zeros(40)
        z_road1 = np.zeros(40)
        ring_road1 = np.full(40, 5)

        # Ring 6: Road 2 (left fork, far away)
        x_road2 = np.linspace(0, 20, 40)
        y_road2 = np.linspace(10, 30, 40)  # Far from road 1
        z_road2 = np.zeros(40)
        ring_road2 = np.full(40, 6)

        # Combine
        x = np.concatenate([x_road1, x_road2])
        y = np.concatenate([y_road1, y_road2])
        z = np.concatenate([z_road1, z_road2])
        ring = np.concatenate([ring_road1, ring_road2])

        point_cloud = np.column_stack([x, y, z, ring])

        # Boundary detections (2 gaps each, but far apart)
        boundaries = [
            {
                'ring_id': 5,
                'y_min': -2.0,
                'y_max': 2.0,
                'discontinuities': [0, 20],
                'confidence': 0.8
            },
            {
                'ring_id': 6,
                'y_min': 8.0,
                'y_max': 32.0,
                'discontinuities': [0, 20],
                'confidence': 0.8
            }
        ]

        # Detect roundabouts
        roundabouts = self.detector.detect_roundabouts(point_cloud, boundaries)

        # Assertions
        # Should NOT detect roundabout (roads too far apart)
        self.assertEqual(len(roundabouts), 0,
                        "Should NOT detect roundabout for true bifurcation")

        print(f"✓ Test 2 passed: Bifurcation correctly rejected ({len(roundabouts)} roundabouts detected)")

    # =========================================================================
    # Test 3: Drive Leg Identification (Right-Hand Traffic)
    # =========================================================================

    def test_drive_leg_identification(self):
        """Test that right-hand traffic correctly identifies exit as driving leg."""
        # Create roundabout geometry with known entry/exit
        geom = RoundaboutGeometry(
            roundabout_id=0,
            center_x=10.0,
            center_y=0.0,
            radius=10.0,
            entry_azimuth=np.pi,  # 180° (straight ahead)
            exit_azimuth=np.pi/2,  # 90° (right)
            entry_ring_idx=5,
            exit_ring_idx=6,
            confidence=0.9,
            is_right_hand_exit=True,
            affected_rings=[5, 6]
        )

        # Identify driving leg
        drive_leg = self.handler.identify_drive_leg(geom)

        # Assertions
        self.assertEqual(drive_leg, 1,
                        "Right-hand traffic: exit (leg 1) should be driving leg")

        # Test weight assignment
        weights = self.handler.weight_rings(geom)

        self.assertEqual(weights[6], self.handler.ring_weight_driving,
                        f"Exit ring (6) should have weight={self.handler.ring_weight_driving}")
        self.assertEqual(weights[5], self.handler.ring_weight_non_driving,
                        f"Entry ring (5) should have weight={self.handler.ring_weight_non_driving}")

        print(f"✓ Test 3 passed: Drive leg={drive_leg} (exit), weights[5]={weights[5]:.1f}, weights[6]={weights[6]:.1f}")

    # =========================================================================
    # Test 4: Circle Fitting on Roundabout Arcs
    # =========================================================================

    def test_circle_fitting_roundabout(self):
        """Test that circle fitting accurately estimates roundabout geometry."""
        # Generate perfect circular arc
        center_x, center_y = 15.0, 5.0
        radius = 8.0

        # Arc from 0° to 180° (half circle)
        theta = np.linspace(0, np.pi, 100)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        z = np.zeros(len(x))
        ring = np.full(len(x), 7)

        ring_points = np.column_stack([x, y, z, ring])

        # Fit arc geometry
        y_range = (center_y - radius, center_y + radius)
        cx, cy, r = self.detector.fit_arc_geometry(ring_points, y_range)

        # Assertions
        self.assertAlmostEqual(cx, center_x, delta=0.5,
                              msg=f"Fitted center X ({cx:.2f}) should match true ({center_x:.2f})")
        self.assertAlmostEqual(cy, center_y, delta=0.5,
                              msg=f"Fitted center Y ({cy:.2f}) should match true ({center_y:.2f})")
        self.assertAlmostEqual(r, radius, delta=0.5,
                              msg=f"Fitted radius ({r:.2f}) should match true ({radius:.2f})")

        print(f"✓ Test 4 passed: Circle fitting accurate - "
              f"center=({cx:.2f}, {cy:.2f}), radius={r:.2f}m")

    # =========================================================================
    # Test 5: Ring Consistency Check
    # =========================================================================

    def test_ring_consistency_check(self):
        """Test ring consistency validation for roundabout vs bifurcation."""
        # Test 1: Consistent centers (roundabout)
        roundabout_centers = np.array([
            [10.0, 0.0],
            [10.1, 0.05],
            [9.95, -0.02],
            [10.05, 0.03]
        ])

        consistency_roundabout = self.detector.check_ring_consistency(roundabout_centers)

        self.assertGreater(consistency_roundabout, 0.7,
                          msg="Roundabout centers should have high consistency (>0.7)")

        # Test 2: Diverging centers (bifurcation)
        bifurcation_centers = np.array([
            [10.0, 0.0],
            [12.0, 5.0],
            [14.0, 10.0],
            [16.0, 15.0]
        ])

        consistency_bifurcation = self.detector.check_ring_consistency(bifurcation_centers)

        self.assertLess(consistency_bifurcation, 0.5,
                       msg="Bifurcation centers should have low consistency (<0.5)")

        print(f"✓ Test 5 passed: Ring consistency - roundabout={consistency_roundabout:.2f}, "
              f"bifurcation={consistency_bifurcation:.2f}")

    # =========================================================================
    # Test 6: Multiple Sequential Roundabouts
    # =========================================================================

    def test_multiple_sequential_roundabouts(self):
        """Test handling of 2+ roundabouts in sequence."""
        # Generate 2 roundabouts at different positions

        # Roundabout 1: Center at (10, 0), radius=8m
        center1_x, center1_y = 10.0, 0.0
        radius1 = 8.0

        theta1_entry = np.linspace(np.pi - np.pi/6, np.pi + np.pi/6, 30)
        x1_entry = center1_x + radius1 * np.cos(theta1_entry)
        y1_entry = center1_y + radius1 * np.sin(theta1_entry)
        z1_entry = np.zeros(len(x1_entry))
        ring1_entry = np.full(len(x1_entry), 3)

        theta1_exit = np.linspace(np.pi/2 - np.pi/6, np.pi/2 + np.pi/6, 30)
        x1_exit = center1_x + radius1 * np.cos(theta1_exit)
        y1_exit = center1_y + radius1 * np.sin(theta1_exit)
        z1_exit = np.zeros(len(x1_exit))
        ring1_exit = np.full(len(x1_exit), 4)

        # Roundabout 2: Center at (30, 0), radius=10m
        center2_x, center2_y = 30.0, 0.0
        radius2 = 10.0

        theta2_entry = np.linspace(np.pi - np.pi/6, np.pi + np.pi/6, 30)
        x2_entry = center2_x + radius2 * np.cos(theta2_entry)
        y2_entry = center2_y + radius2 * np.sin(theta2_entry)
        z2_entry = np.zeros(len(x2_entry))
        ring2_entry = np.full(len(x2_entry), 7)

        theta2_exit = np.linspace(np.pi/2 - np.pi/6, np.pi/2 + np.pi/6, 30)
        x2_exit = center2_x + radius2 * np.cos(theta2_exit)
        y2_exit = center2_y + radius2 * np.sin(theta2_exit)
        z2_exit = np.zeros(len(x2_exit))
        ring2_exit = np.full(len(x2_exit), 8)

        # Combine all segments
        x = np.concatenate([x1_entry, x1_exit, x2_entry, x2_exit])
        y = np.concatenate([y1_entry, y1_exit, y2_entry, y2_exit])
        z = np.concatenate([z1_entry, z1_exit, z2_entry, z2_exit])
        ring = np.concatenate([ring1_entry, ring1_exit, ring2_entry, ring2_exit])

        point_cloud = np.column_stack([x, y, z, ring])

        # Boundary detections
        boundaries = [
            {'ring_id': 3, 'y_min': -4.0, 'y_max': 4.0, 'discontinuities': [0, 15], 'confidence': 0.8},
            {'ring_id': 4, 'y_min': 4.0, 'y_max': 12.0, 'discontinuities': [0, 15], 'confidence': 0.8},
            {'ring_id': 7, 'y_min': -5.0, 'y_max': 5.0, 'discontinuities': [0, 15], 'confidence': 0.8},
            {'ring_id': 8, 'y_min': 5.0, 'y_max': 15.0, 'discontinuities': [0, 15], 'confidence': 0.8},
        ]

        # Detect roundabouts
        roundabouts = self.detector.detect_roundabouts(point_cloud, boundaries)

        # Assertions
        self.assertGreaterEqual(len(roundabouts), 2,
                               f"Should detect 2 roundabouts, found {len(roundabouts)}")

        print(f"✓ Test 6 passed: Detected {len(roundabouts)} roundabouts in sequence")
        for i, geom in enumerate(roundabouts):
            print(f"  Roundabout {i+1}: center=({geom.center_x:.2f}, {geom.center_y:.2f}), "
                  f"radius={geom.radius:.2f}m")

    # =========================================================================
    # Test 7: Y-Road Boundary Adjustment (Circular vs Linear)
    # =========================================================================

    def test_y_road_boundary_adjustment(self):
        """Test circular arc boundary adjustment vs linear detection."""
        # Generate roundabout arc points
        center_x, center_y = 12.0, 3.0
        radius = 9.0

        # Arc from 45° to 135° (90° arc)
        theta = np.linspace(np.pi/4, 3*np.pi/4, 50)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        z = np.zeros(len(x))
        ring = np.full(len(x), 6)

        ring_points = np.column_stack([x, y, z, ring])

        # Original boundaries (linear detection)
        # These would be y_min/y_max of all points
        original_y_min = float(np.min(y))
        original_y_max = float(np.max(y))

        # Create roundabout geometry
        geom = RoundaboutGeometry(
            roundabout_id=0,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            entry_azimuth=np.pi,
            exit_azimuth=np.pi/2,
            entry_ring_idx=5,
            exit_ring_idx=6,
            confidence=0.9,
            is_right_hand_exit=True,
            affected_rings=[5, 6]
        )

        # Adjust boundaries using circular constraint
        adjusted_y_min, adjusted_y_max = self.handler.adjust_boundaries_in_roundabout(
            ring_points, geom, original_y_min, original_y_max
        )

        # Assertions
        # Adjusted boundaries should be similar to original (all points on circle)
        # but may filter outliers
        self.assertLessEqual(abs(adjusted_y_min - original_y_min), 1.0,
                            msg="Adjusted y_min should be close to original")
        self.assertLessEqual(abs(adjusted_y_max - original_y_max), 1.0,
                            msg="Adjusted y_max should be close to original")

        # Ensure adjusted boundaries are valid
        self.assertLess(adjusted_y_min, adjusted_y_max,
                       msg="Adjusted y_min should be < y_max")

        print(f"✓ Test 7 passed: Boundary adjustment - "
              f"original=({original_y_min:.2f}, {original_y_max:.2f}), "
              f"adjusted=({adjusted_y_min:.2f}, {adjusted_y_max:.2f})")


# =============================================================================
# Validation Data Tests (Real Captured Curved/Roundabout Data)
# =============================================================================

class TestRoundaboutWithValidationData(unittest.TestCase):
    """Test roundabout detection using real captured validation data."""

    @classmethod
    def setUpClass(cls):
        """Set up validation data paths and detectors."""
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.roundabout_file = cls.workspace_root / 'roundabouts_curve_points_validation.txt'
        cls.curved_file = cls.workspace_root / 'whole_frame_points_curved_line.txt'

        cls.detector = RoundaboutDetector(
            proximity_threshold=30.0,
            radius_tolerance=2.0,
            typical_roundabout_radius=10.0,
            min_consistency_score=0.7,
            logger=None
        )

        cls.handler = YRoadHandler(
            right_hand_traffic=True,
            num_rings=12,
            ring_weight_driving=1.0,
            ring_weight_non_driving=0.3,
            circular_arc_tolerance=0.5,
            logger=None
        )

    def _parse_validation_file(self, filepath):
        """Parse validation txt file and extract all frames of point data."""
        import struct

        if not filepath.exists():
            return []

        with open(filepath, 'r') as f:
            content = f.read()

        # Split by YAML document separator
        documents = content.split('---')
        all_frames = []

        for doc in documents:
            doc = doc.strip()
            if not doc:
                continue

            data_bytes = []
            in_data = False

            for line in doc.split('\n'):
                line = line.strip()
                if line.startswith('data:'):
                    in_data = True
                elif in_data and line.startswith('-'):
                    byte_val = line[1:].strip().strip("'\"")
                    if byte_val == '...' or byte_val == "'...'":
                        break
                    try:
                        data_bytes.append(int(byte_val))
                    except ValueError:
                        continue
                elif in_data and line and not line.startswith('-'):
                    in_data = False

            if len(data_bytes) < 16:
                continue

            # Decode points
            points = []
            byte_array = bytes(data_bytes)
            num_points = len(byte_array) // 16

            for i in range(num_points):
                offset = i * 16
                if offset + 16 > len(byte_array):
                    break
                x, y, z, intensity = struct.unpack('<4f', byte_array[offset:offset+16])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    dist = np.sqrt(x**2 + y**2)
                    ring_id = min(int(dist / 1.5), 11)
                    points.append([x, y, z, ring_id])

            if points:
                all_frames.append(np.array(points))

        return all_frames

    def test_curved_road_geometry_analysis(self):
        """Test geometry analysis on curved road data."""
        if not self.curved_file.exists():
            self.skipTest(f"Validation file not found: {self.curved_file}")

        frames = self._parse_validation_file(self.curved_file)
        if not frames or len(frames[0]) < 3:
            self.skipTest("Not enough points in curved file")

        points = frames[0]
        print(f"\n[Curved Road Validation] {len(points)} points")

        # Compute curvature characteristics
        if len(points) >= 3:
            # Use covariance to analyze point distribution
            cov = np.cov(points[:, 0], points[:, 1])
            eigenvalues = np.linalg.eigvalsh(cov)

            # Ratio indicates spread pattern
            if eigenvalues[0] > 0.001:
                spread_ratio = eigenvalues[1] / eigenvalues[0]
                print(f"  Eigenvalue spread ratio: {spread_ratio:.2f}")

            # Y-spread indicates lateral extent
            y_spread = points[:, 1].max() - points[:, 1].min()
            x_extent = points[:, 0].max() - points[:, 0].min()
            print(f"  Y spread: {y_spread:.2f}m, X extent: {x_extent:.2f}m")

            self.assertGreater(y_spread, 0.1, "Should have measurable spread")

    def test_roundabout_multi_frame_analysis(self):
        """Test roundabout data with multiple frames."""
        if not self.roundabout_file.exists():
            self.skipTest(f"Validation file not found: {self.roundabout_file}")

        frames = self._parse_validation_file(self.roundabout_file)
        if not frames:
            self.skipTest("No frames parsed from roundabout file")

        print(f"\n[Roundabout Multi-Frame Validation] {len(frames)} frames")

        for i, points in enumerate(frames):
            if len(points) < 3:
                continue

            # Analyze each frame
            centroid = np.mean(points[:, :2], axis=0)
            std_dev = np.std(points[:, :2], axis=0)

            print(f"  Frame {i+1}: {len(points)} pts, "
                  f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), "
                  f"std=({std_dev[0]:.2f}, {std_dev[1]:.2f})")

            # Check ring distribution
            unique_rings = np.unique(points[:, 3].astype(int))
            print(f"    Rings: {list(unique_rings)}")

        self.assertGreater(len(frames), 0, "Should have parsed at least one frame")

    def test_curvature_estimation_from_real_data(self):
        """Estimate curvature from real curved road data."""
        if not self.curved_file.exists():
            self.skipTest(f"Validation file not found: {self.curved_file}")

        frames = self._parse_validation_file(self.curved_file)
        if not frames or len(frames[0]) < 5:
            self.skipTest("Need at least 5 points for curvature estimation")

        points = frames[0][:, :2]  # Use x, y only

        if len(points) >= 5:
            # Sort by angle from centroid for curve fitting
            centroid = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1],
                               points[:, 0] - centroid[0])
            sorted_idx = np.argsort(angles)
            sorted_points = points[sorted_idx]

            # Compute discrete curvature if we have enough points
            if len(sorted_points) >= 3:
                # Estimate radius of curvature
                dists = np.sqrt(np.sum((sorted_points - centroid)**2, axis=1))
                avg_radius = np.mean(dists)
                radius_std = np.std(dists)

                print(f"\n[Curvature Estimation]")
                print(f"  Average radius from centroid: {avg_radius:.2f}m")
                print(f"  Radius std deviation: {radius_std:.2f}m")
                print(f"  Curvature indicator: {1.0/avg_radius if avg_radius > 0.1 else 0:.4f}")


# =============================================================================
# Test Runner
# =============================================================================

def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRoundaboutHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestRoundaboutWithValidationData))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY - ROUNDABOUT HANDLER")
    print(f"{'='*70}")
    print(f"Total tests run:     {result.testsRun}")
    print(f"Successes:           {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:            {len(result.failures)}")
    print(f"Errors:              {len(result.errors)}")
    print(f"{'='*70}")

    # Calculate metrics
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100

    print(f"\nMETRICS:")
    print(f"  Test pass rate: {success_rate:.1f}%")
    print(f"  Target: 100% (all 7 tests passing)")

    if success_rate >= 100:
        print(f"\n✓ All tests passed! Ready for integration.")
    else:
        print(f"\n⚠ Some tests failed. Review failures above.")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
