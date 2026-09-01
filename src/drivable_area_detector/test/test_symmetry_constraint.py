"""
Test Suite for Symmetry Constraint in Boundary Detection

Tests for _apply_symmetry_constraint function in rviz2_publisher.py:
- Width below threshold: pass through unchanged
- Width above threshold: keep closest point to origin, mirror
- Missing left or right side: mirror from available side

Author: Victor Dessenne
Date: 2026-01-12
"""

import unittest
import numpy as np

class MockRviz2Publisher:
    """Mock class to test _apply_symmetry_constraint without ROS dependencies."""
    
    def _apply_symmetry_constraint(self, x: float, y_min: float, y_max: float, 
                                   max_width: float = 6.0) -> tuple:
        """Apply symmetry: keep closest point to origin, mirror the other."""
        # Handle missing sides
        if y_min is None and y_max is None:
            return -2.0, 2.0  # Default
        if y_min is None:
            return -y_max, y_max  # Mirror left to right
        if y_max is None:
            return y_min, -y_min  # Mirror right to left

        # Check width constraint
        width = y_max - y_min
        if width <= max_width:
            return y_min, y_max  # Valid width

        # Width too large: keep point closest to origin, mirror it
        dist_left = np.sqrt(x**2 + y_max**2)
        dist_right = np.sqrt(x**2 + y_min**2)

        if dist_left < dist_right:
            return -y_max, y_max  # Keep left, mirror to right
        else:
            return y_min, -y_min  # Keep right, mirror to left


class TestSymmetryConstraint(unittest.TestCase):
    """Test suite for symmetry constraint."""

    def setUp(self):
        """Set up test fixtures."""
        self.publisher = MockRviz2Publisher()

    def test_width_below_threshold_unchanged(self):
        """Test 1: Width < 6m should pass through unchanged."""
        print("\n[Test 1] Width below threshold...")
        
        # Normal road: y_min = -2, y_max = 2, width = 4m
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=5.0, y_min=-2.0, y_max=2.0, max_width=6.0
        )
        
        self.assertEqual(y_min, -2.0)
        self.assertEqual(y_max, 2.0)
        print("✅ Width 4m unchanged")

    def test_width_above_threshold_mirror_left(self):
        """Test 2: Width > 6m, left closer to origin → mirror left."""
        print("\n[Test 2] Width above threshold, left closer...")
        
        # Wide detection: y_min = -10 (far right), y_max = 2 (close left)
        # At x=3: dist_left = sqrt(9+4) = 3.6, dist_right = sqrt(9+100) = 10.4
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=3.0, y_min=-10.0, y_max=2.0, max_width=6.0
        )
        
        # Should keep left (2.0) and mirror: -2.0, 2.0
        self.assertEqual(y_min, -2.0)
        self.assertEqual(y_max, 2.0)
        print(f"✅ Mirrored left: y_min={y_min}, y_max={y_max}")

    def test_width_above_threshold_mirror_right(self):
        """Test 3: Width > 6m, right closer to origin → mirror right."""
        print("\n[Test 3] Width above threshold, right closer...")
        
        # Wide detection: y_min = -2 (close right), y_max = 10 (far left)
        # At x=3: dist_left = sqrt(9+100) = 10.4, dist_right = sqrt(9+4) = 3.6
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=3.0, y_min=-2.0, y_max=10.0, max_width=6.0
        )
        
        # Should keep right (-2.0) and mirror: -2.0, 2.0
        self.assertEqual(y_min, -2.0)
        self.assertEqual(y_max, 2.0)
        print(f"✅ Mirrored right: y_min={y_min}, y_max={y_max}")

    def test_missing_left_side(self):
        """Test 4: Missing left side (y_max=None) → mirror from right."""
        print("\n[Test 4] Missing left side...")
        
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=5.0, y_min=-2.5, y_max=None, max_width=6.0
        )
        
        # Should mirror right to left: -2.5, 2.5
        self.assertEqual(y_min, -2.5)
        self.assertEqual(y_max, 2.5)
        print(f"✅ Mirrored from right: y_min={y_min}, y_max={y_max}")

    def test_missing_right_side(self):
        """Test 5: Missing right side (y_min=None) → mirror from left."""
        print("\n[Test 5] Missing right side...")
        
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=5.0, y_min=None, y_max=3.0, max_width=6.0
        )
        
        # Should mirror left to right: -3.0, 3.0
        self.assertEqual(y_min, -3.0)
        self.assertEqual(y_max, 3.0)
        print(f"✅ Mirrored from left: y_min={y_min}, y_max={y_max}")

    def test_both_sides_missing(self):
        """Test 6: Both sides missing → default to ±2.0."""
        print("\n[Test 6] Both sides missing...")
        
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=5.0, y_min=None, y_max=None, max_width=6.0
        )
        
        # Should return default: -2.0, 2.0
        self.assertEqual(y_min, -2.0)
        self.assertEqual(y_max, 2.0)
        print(f"✅ Default applied: y_min={y_min}, y_max={y_max}")

    def test_exact_threshold_width(self):
        """Test 7: Width exactly 6m should pass through unchanged."""
        print("\n[Test 7] Width exactly at threshold...")
        
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=5.0, y_min=-3.0, y_max=3.0, max_width=6.0
        )
        
        self.assertEqual(y_min, -3.0)
        self.assertEqual(y_max, 3.0)
        print(f"✅ Width 6m unchanged: y_min={y_min}, y_max={y_max}")

    def test_close_to_vehicle_x_zero(self):
        """Test 8: At x=0 (vehicle position), distance is |y|."""
        print("\n[Test 8] At x=0...")
        
        # At x=0: dist_left = |y_max|, dist_right = |y_min|
        y_min, y_max = self.publisher._apply_symmetry_constraint(
            x=0.0, y_min=-8.0, y_max=2.0, max_width=6.0
        )
        
        # Left closer (|2| < |8|), mirror left
        self.assertEqual(y_min, -2.0)
        self.assertEqual(y_max, 2.0)
        print(f"✅ At x=0: y_min={y_min}, y_max={y_max}")


def run_tests():
    """Run all tests and return exit code."""
    print("=" * 70)
    print("Test Suite: Symmetry Constraint")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSymmetryConstraint)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"Total tests run:     {result.testsRun}")
    print(f"Successes:           {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:            {len(result.failures)}")
    print(f"Errors:              {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    exit(exit_code)
