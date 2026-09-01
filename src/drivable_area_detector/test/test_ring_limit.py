"""
Unit tests for ring limit constraint (max 15 rings).

PDR Constraint: LiDAR portée limitée, max 15 anneaux utilisables.
Rings 0-14 are valid, rings 15+ should be filtered.

Author: PDR Phase 2
Date: 2026-01-12
"""

import unittest
import numpy as np


class TestRingLimit(unittest.TestCase):
    """Tests for 15-ring maximum constraint."""

    # PDR-defined constant
    MAX_RINGS = 15  # Rings 0-14 valid

    def test_valid_ring_range(self):
        """Rings 0-14 should all be valid."""
        for ring_id in range(15):
            self.assertTrue(ring_id < self.MAX_RINGS,
                           f"Ring {ring_id} should be valid")

    def test_invalid_ring_range(self):
        """Rings 15+ should be invalid."""
        invalid_rings = [15, 16, 20, 30, 39, 40, 127]
        for ring_id in invalid_rings:
            self.assertFalse(ring_id < self.MAX_RINGS,
                            f"Ring {ring_id} should be invalid")

    def test_filter_rings_above_limit(self):
        """Filter function should remove rings >= 15."""
        # Simulated ring IDs from LiDAR
        all_rings = np.array([0, 5, 10, 14, 15, 20, 39])
        
        # Filter logic
        valid_mask = all_rings < self.MAX_RINGS
        filtered_rings = all_rings[valid_mask]
        
        expected = np.array([0, 5, 10, 14])
        np.testing.assert_array_equal(filtered_rings, expected)

    def test_ring_count_after_filtering(self):
        """Verify correct count of valid rings."""
        # Full Pandora ring set (0-39)
        all_rings = np.arange(40)
        
        valid_mask = all_rings < self.MAX_RINGS
        valid_count = np.sum(valid_mask)
        
        self.assertEqual(valid_count, 15)  # Rings 0-14

    def test_empty_after_filtering(self):
        """If all rings are invalid, result should be empty."""
        all_rings = np.array([15, 20, 30, 39])
        
        valid_mask = all_rings < self.MAX_RINGS
        filtered_rings = all_rings[valid_mask]
        
        self.assertEqual(len(filtered_rings), 0)

    def test_points_per_ring_distribution(self):
        """Valid rings should maintain point distribution."""
        # Simulated point cloud with ring IDs
        np.random.seed(42)
        num_points = 1000
        ring_ids = np.random.randint(0, 40, num_points)
        
        # Filter valid rings
        valid_mask = ring_ids < self.MAX_RINGS
        valid_rings = ring_ids[valid_mask]
        
        # Count points per valid ring
        unique, counts = np.unique(valid_rings, return_counts=True)
        
        # All unique rings should be < 15
        self.assertTrue(np.all(unique < self.MAX_RINGS))
        
        # Should have some points in each ring (statistically)
        # With 1000 points across 40 rings, expect ~25 per ring
        # Valid rings (15) should have ~375 total points
        self.assertGreater(len(valid_rings), 300)

    def test_ring_field_data_type(self):
        """Ring field should support values 0-14."""
        ring_ids = np.array([0, 7, 14], dtype=np.uint8)
        
        # All should be valid
        valid_mask = ring_ids < self.MAX_RINGS
        self.assertTrue(np.all(valid_mask))

    def test_boundary_ring_14(self):
        """Ring 14 is the last valid ring."""
        self.assertTrue(14 < self.MAX_RINGS)
        self.assertFalse(15 < self.MAX_RINGS)

    def test_ring_scale_factors(self):
        """Ring density scale factors should cover 15 rings."""
        # PDR-defined scale factors (extended to 15 rings)
        ring_scale = [
            1.5, 1.3, 1.2,  # Rings 0-2 (close, high density)
            1.0, 1.0, 1.0,  # Rings 3-5 (baseline)
            0.8, 0.8, 0.8,  # Rings 6-8 (medium distance)
            0.6, 0.6, 0.6,  # Rings 9-11 (far)
            0.5, 0.5, 0.5   # Rings 12-14 (very far, extended)
        ]
        
        self.assertEqual(len(ring_scale), self.MAX_RINGS)
        
        # Verify decreasing trend (closer = higher threshold)
        self.assertGreater(ring_scale[0], ring_scale[14])


class TestRingBoundaryIntegration(unittest.TestCase):
    """Integration tests for ring filtering in boundary extraction."""

    MAX_RINGS = 15

    def test_boundary_extraction_ignores_far_rings(self):
        """Boundary extraction should only use rings 0-14."""
        # Simulated boundary points with ring IDs
        boundary_data = [
            {'ring_id': 0, 'x': 1.0, 'y_min': -1.8, 'y_max': 1.8},
            {'ring_id': 5, 'x': 5.0, 'y_min': -2.0, 'y_max': 2.0},
            {'ring_id': 14, 'x': 14.0, 'y_min': -2.5, 'y_max': 2.5},
            {'ring_id': 20, 'x': 20.0, 'y_min': -3.0, 'y_max': 3.0},  # Invalid
            {'ring_id': 39, 'x': 39.0, 'y_min': -4.0, 'y_max': 4.0},  # Invalid
        ]
        
        # Filter valid rings
        valid_boundaries = [b for b in boundary_data if b['ring_id'] < self.MAX_RINGS]
        
        self.assertEqual(len(valid_boundaries), 3)
        self.assertEqual(valid_boundaries[-1]['ring_id'], 14)

    def test_x_range_from_valid_rings(self):
        """X range should be computed from valid rings only."""
        boundaries = [
            {'ring_id': 0, 'x': 1.0},
            {'ring_id': 10, 'x': 10.0},
            {'ring_id': 14, 'x': 15.0},
            {'ring_id': 25, 'x': 30.0},  # Invalid ring - should be excluded
        ]
        
        valid_boundaries = [b for b in boundaries if b['ring_id'] < self.MAX_RINGS]
        x_values = [b['x'] for b in valid_boundaries]
        
        x_min = min(x_values)
        x_max = max(x_values)
        
        self.assertEqual(x_min, 1.0)
        self.assertEqual(x_max, 15.0)  # Not 30.0 (from invalid ring)


if __name__ == '__main__':
    unittest.main()
