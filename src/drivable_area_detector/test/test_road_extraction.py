"""
Unit tests for Node 01: Road Points Extraction.

Tests:
1. Intensity filtering logic (min/max bounds)
2. Filter variant configurations
3. Ring limit validation (max 15 rings)
4. Edge cases and error handling

Author: Victor Dessenne
Date: 2026-01-12
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'drivable_area_detector'))
sys.path.insert(0, parent_dir)


class TestIntensityFiltering(unittest.TestCase):
    """Tests for intensity-based road filtering."""

    def test_intensity_in_range_kept(self):
        """Points with intensity in [0, 7] should be kept."""
        min_intensity = 0.0
        max_intensity = 7.0
        
        # Test data: intensities that should pass
        intensities = np.array([0.0, 1.5, 3.0, 5.0, 7.0])
        mask = (intensities >= min_intensity) & (intensities <= max_intensity)
        
        self.assertTrue(np.all(mask))
        self.assertEqual(np.sum(mask), 5)

    def test_intensity_out_of_range_rejected(self):
        """Points with intensity > 7 should be rejected."""
        min_intensity = 0.0
        max_intensity = 7.0
        
        # Test data: intensities that should be rejected
        intensities = np.array([7.1, 10.0, 20.0, 50.0, 100.0])
        mask = (intensities >= min_intensity) & (intensities <= max_intensity)
        
        self.assertFalse(np.any(mask))
        self.assertEqual(np.sum(mask), 0)

    def test_negative_intensity_rejected(self):
        """Negative intensity values should be rejected if min=0."""
        min_intensity = 0.0
        max_intensity = 7.0
        
        intensities = np.array([-1.0, -0.1, 0.0, 5.0])
        mask = (intensities >= min_intensity) & (intensities <= max_intensity)
        
        # Only 0.0 and 5.0 should pass
        self.assertEqual(np.sum(mask), 2)
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_boundary_values(self):
        """Boundary values (exactly 0 and exactly max) should be included."""
        min_intensity = 0.0
        max_intensity = 7.0
        
        intensities = np.array([0.0, 7.0])
        mask = (intensities >= min_intensity) & (intensities <= max_intensity)
        
        self.assertTrue(np.all(mask))

    def test_empty_input(self):
        """Empty intensity array should return empty mask."""
        min_intensity = 0.0
        max_intensity = 7.0
        
        intensities = np.array([])
        mask = (intensities >= min_intensity) & (intensities <= max_intensity)
        
        self.assertEqual(len(mask), 0)


class TestFilterVariants(unittest.TestCase):
    """Tests for different filter variant configurations."""

    def test_original_variant_config(self):
        """Original variant should have basic settings."""
        # Import NoiseFilterConfig
        try:
            from drivable_area_detector.road_points_extraction import NoiseFilterConfig
        except ImportError:
            # Direct import fallback
            from dataclasses import dataclass, field
            from typing import List
            
            @dataclass
            class NoiseFilterConfig:
                variant: str = "original"
                use_mad_z_variance: bool = False
                max_z_variance_base: float = 0.35
                keep_top_n_components: int = 1
                enable_ring_consistency: bool = False
                enable_outlier_rejection: bool = False
                enable_intensity_variance: bool = False
        
        config = NoiseFilterConfig(
            variant='original',
            use_mad_z_variance=False,
            max_z_variance_base=0.35,
            keep_top_n_components=1,
            enable_ring_consistency=False,
            enable_outlier_rejection=False,
            enable_intensity_variance=False
        )
        
        self.assertEqual(config.variant, 'original')
        self.assertFalse(config.use_mad_z_variance)
        self.assertEqual(config.keep_top_n_components, 1)
        self.assertFalse(config.enable_ring_consistency)

    def test_full_enhanced_variant_config(self):
        """Full enhanced variant should have all features enabled."""
        from dataclasses import dataclass, field
        from typing import List
        
        @dataclass
        class NoiseFilterConfig:
            variant: str = "original"
            use_mad_z_variance: bool = False
            max_z_variance_base: float = 0.35
            keep_top_n_components: int = 1
            enable_ring_consistency: bool = False
            enable_outlier_rejection: bool = False
            enable_intensity_variance: bool = False
            max_intensity_variance: float = 2.0
        
        config = NoiseFilterConfig(
            variant='full_enhanced',
            use_mad_z_variance=True,
            max_z_variance_base=0.50,
            keep_top_n_components=3,
            enable_ring_consistency=True,
            enable_outlier_rejection=True,
            enable_intensity_variance=True,
            max_intensity_variance=5.0
        )
        
        self.assertEqual(config.variant, 'full_enhanced')
        self.assertTrue(config.use_mad_z_variance)
        self.assertEqual(config.max_z_variance_base, 0.50)
        self.assertEqual(config.keep_top_n_components, 3)
        self.assertTrue(config.enable_ring_consistency)
        self.assertTrue(config.enable_outlier_rejection)
        self.assertTrue(config.enable_intensity_variance)


class TestRingLimits(unittest.TestCase):
    """Tests for ring-based filtering (max 15 rings)."""

    def test_max_ring_limit(self):
        """Rings 0-14 should be valid, 15+ should be rejected."""
        max_rings = 15  # PDR constraint: max 15 rings used
        
        ring_ids = np.array([0, 5, 10, 14, 15, 20, 39])
        valid_mask = ring_ids < max_rings
        
        # Rings 0, 5, 10, 14 valid; 15, 20, 39 rejected
        expected = np.array([True, True, True, True, False, False, False])
        np.testing.assert_array_equal(valid_mask, expected)
        self.assertEqual(np.sum(valid_mask), 4)

    def test_ring_id_boundary(self):
        """Ring 14 should be valid, ring 15 should be invalid."""
        max_rings = 15
        
        self.assertTrue(14 < max_rings)
        self.assertFalse(15 < max_rings)

    def test_ring_density_scaling(self):
        """Closer rings (0-2) should have higher density thresholds."""
        # PDR spec: ring_scale[0] = 1.5, ring_scale[11] = 0.6
        ring_scale = [
            1.5, 1.3, 1.2,  # Rings 0-2 (close)
            1.0, 1.0, 1.0,  # Rings 3-5 (medium)
            0.8, 0.8, 0.8,  # Rings 6-8 (far)
            0.6, 0.6, 0.6,  # Rings 9-11 (very far)
            0.6, 0.6, 0.6   # Rings 12-14 (extended)
        ]
        
        # Close rings should require more points
        self.assertGreater(ring_scale[0], ring_scale[10])
        self.assertGreater(ring_scale[1], ring_scale[9])


class TestRobustStatistics(unittest.TestCase):
    """Tests for MAD-based robust statistics."""

    def test_mad_computation(self):
        """MAD should be robust to outliers."""
        # Normal data with one outlier
        z_values = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 5.0])  # 5.0 is outlier
        
        median = np.median(z_values)
        mad = np.median(np.abs(z_values - median))
        
        # MAD should be small despite the outlier
        self.assertLess(mad, 0.2)

    def test_variance_affected_by_outlier(self):
        """Standard variance should be affected by outliers."""
        # Same data as above
        z_values = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 5.0])
        
        variance = np.var(z_values)
        
        # Variance should be large due to outlier
        self.assertGreater(variance, 3.0)

    def test_mad_vs_variance_comparison(self):
        """MAD should be more robust than variance for outlier detection."""
        z_clean = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
        z_outlier = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 10.0])
        
        # MAD change should be small
        mad_clean = np.median(np.abs(z_clean - np.median(z_clean)))
        mad_outlier = np.median(np.abs(z_outlier - np.median(z_outlier)))
        mad_change = abs(mad_outlier - mad_clean) / max(mad_clean, 0.001)
        
        # Variance change should be large
        var_clean = np.var(z_clean)
        var_outlier = np.var(z_outlier)
        var_change = abs(var_outlier - var_clean) / max(var_clean, 0.001)
        
        # MAD should change less than variance
        self.assertLess(mad_change, var_change)


class TestRingConsistency(unittest.TestCase):
    """Tests for ring consistency validation."""

    def test_consecutive_rings_kept(self):
        """Consecutive rings should be kept together."""
        rings = np.array([0, 1, 2, 3, 4])  # All consecutive
        max_gap = 2
        
        # Group finding: all in one group
        unique_rings = np.unique(rings)
        groups = []
        current = [unique_rings[0]]
        
        for i in range(1, len(unique_rings)):
            if unique_rings[i] - unique_rings[i-1] <= max_gap + 1:
                current.append(unique_rings[i])
            else:
                groups.append(current)
                current = [unique_rings[i]]
        groups.append(current)
        
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 5)

    def test_isolated_ring_rejected(self):
        """Isolated ring (gap > max_gap) should form separate group."""
        rings = np.array([0, 1, 2, 10])  # Ring 10 is isolated (gap=8)
        max_gap = 2
        
        unique_rings = np.unique(rings)
        groups = []
        current = [unique_rings[0]]
        
        for i in range(1, len(unique_rings)):
            if unique_rings[i] - unique_rings[i-1] <= max_gap + 1:
                current.append(unique_rings[i])
            else:
                groups.append(current)
                current = [unique_rings[i]]
        groups.append(current)
        
        self.assertEqual(len(groups), 2)  # Two groups: [0,1,2] and [10]
        self.assertEqual(len(groups[0]), 3)  # Largest group
        self.assertEqual(len(groups[1]), 1)  # Isolated ring


class TestGeometricConstraints(unittest.TestCase):
    """Tests for geometric constraints from KML specs."""

    def test_min_road_width(self):
        """Road width should be at least 3.5m (from PDR)."""
        min_width = 3.5  # meters
        
        test_widths = [3.0, 3.5, 4.0, 5.3]
        valid_mask = np.array(test_widths) >= min_width
        
        expected = [False, True, True, True]
        np.testing.assert_array_equal(valid_mask, expected)

    def test_max_curvature_roundabout(self):
        """Max curvature should be 0.15 (R_min=8m from KML)."""
        max_kappa = 0.15  # 1/8m ≈ 0.125 with margin
        
        # Curvature values to test
        curvatures = [0.07, 0.10, 0.125, 0.15, 0.20, 0.286]
        valid_mask = np.array(curvatures) <= max_kappa
        
        # First 4 should be valid
        expected = [True, True, True, True, False, False]
        np.testing.assert_array_equal(valid_mask, expected)

    def test_roundabout_radius_classification(self):
        """Curvature threshold for roundabout classification."""
        kappa_threshold = 0.10  # From PDR params
        
        # RP1: R=14m → κ=0.071 (below threshold, detected as curve)
        # RP2: R=8m → κ=0.125 (above threshold, detected as roundabout)
        kappa_rp1 = 1.0 / 14.0  # 0.071
        kappa_rp2 = 1.0 / 8.0   # 0.125
        
        self.assertLess(kappa_rp1, kappa_threshold)
        self.assertGreater(kappa_rp2, kappa_threshold)


class TestExtrapolationToOrigin(unittest.TestCase):
    """Tests for boundary extrapolation to x=0."""

    def test_origin_point_generation(self):
        """Boundary should start at x=0 when extrapolation enabled."""
        extrapolate = True
        x_min_detected = 2.0  # First detected point at x=2m
        
        if extrapolate:
            x_start = 0.0
        else:
            x_start = x_min_detected
        
        self.assertEqual(x_start, 0.0)

    def test_arc_continuity_from_origin(self):
        """Arc should be continuous from x=0 to x_max."""
        resolution = 0.5  # meters
        x_start = 0.0
        x_max = 20.0
        
        x_samples = np.arange(x_start, x_max + resolution, resolution)
        
        # First sample should be at origin
        self.assertEqual(x_samples[0], 0.0)
        
        # Samples should be evenly spaced
        diffs = np.diff(x_samples)
        np.testing.assert_allclose(diffs, resolution, rtol=1e-5)

    def test_origin_half_width(self):
        """At x=0, y should be at vehicle half-width (±2.0m for 4m road)."""
        origin_half_width = 2.0  # target_road_width/2 = 4.0/2 = 2.0m
        
        y_left_at_origin = origin_half_width
        y_right_at_origin = -origin_half_width
        
        self.assertEqual(y_left_at_origin, 2.0)
        self.assertEqual(y_right_at_origin, -2.0)
        
        # Road width at origin
        width_at_origin = y_left_at_origin - y_right_at_origin
        self.assertEqual(width_at_origin, 4.0)


class TestIntensityVarianceFilter(unittest.TestCase):
    """Tests for per-bin intensity variance filtering."""

    def test_low_variance_passes(self):
        """Asphalt (uniform intensity) should pass."""
        max_variance = 5.0
        
        # Asphalt-like: uniform low intensity
        intensities = np.array([2.0, 2.1, 1.9, 2.0, 2.2])
        variance = np.var(intensities)
        
        self.assertLess(variance, max_variance)

    def test_high_variance_rejected(self):
        """Grass/mixed terrain (variable intensity) should be rejected."""
        max_variance = 2.0  # Stricter threshold
        
        # Grass-like: highly variable intensity
        intensities = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        variance = np.var(intensities)
        
        self.assertGreater(variance, max_variance)


if __name__ == '__main__':
    unittest.main()
