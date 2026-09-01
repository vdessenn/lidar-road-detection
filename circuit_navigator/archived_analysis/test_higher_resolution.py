#!/usr/bin/env python3
"""Test with higher resolution (0.25m instead of 0.5m)."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def test_resolution(bag_path: str):
    """Test different resolutions."""

    for resolution in [0.5, 0.25, 0.1]:
        print(f"\n{'='*50}")
        print(f"Resolution: {resolution}m")
        print('='*50)

        extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
        data = extractor.extract()

        raw_grid = data.raw_data
        has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

        # Count components in raw
        labeled_raw, num_raw = ndimage.label(has_point)
        sizes_raw = [(np.sum(labeled_raw == i)) for i in range(1, num_raw + 1)]
        largest_raw = max(sizes_raw) if sizes_raw else 0
        print(f"Raw: {np.sum(has_point)} cells in {num_raw} components (largest: {largest_raw})")

        # Closing
        closing_radius_m = 3.0
        closing_radius_px = max(1, int(closing_radius_m / resolution))
        kernel = disk(closing_radius_px)
        closed = closing(has_point, kernel)

        min_size_m2 = 50
        min_size_pixels = int(min_size_m2 / (resolution * resolution))
        cleaned = remove_small_objects(closed, min_size=min_size_pixels)

        # Skeleton
        skeleton = skeletonize(cleaned)
        labeled_skel, num_skel = ndimage.label(skeleton)
        sizes_skel = [(np.sum(labeled_skel == i)) for i in range(1, num_skel + 1)]
        largest_skel = max(sizes_skel) if sizes_skel else 0

        print(f"Skeleton: {np.sum(skeleton)} cells in {num_skel} components (largest: {largest_skel})")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    test_resolution(bag_path)
