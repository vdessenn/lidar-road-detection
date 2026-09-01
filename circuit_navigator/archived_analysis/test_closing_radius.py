#!/usr/bin/env python3
"""Test different closing radii to find one that creates a connected skeleton."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def test_closing(bag_path: str, resolution: float = 0.5):
    """Test different closing radii."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Test different closing radii
    for radius_m in [2.0, 3.0, 4.0, 5.0, 6.0]:
        closing_radius_px = max(1, int(radius_m / resolution))
        kernel = disk(closing_radius_px)
        closed = closing(has_point, kernel)
        min_size_pixels = int(50 / (resolution * resolution))
        cleaned = remove_small_objects(closed, min_size=min_size_pixels)

        skeleton = skeletonize(cleaned)
        labeled, num_components = ndimage.label(skeleton)

        # Find largest component
        if num_components > 0:
            sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
            sizes.sort(key=lambda x: -x[1])
            largest_size = sizes[0][1]
        else:
            largest_size = 0

        print(f"Closing radius {radius_m}m ({closing_radius_px}px):")
        print(f"  Skeleton pixels: {np.sum(skeleton)}")
        print(f"  Connected components: {num_components}")
        print(f"  Largest component: {largest_size} pixels")
        print()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    test_closing(bag_path)
