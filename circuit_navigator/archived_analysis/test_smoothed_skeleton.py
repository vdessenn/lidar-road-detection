#!/usr/bin/env python3
"""Test skeleton connectivity after smoothing."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize, opening

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def test_smoothed(bag_path: str, resolution: float = 0.5):
    """Test skeleton connectivity after smoothing."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Smoothed processing
    closing_radius_px = max(1, int(3.0 / resolution))
    closing_kernel = disk(closing_radius_px)
    closed = closing(has_point, closing_kernel)

    opening_radius_px = max(1, int(1.5 / resolution))
    opening_kernel = disk(opening_radius_px)
    smoothed = opening(closed, opening_kernel)

    min_size_pixels = int(100 / (resolution * resolution))
    cleaned = remove_small_objects(smoothed, min_size=min_size_pixels)

    skeleton = skeletonize(cleaned)
    labeled, num_components = ndimage.label(skeleton)

    print(f"After smoothing (closing=3m, opening=1.5m):")
    print(f"  Skeleton pixels: {np.sum(skeleton)}")
    print(f"  Connected components: {num_components}")

    if num_components > 0:
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        sizes.sort(key=lambda x: -x[1])
        print(f"  Largest components:")
        for comp_id, size in sizes[:10]:
            coords = np.argwhere(labeled == comp_id)
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            world_y = (origin[1] + min_y * resolution, origin[1] + max_y * resolution)
            world_x = (origin[0] + min_x * resolution, origin[0] + max_x * resolution)
            print(f"    C{comp_id}: {size}px, X:[{world_x[0]:.0f},{world_x[1]:.0f}], Y:[{world_y[0]:.0f},{world_y[1]:.0f}]")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    test_smoothed(bag_path)
