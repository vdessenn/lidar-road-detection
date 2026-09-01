#!/usr/bin/env python3
"""Check skeleton connectivity on the cleaned (pre-erosion) grid."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def check_connectivity(bag_path: str, resolution: float = 0.5):
    """Check skeleton connectivity on different processing stages."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)
    closing_radius_px = max(1, int(2.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)
    min_size_pixels = int(50 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    # Check skeleton on cleaned (no erosion)
    skeleton_cleaned = skeletonize(cleaned)
    labeled_cleaned, num_cleaned = ndimage.label(skeleton_cleaned)
    print(f"Skeleton on CLEANED (no erosion):")
    print(f"  Total pixels: {np.sum(skeleton_cleaned)}")
    print(f"  Connected components: {num_cleaned}")

    # Show largest components
    if num_cleaned > 0:
        sizes = [(i, np.sum(labeled_cleaned == i)) for i in range(1, num_cleaned + 1)]
        sizes.sort(key=lambda x: -x[1])
        print(f"  Largest components:")
        for comp_id, size in sizes[:5]:
            coords = np.argwhere(labeled_cleaned == comp_id)
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            world_y_range = (origin[1] + min_y * resolution, origin[1] + max_y * resolution)
            world_x_range = (origin[0] + min_x * resolution, origin[0] + max_x * resolution)
            print(f"    Component {comp_id}: {size} pixels, X:[{world_x_range[0]:.1f}, {world_x_range[1]:.1f}], Y:[{world_y_range[0]:.1f}, {world_y_range[1]:.1f}]")

    # Now check with safety margin erosion
    safety_radius_px = max(1, int(0.50 / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    skeleton_navigable = skeletonize(navigable)
    labeled_navigable, num_navigable = ndimage.label(skeleton_navigable)
    print(f"\nSkeleton on NAVIGABLE (with 50cm erosion):")
    print(f"  Total pixels: {np.sum(skeleton_navigable)}")
    print(f"  Connected components: {num_navigable}")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    check_connectivity(bag_path)
