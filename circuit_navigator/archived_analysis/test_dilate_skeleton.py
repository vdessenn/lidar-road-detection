#!/usr/bin/env python3
"""Try dilating skeleton fragments to connect them, then re-skeletonize."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize, opening, dilation

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def test_dilate_skeleton(bag_path: str, resolution: float = 0.5):
    """Connect skeleton fragments by dilation + re-skeletonization."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Initial processing
    closing_radius_px = max(1, int(3.0 / resolution))
    closing_kernel = disk(closing_radius_px)
    closed = closing(has_point, closing_kernel)

    opening_radius_px = max(1, int(1.5 / resolution))
    opening_kernel = disk(opening_radius_px)
    smoothed = opening(closed, opening_kernel)

    min_size_pixels = int(100 / (resolution * resolution))
    cleaned = remove_small_objects(smoothed, min_size=min_size_pixels)

    # First skeletonization
    skeleton1 = skeletonize(cleaned)
    print(f"Initial skeleton: {np.sum(skeleton1)} pixels")

    # Dilate skeleton to connect nearby fragments
    for dilation_radius in [2, 3, 4, 5]:
        dilation_kernel = disk(dilation_radius)
        dilated = dilation(skeleton1, dilation_kernel)

        # Re-skeletonize the dilated version
        skeleton2 = skeletonize(dilated)

        # Check connectivity
        labeled, num_components = ndimage.label(skeleton2)
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        sizes.sort(key=lambda x: -x[1])
        largest = sizes[0][1] if sizes else 0

        print(f"Dilation radius {dilation_radius}: {np.sum(skeleton2)} pixels, "
              f"{num_components} components, largest={largest}")

        # If we get a single large component, use it
        if num_components == 1 or (num_components <= 5 and largest > 200):
            print(f"  -> Good! Using dilation radius {dilation_radius}")

            # Visualize
            rows, cols = cleaned.shape
            extent = [
                origin[0],
                origin[0] + cols * resolution,
                origin[1],
                origin[1] + rows * resolution
            ]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original cleaned
            axes[0].imshow(cleaned.astype(float), origin='lower', extent=extent, cmap='gray')
            axes[0].set_title("Cleaned navigable area")

            # Initial skeleton
            display1 = np.zeros((*cleaned.shape, 3))
            display1[cleaned] = [0.8, 0.8, 0.8]
            display1[skeleton1] = [1, 0, 0]
            axes[1].imshow(display1, origin='lower', extent=extent)
            axes[1].set_title(f"Initial skeleton ({np.sum(skeleton1)} px)")

            # Final skeleton
            display2 = np.zeros((*cleaned.shape, 3))
            display2[cleaned] = [0.8, 0.8, 0.8]
            display2[skeleton2] = [0, 0, 1]
            axes[2].imshow(display2, origin='lower', extent=extent)
            axes[2].set_title(f"After dilation={dilation_radius} ({np.sum(skeleton2)} px, {num_components} comp)")

            plt.tight_layout()
            output = os.path.join(os.path.dirname(bag_path), "skeleton_dilation_test.png")
            plt.savefig(output, dpi=150)
            print(f"Saved to {output}")
            plt.close()

            return skeleton2, dilation_radius

    print("No good configuration found")
    return None, None


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    test_dilate_skeleton(bag_path)
