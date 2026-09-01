#!/usr/bin/env python3
"""Test medial axis transform for centerline extraction."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, opening, medial_axis

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def test_medial_axis(bag_path: str, resolution: float = 0.5):
    """Test medial axis for centerline."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Processing
    closing_radius_px = max(1, int(3.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    min_size_pixels = int(100 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    # Medial axis transform - returns skeleton AND distance
    skel, distance = medial_axis(cleaned, return_distance=True)

    # Check connectivity
    labeled, num_components = ndimage.label(skel)
    print(f"Medial axis: {np.sum(skel)} pixels in {num_components} components")

    if num_components > 0:
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        sizes.sort(key=lambda x: -x[1])
        print(f"Largest components:")
        for comp_id, size in sizes[:5]:
            print(f"  Component {comp_id}: {size} pixels")

    # Also try thickening the area first
    print("\nTrying with dilation first:")
    dilation_kernel = disk(3)
    thickened = ndimage.binary_dilation(cleaned, dilation_kernel)
    skel2, dist2 = medial_axis(thickened, return_distance=True)

    labeled2, num_comp2 = ndimage.label(skel2)
    print(f"Medial axis (thickened): {np.sum(skel2)} pixels in {num_comp2} components")

    if num_comp2 > 0:
        sizes2 = [(i, np.sum(labeled2 == i)) for i in range(1, num_comp2 + 1)]
        sizes2.sort(key=lambda x: -x[1])
        print(f"Largest components:")
        for comp_id, size in sizes2[:5]:
            print(f"  Component {comp_id}: {size} pixels")

    # Visualize
    rows, cols = cleaned.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Cleaned area
    axes[0, 0].imshow(cleaned.astype(float), origin='lower', extent=extent, cmap='Greys')
    axes[0, 0].set_title("Cleaned navigable area")

    # Medial axis
    display1 = np.zeros((*cleaned.shape, 3))
    display1[cleaned] = [0.8, 0.8, 0.8]
    display1[skel] = [1, 0, 0]
    axes[0, 1].imshow(display1, origin='lower', extent=extent)
    axes[0, 1].set_title(f"Medial axis: {np.sum(skel)} px, {num_components} comp")

    # Distance transform on medial axis
    dist_on_skel = distance * skel
    im = axes[1, 0].imshow(dist_on_skel, origin='lower', extent=extent, cmap='hot')
    axes[1, 0].set_title("Distance on medial axis (track width)")
    plt.colorbar(im, ax=axes[1, 0])

    # Thickened medial axis
    display2 = np.zeros((*thickened.shape, 3))
    display2[thickened] = [0.8, 0.8, 0.8]
    display2[skel2] = [0, 0, 1]
    axes[1, 1].imshow(display2, origin='lower', extent=extent)
    axes[1, 1].set_title(f"Medial axis (thickened): {np.sum(skel2)} px, {num_comp2} comp")

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "medial_axis_test.png")
    plt.savefig(output, dpi=150)
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    test_medial_axis(bag_path)
