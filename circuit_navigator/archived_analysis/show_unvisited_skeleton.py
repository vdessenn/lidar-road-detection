#!/usr/bin/env python3
"""Show where unvisited skeleton points are."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor
from circuit_navigator.map_processing.centerline import CenterlineExtractor


def show_unvisited(bag_path: str, resolution: float = 0.5):
    """Show unvisited skeleton points."""

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
    safety_radius_px = max(1, int(0.50 / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    # Get centerline
    cl_extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = cl_extractor.extract(navigable, resolution, origin, method="skeleton")

    skeleton = skeletonize(navigable)
    rows, cols = navigable.shape

    def pixel_to_world(row, col):
        wx = origin[0] + col * resolution
        wy = origin[1] + row * resolution
        return wx, wy

    # Find visited and unvisited
    skeleton_coords = np.argwhere(skeleton)
    visited_set = {tuple(p.astype(int)) for p in centerline.points_pixel}
    unvisited = [tuple(c) for c in skeleton_coords if tuple(c) not in visited_set]

    print(f"Skeleton: {len(skeleton_coords)} pixels")
    print(f"Visited: {len(visited_set)} points")
    print(f"Unvisited: {len(unvisited)} points")

    # Analyze unvisited points by region
    print("\nUnvisited points by Y range:")
    y_ranges = [(0, 50), (50, 100), (100, 130), (130, 160)]
    for y_min, y_max in y_ranges:
        in_range = []
        for r, c in unvisited:
            wx, wy = pixel_to_world(r, c)
            if y_min <= wy < y_max:
                in_range.append((wx, wy))
        if in_range:
            x_values = [p[0] for p in in_range]
            print(f"  Y={y_min}-{y_max}: {len(in_range)} points, X from {min(x_values):.1f} to {max(x_values):.1f}")

    print("\nFirst 20 unvisited points:")
    for r, c in unvisited[:20]:
        wx, wy = pixel_to_world(r, c)
        print(f"  ({wx:.1f}, {wy:.1f})")

    # Visualize
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, ax = plt.subplots(figsize=(14, 10))

    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]

    # Blue for visited
    for p in centerline.points_pixel:
        r, c = int(p[0]), int(p[1])
        if 0 <= r < rows and 0 <= c < cols:
            display[r, c] = [0, 0, 1]

    # Red for unvisited
    for r, c in unvisited:
        display[r, c] = [1, 0, 0]

    ax.imshow(display, origin='lower', extent=extent)
    ax.set_title(f"Blue = visited ({len(visited_set)}), Red = unvisited ({len(unvisited)})")

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "unvisited_skeleton.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    show_unvisited(bag_path)
