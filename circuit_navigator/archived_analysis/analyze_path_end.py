#!/usr/bin/env python3
"""Analyze where the path ends and what skeleton points remain unvisited."""

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


def analyze_path_end(bag_path: str, resolution: float = 0.5):
    """Analyze where path ends and what's left."""

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

    # Extract centerline
    cl_extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = cl_extractor.extract(navigable, resolution, origin, method="skeleton")

    # Get the skeleton and pruned version
    skeleton = skeletonize(navigable)

    # Get ordered points
    ordered_pixels = centerline.points_pixel

    print(f"Skeleton: {np.sum(skeleton)} pixels")
    print(f"Ordered path: {len(ordered_pixels)} points")
    print(f"Path length: {centerline.length_m:.1f}m")

    # Find unvisited skeleton points
    skeleton_coords = np.argwhere(skeleton)
    visited_set = {tuple(p.astype(int)) for p in ordered_pixels}
    unvisited = [tuple(c) for c in skeleton_coords if tuple(c) not in visited_set]

    print(f"\nUnvisited skeleton points: {len(unvisited)}")

    # Show where path ends
    if len(ordered_pixels) > 0:
        end_pixel = ordered_pixels[-1]
        end_world = centerline.points_world[-1]
        print(f"\nPath ends at: pixel=({end_pixel[0]:.0f}, {end_pixel[1]:.0f}), world=({end_world[0]:.1f}, {end_world[1]:.1f})")

    # Show where unvisited points are
    if unvisited:
        unvisited_arr = np.array(unvisited)
        for u in unvisited[:20]:  # Show first 20
            world_x = origin[0] + u[1] * resolution
            world_y = origin[1] + u[0] * resolution
            print(f"  Unvisited: ({world_x:.1f}, {world_y:.1f})")

    # Visualize
    rows, cols = navigable.shape
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Full track view
    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]
    display[skeleton] = [0.5, 0.5, 0.5]

    # Mark visited in blue
    for p in ordered_pixels:
        r, c = int(p[0]), int(p[1])
        if 0 <= r < rows and 0 <= c < cols:
            display[r, c] = [0, 0, 1]

    # Mark unvisited in red
    for u in unvisited:
        display[u[0], u[1]] = [1, 0, 0]

    axes[0].imshow(display, origin='lower', extent=extent)
    axes[0].set_title(f"Blue=visited ({len(ordered_pixels)}), Red=unvisited ({len(unvisited)})")

    # Mark start and end
    if len(centerline.points_world) > 0:
        axes[0].plot(centerline.points_world[0, 0], centerline.points_world[0, 1], 'g^', markersize=15, label='Start')
        axes[0].plot(centerline.points_world[-1, 0], centerline.points_world[-1, 1], 'rv', markersize=15, label='End')
        axes[0].legend()

    # Zoomed view of second roundabout area
    display2 = display.copy()
    axes[1].imshow(display2, origin='lower', extent=extent)
    axes[1].set_xlim(-130, -80)
    axes[1].set_ylim(110, 155)
    axes[1].set_title("Zoomed: Second roundabout area")

    if len(centerline.points_world) > 0:
        axes[1].plot(centerline.points_world[-1, 0], centerline.points_world[-1, 1], 'rv', markersize=15)

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "path_end_analysis.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_path_end(bag_path)
