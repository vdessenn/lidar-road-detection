#!/usr/bin/env python3
"""Analyze why the path hits a dead end at E3."""

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


def analyze_dead_end(bag_path: str, resolution: float = 0.5):
    """Analyze the dead end at E3."""

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

    skeleton = skeletonize(navigable)

    # E3 is at (-82.9, 136.1) = row=285, col=190
    e3_row, e3_col = 285, 190
    e3_world = (origin[0] + e3_col * resolution, origin[1] + e3_row * resolution)
    print(f"E3 location: world=({e3_world[0]:.1f}, {e3_world[1]:.1f}), pixel=({e3_row}, {e3_col})")

    # Check what's around E3
    print(f"\nAround E3 (20x20 pixel area):")
    print(f"  Navigable area exists: {navigable[e3_row-10:e3_row+10, e3_col-10:e3_col+10].sum()} cells")
    print(f"  Skeleton exists: {skeleton[e3_row-10:e3_row+10, e3_col-10:e3_col+10].sum()} cells")

    # Check where the navigable area continues from E3
    print(f"\nNavigable cells in different directions from E3:")
    for direction, (dr, dc) in [("UP (increasing Y)", (1, 0)),
                                  ("DOWN (decreasing Y)", (-1, 0)),
                                  ("LEFT (decreasing X)", (0, -1)),
                                  ("RIGHT (increasing X)", (0, 1))]:
        count = 0
        for dist in range(1, 50):
            r, c = e3_row + dr * dist, e3_col + dc * dist
            if 0 <= r < navigable.shape[0] and 0 <= c < navigable.shape[1]:
                if navigable[r, c]:
                    count += 1
        print(f"  {direction}: {count} navigable cells in next 25m")

    # Visualize the area around E3
    rows, cols = navigable.shape
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Full track with E3 highlighted
    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.7, 0.9, 0.7]
    display[skeleton] = [0, 0, 1]

    axes[0].imshow(display, origin='lower', extent=extent)
    axes[0].plot(e3_world[0], e3_world[1], 'r*', markersize=20, label='E3 (dead end)')
    axes[0].set_title("Full track - E3 marked with red star")
    axes[0].legend()

    # Zoomed view around E3
    zoom_margin = 15  # meters
    axes[1].imshow(display, origin='lower', extent=extent)
    axes[1].set_xlim(e3_world[0] - zoom_margin, e3_world[0] + zoom_margin)
    axes[1].set_ylim(e3_world[1] - zoom_margin, e3_world[1] + zoom_margin)
    axes[1].plot(e3_world[0], e3_world[1], 'r*', markersize=20)
    axes[1].set_title(f"Zoomed around E3\nGreen=navigable, Blue=skeleton")

    # Mark where the return path SHOULD be (going down-left from E3)
    return_direction = (-1, -1)  # Going down and left to return
    for dist in range(1, 30):
        r = e3_row + return_direction[0] * dist
        c = e3_col + return_direction[1] * dist
        if 0 <= r < rows and 0 <= c < cols:
            wx = origin[0] + c * resolution
            wy = origin[1] + r * resolution
            if navigable[r, c]:
                axes[1].plot(wx, wy, 'g.', markersize=3)
            else:
                axes[1].plot(wx, wy, 'r.', markersize=3)

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "dead_end_analysis.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_dead_end(bag_path)
