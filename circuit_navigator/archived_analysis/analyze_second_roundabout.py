#!/usr/bin/env python3
"""
Analyze the second roundabout area to understand the track structure.
"""

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


def analyze_second_roundabout(bag_path: str, resolution: float = 0.5):
    """Zoom into the second roundabout area."""

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

    rows, cols = navigable.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    # Create zoomed visualization of second roundabout
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Full track with second roundabout highlighted
    ax1 = axes[0]
    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.7, 0.9, 0.7]  # Light green
    display[skeleton] = [0.0, 0.0, 1.0]    # Blue skeleton

    ax1.imshow(display, origin='lower', extent=extent)
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=150, color='red', linestyle='--', alpha=0.5)
    ax1.set_title("Full Track - Second roundabout area between red lines")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Panel 2: Zoomed second roundabout
    ax2 = axes[1]
    ax2.imshow(display, origin='lower', extent=extent)
    ax2.set_xlim(-130, -80)  # Zoom to second roundabout
    ax2.set_ylim(115, 155)
    ax2.set_title("Second Roundabout Area (zoomed)\nLooking for return path")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Find skeleton points in this area
    skeleton_coords = np.argwhere(skeleton)
    for row, col in skeleton_coords:
        world_x = origin[0] + col * resolution
        world_y = origin[1] + row * resolution
        if -130 < world_x < -80 and 115 < world_y < 155:
            ax2.plot(world_x, world_y, 'b.', markersize=3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(bag_path), "second_roundabout_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_second_roundabout(bag_path)
