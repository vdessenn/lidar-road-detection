#!/usr/bin/env python3
"""Find where the return path is in the track data."""

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


def find_return_path(bag_path: str, resolution: float = 0.5):
    """Find where the return path is."""

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

    # Check for navigable cells at different X positions going DOWN from Y=130 to Y=50
    print("Searching for return path (navigable cells going DOWN):")
    print("Looking for vertical strips of navigable cells from Y=130 to Y=50\n")

    # Convert world to pixel
    def world_to_pixel(wx, wy):
        col = int((wx - origin[0]) / resolution)
        row = int((wy - origin[1]) / resolution)
        return row, col

    # Check different X positions
    for world_x in [-165, -160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110, -105, -100, -95, -90, -85]:
        count = 0
        skeleton_count = 0
        for world_y in range(50, 130):
            r, c = world_to_pixel(world_x, world_y)
            if 0 <= r < rows and 0 <= c < cols:
                if navigable[r, c]:
                    count += 1
                if skeleton[r, c]:
                    skeleton_count += 1
        if count > 10:  # At least some navigable cells
            print(f"  X={world_x}: {count} navigable cells, {skeleton_count} skeleton cells")

    # Look for the second roundabout exit going south
    print("\n\nSearching for path from second roundabout going SOUTH:")
    # Second roundabout is around X=-85 to -120, Y=125-150
    # The exit going south should be around X=-110 to -120 (west side of roundabout)

    for world_x in range(-125, -100, 2):
        path_exists = True
        print(f"\n  Checking X={world_x}:")
        for world_y in [125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]:
            r, c = world_to_pixel(world_x, world_y)
            if 0 <= r < rows and 0 <= c < cols:
                has_nav = navigable[r, c]
                has_skel = skeleton[r, c]
                if has_nav:
                    marker = "NAV+SKEL" if has_skel else "NAV"
                else:
                    marker = "---"
                    path_exists = False
            else:
                marker = "OOB"
                path_exists = False
            print(f"    Y={world_y}: {marker}")

    # Visualize
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, ax = plt.subplots(figsize=(12, 12))

    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.7, 0.9, 0.7]
    display[skeleton] = [0, 0, 1]

    ax.imshow(display, origin='lower', extent=extent)

    # Draw vertical lines at key X positions to show where return path should be
    for x in [-165, -120, -110, -100, -85]:
        ax.axvline(x=x, color='red', linestyle='--', alpha=0.5, label=f'X={x}')

    # Draw horizontal lines at key Y positions
    for y in [50, 75, 100, 125]:
        ax.axhline(y=y, color='orange', linestyle=':', alpha=0.5)

    ax.set_title("Track with vertical reference lines\nGreen=navigable, Blue=skeleton")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "return_path_search.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    find_return_path(bag_path)
