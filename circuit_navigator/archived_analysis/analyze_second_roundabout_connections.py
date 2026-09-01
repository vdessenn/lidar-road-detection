#!/usr/bin/env python3
"""Analyze connections around the second roundabout."""

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
    """Analyze skeleton connections around second roundabout."""

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

    # Helper functions
    def world_to_pixel(wx, wy):
        col = int((wx - origin[0]) / resolution)
        row = int((wy - origin[1]) / resolution)
        return row, col

    def pixel_to_world(row, col):
        wx = origin[0] + col * resolution
        wy = origin[1] + row * resolution
        return wx, wy

    def count_neighbors(r, c):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and skeleton[nr, nc]:
                    count += 1
        return count

    # Find all skeleton points around the second roundabout (Y = 125-150, X = -125 to -85)
    print("Skeleton points around second roundabout:")
    print("Region: X = -125 to -85, Y = 120 to 150")
    print("-" * 60)

    skeleton_points = []
    junction_points = []
    endpoint_points = []

    for r in range(rows):
        for c in range(cols):
            if skeleton[r, c]:
                wx, wy = pixel_to_world(r, c)
                if -125 <= wx <= -85 and 120 <= wy <= 150:
                    n = count_neighbors(r, c)
                    skeleton_points.append((r, c, wx, wy, n))
                    if n >= 3:
                        junction_points.append((r, c, wx, wy, n))
                    elif n == 1:
                        endpoint_points.append((r, c, wx, wy))

    print(f"Total skeleton points in region: {len(skeleton_points)}")
    print(f"Junction points (3+ neighbors): {len(junction_points)}")
    print(f"Endpoint points (1 neighbor): {len(endpoint_points)}")

    print("\nJunction points:")
    for r, c, wx, wy, n in junction_points:
        print(f"  ({wx:.1f}, {wy:.1f}) - {n} neighbors")

    print("\nEndpoint points:")
    for r, c, wx, wy in endpoint_points:
        print(f"  ({wx:.1f}, {wy:.1f})")

    # Find where the path currently ends and what's nearby
    path_end_r, path_end_c = 272, 115  # From earlier analysis
    path_end_wx, path_end_wy = pixel_to_world(path_end_r, path_end_c)
    print(f"\nCurrent path end: ({path_end_wx:.1f}, {path_end_wy:.1f})")

    # Find nearby skeleton points (within 20 pixels)
    print("\nSkeleton points near path end (within 10m):")
    search_radius = int(10 / resolution)  # 10m = 20 pixels
    nearby = []
    for r, c, wx, wy, n in skeleton_points:
        dist = np.sqrt((r - path_end_r)**2 + (c - path_end_c)**2)
        if dist <= search_radius and dist > 0:
            nearby.append((dist, wx, wy, n))

    nearby.sort()
    for dist, wx, wy, n in nearby[:10]:
        print(f"  {dist * resolution:.1f}m: ({wx:.1f}, {wy:.1f}) - {n} neighbors")

    # Visualize the second roundabout area
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create display
    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.7, 0.9, 0.7]

    # Color skeleton by neighbor count
    for r, c in np.argwhere(skeleton):
        n = count_neighbors(r, c)
        if n >= 3:
            display[r, c] = [1, 0, 0]  # Red for junctions
        elif n == 1:
            display[r, c] = [1, 1, 0]  # Yellow for endpoints
        else:
            display[r, c] = [0, 0, 1]  # Blue for regular

    ax.imshow(display, origin='lower', extent=extent)

    # Zoom to second roundabout
    ax.set_xlim(-130, -80)
    ax.set_ylim(115, 155)

    # Mark path end
    ax.plot(path_end_wx, path_end_wy, 'k*', markersize=20, label='Path end')

    # Draw circles at junctions and endpoints
    for r, c, wx, wy, n in junction_points:
        ax.plot(wx, wy, 'ro', markersize=10)
    for r, c, wx, wy in endpoint_points:
        ax.plot(wx, wy, 'yo', markersize=8)

    ax.set_title("Second Roundabout Skeleton\nRed=junction, Yellow=endpoint, Blue=regular, Black*=path end")
    ax.legend()

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "second_roundabout_connections.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_second_roundabout(bag_path)
