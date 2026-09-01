#!/usr/bin/env python3
"""Check skeleton connectivity near the start point."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def check_start_area(bag_path: str, resolution: float = 0.5):
    """Check skeleton connectivity near start."""

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

    def pixel_to_world(row, col):
        wx = origin[0] + col * resolution
        wy = origin[1] + row * resolution
        return wx, wy

    # Start point is (102, 63), end is (126, 20)
    start_r, start_c = 102, 63
    end_r, end_c = 126, 20

    start_wx, start_wy = pixel_to_world(start_r, start_c)
    end_wx, end_wy = pixel_to_world(end_r, end_c)

    print(f"Start: pixel=({start_r}, {start_c}), world=({start_wx:.1f}, {start_wy:.1f})")
    print(f"End: pixel=({end_r}, {end_c}), world=({end_wx:.1f}, {end_wy:.1f})")
    print(f"Distance: {np.sqrt((start_r - end_r)**2 + (start_c - end_c)**2) * resolution:.1f}m")

    # Check skeleton points in the bottom area (rows 80-130)
    print("\nSkeleton points in bottom area (Y = 33-60):")
    bottom_points = []
    for r in range(80, 130):
        for c in range(cols):
            if skeleton[r, c]:
                wx, wy = pixel_to_world(r, c)
                bottom_points.append((r, c, wx, wy))

    print(f"Found {len(bottom_points)} skeleton points")
    if bottom_points:
        # Group by approximate X position
        by_x = {}
        for r, c, wx, wy in bottom_points:
            x_key = int(wx / 5) * 5  # Group by 5m
            if x_key not in by_x:
                by_x[x_key] = []
            by_x[x_key].append((r, c, wx, wy))

        for x_key in sorted(by_x.keys()):
            pts = by_x[x_key]
            y_vals = [p[3] for p in pts]
            print(f"  X~{x_key}: {len(pts)} points, Y from {min(y_vals):.1f} to {max(y_vals):.1f}")

    # Check if there's a path between start and end in the skeleton
    print(f"\nChecking connectivity between start and end...")

    # Use BFS to find shortest path
    from collections import deque

    def get_neighbors(r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and skeleton[nr, nc]:
                    neighbors.append((nr, nc))
        return neighbors

    queue = deque([(start_r, start_c, 0)])
    visited = {(start_r, start_c)}
    found = False

    while queue:
        r, c, dist = queue.popleft()
        if abs(r - end_r) <= 1 and abs(c - end_c) <= 1:
            print(f"Path exists! Distance: {dist} steps")
            found = True
            break

        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    if not found:
        print("No direct path in skeleton between start and end")
        print(f"Visited {len(visited)} skeleton points during search")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    check_start_area(bag_path)
