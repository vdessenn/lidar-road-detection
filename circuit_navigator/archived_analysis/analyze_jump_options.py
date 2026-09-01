#!/usr/bin/env python3
"""Analyze what jump options are available from the path end point."""

import numpy as np
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def analyze_jump_options(bag_path: str, resolution: float = 0.5):
    """Analyze jump options from path end."""

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

    def world_to_pixel(wx, wy):
        col = int((wx - origin[0]) / resolution)
        row = int((wy - origin[1]) / resolution)
        return row, col

    # Path end from debug output: (-82.9, 136.1) -> convert to pixel
    path_end_wx, path_end_wy = -82.9, 136.1
    path_end_r, path_end_c = world_to_pixel(path_end_wx, path_end_wy)

    print(f"Path end: world=({path_end_wx:.1f}, {path_end_wy:.1f}), pixel=({path_end_r}, {path_end_c})")

    # Previous direction at path end: was going EAST (0, 1) roughly
    prev_dir = (0.0, 1.0)

    # Get ALL skeleton points
    skeleton_coords = np.argwhere(skeleton)
    print(f"Total skeleton points: {len(skeleton_coords)}")

    # Simulate visited set (all skeleton points from path start to end)
    # For simplicity, let's say we visited ~300 points
    # Find unvisited points

    dist_transform = ndimage.distance_transform_edt(navigable)

    # Find all skeleton points and their distance from path end
    distances = []
    for coord in skeleton_coords:
        dr = coord[0] - path_end_r
        dc = coord[1] - path_end_c
        dist = np.sqrt(dr*dr + dc*dc)
        wx, wy = pixel_to_world(coord[0], coord[1])
        distances.append((dist, coord[0], coord[1], wx, wy))

    distances.sort()

    print(f"\nNearest skeleton points from path end:")
    for i, (dist, r, c, wx, wy) in enumerate(distances[:20]):
        # Calculate direction score
        dr = r - path_end_r
        dc = c - path_end_c
        if dist > 0:
            new_dir = (dr / dist, dc / dist)
            dir_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
        else:
            dir_score = 0

        # Y direction (going DOWN = returning to first roundabout)
        going_down = wy < path_end_wy

        print(f"  {i+1}. {dist * resolution:.1f}m: ({wx:.1f}, {wy:.1f}) dir_score={dir_score:.2f} {'↓DOWN' if going_down else ''}")

    # What points are in the "return path" direction? (going DOWN/SOUTH)
    print(f"\nSkeleton points going SOUTH from path end (Y < {path_end_wy:.1f}):")
    south_points = [(d, r, c, wx, wy) for d, r, c, wx, wy in distances
                    if wy < path_end_wy - 5]  # At least 5m south
    print(f"Found {len(south_points)} points going south")

    if south_points:
        print("First 10 southern points:")
        for i, (dist, r, c, wx, wy) in enumerate(south_points[:10]):
            print(f"  {dist * resolution:.1f}m: ({wx:.1f}, {wy:.1f})")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_jump_options(bag_path)
