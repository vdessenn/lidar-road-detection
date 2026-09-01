#!/usr/bin/env python3
"""Debug junction decisions during path ordering."""

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


def debug_junction_decisions(bag_path: str, resolution: float = 0.5):
    """Debug junction decisions step by step."""

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

    def count_neighbors_arr(skeleton):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.convolve(skeleton.astype(int), kernel, mode='constant')

    # Find skeleton points
    skeleton_coords = np.argwhere(skeleton)
    skeleton_set = {tuple(c) for c in skeleton_coords}

    # Distance transform for scoring
    dist_transform = ndimage.distance_transform_edt(navigable)

    # Find endpoints and junctions
    neighbor_count = count_neighbors_arr(skeleton)
    endpoints = skeleton & (neighbor_count == 1)
    endpoint_coords = np.argwhere(endpoints)

    # Start from bottom-most endpoint
    if len(endpoint_coords) > 0:
        start_idx = np.argmin(endpoint_coords[:, 0])
        start = tuple(endpoint_coords[start_idx])
    else:
        start_idx = np.argmin(skeleton_coords[:, 0])
        start = tuple(skeleton_coords[start_idx])

    print(f"Starting from: pixel=({start[0]}, {start[1]}), world=({pixel_to_world(*start)[0]:.1f}, {pixel_to_world(*start)[1]:.1f})")

    # Traverse and log junction decisions
    ordered = [start]
    visited = {start}
    current = start
    prev_dir = (1, 0)  # Initial direction: going up

    junction_count = 0

    while True:
        neighbors = get_neighbors(current[0], current[1])
        unvisited = [n for n in neighbors if n not in visited]

        if len(unvisited) == 0:
            # Check if we can jump
            break

        if len(unvisited) >= 2:
            # This is a junction - log the decision
            junction_count += 1
            wx, wy = pixel_to_world(*current)

            # Only log junctions in the second roundabout area
            if -125 <= wx <= -80 and 120 <= wy <= 155:
                print(f"\nJunction #{junction_count} at ({wx:.1f}, {wy:.1f}):")
                print(f"  Current direction: ({prev_dir[0]:.2f}, {prev_dir[1]:.2f})")
                print(f"  Options ({len(unvisited)}):")

                current_dist = dist_transform[current[0], current[1]]
                scores = []

                for neighbor in unvisited:
                    new_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                    neighbor_wx, neighbor_wy = pixel_to_world(*neighbor)

                    # Direction continuity score
                    straight_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]

                    # Right turn score
                    right_dir = (-prev_dir[1], prev_dir[0])
                    right_score = right_dir[0] * new_dir[0] + right_dir[1] * new_dir[1]

                    # Distance score
                    neighbor_dist = dist_transform[neighbor[0], neighbor[1]]
                    dist_score = neighbor_dist / (current_dist + 1.0)

                    # Combined score
                    score = straight_score * 10.0 + dist_score * 3.0 + right_score * 1.5

                    scores.append((neighbor, score, straight_score, dist_score, right_score))
                    print(f"    -> ({neighbor_wx:.1f}, {neighbor_wy:.1f}): "
                          f"straight={straight_score:.2f}, dist={dist_score:.2f}, right={right_score:.2f}, TOTAL={score:.2f}")

                # Find best
                scores.sort(key=lambda x: -x[1])
                chosen = scores[0][0]
                chosen_wx, chosen_wy = pixel_to_world(*chosen)
                print(f"  CHOSE: ({chosen_wx:.1f}, {chosen_wy:.1f}) with score {scores[0][1]:.2f}")

                next_point = chosen
            else:
                # Just pick best scoring for non-logged junctions
                best_score = -float('inf')
                next_point = unvisited[0]
                current_dist = dist_transform[current[0], current[1]]

                for neighbor in unvisited:
                    new_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                    straight_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
                    right_dir = (-prev_dir[1], prev_dir[0])
                    right_score = right_dir[0] * new_dir[0] + right_dir[1] * new_dir[1]
                    neighbor_dist = dist_transform[neighbor[0], neighbor[1]]
                    dist_score = neighbor_dist / (current_dist + 1.0)
                    score = straight_score * 10.0 + dist_score * 3.0 + right_score * 1.5

                    if score > best_score:
                        best_score = score
                        next_point = neighbor
        else:
            next_point = unvisited[0]

        # Update direction
        prev_dir = (next_point[0] - current[0], next_point[1] - current[1])
        length = np.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
        if length > 0:
            prev_dir = (prev_dir[0] / length, prev_dir[1] / length)

        visited.add(next_point)
        ordered.append(next_point)
        current = next_point

    final_wx, final_wy = pixel_to_world(*current)
    print(f"\n\nPath ended at: ({final_wx:.1f}, {final_wy:.1f})")
    print(f"Total points visited: {len(ordered)}")
    print(f"Junctions encountered: {junction_count}")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    debug_junction_decisions(bag_path)
