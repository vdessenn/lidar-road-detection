#!/usr/bin/env python3
"""Debug centerline path to understand where it goes and where it stops."""

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


def debug_path(bag_path: str, resolution: float = 0.5):
    """Debug the centerline path extraction."""

    # Extract navigable grid
    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    # Process to get navigable grid
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)
    closing_radius_px = max(1, int(2.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)
    min_size_pixels = int(50 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)
    safety_radius_px = max(1, int(0.50 / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    # Skeletonize
    skeleton = skeletonize(navigable)

    # Count neighbors
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), neighbor_kernel, mode='constant')

    endpoints = skeleton & (neighbor_count == 1)
    endpoint_coords = np.argwhere(endpoints)

    print(f"Endpoints (world coordinates):")
    for i, (row, col) in enumerate(endpoint_coords):
        world_x = origin[0] + col * resolution
        world_y = origin[1] + row * resolution
        print(f"  E{i}: ({world_x:.1f}, {world_y:.1f}) - row={row}, col={col}")

    # Now trace the path and record junctions and decisions
    dist_transform = ndimage.distance_transform_edt(navigable)

    def get_neighbors(r, c):
        neighbors = []
        rows, cols = skeleton.shape
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if skeleton[nr, nc]:
                        neighbors.append((nr, nc))
        return neighbors

    # Start from bottom endpoint
    start_idx = np.argmin(endpoint_coords[:, 0])
    start = tuple(endpoint_coords[start_idx])
    print(f"\nStarting from E{start_idx}: row={start[0]}, col={start[1]}")

    ordered = [start]
    visited = {start}
    current = start
    prev_dir = (1, 0)

    junction_decisions = []

    while True:
        neighbors = get_neighbors(current[0], current[1])
        unvisited = [n for n in neighbors if n not in visited]

        if len(unvisited) == 0:
            # Check why we stopped
            world_x = origin[0] + current[1] * resolution
            world_y = origin[1] + current[0] * resolution
            print(f"\nPath stopped at ({world_x:.1f}, {world_y:.1f}) - row={current[0]}, col={current[1]}")
            print(f"  All neighbors: {neighbors}")
            print(f"  All visited neighbors: {[n for n in neighbors if n in visited]}")
            break
        elif len(unvisited) == 1:
            next_point = unvisited[0]
        else:
            # Junction - record the decision
            world_x = origin[0] + current[1] * resolution
            world_y = origin[1] + current[0] * resolution

            best_score = -float('inf')
            next_point = unvisited[0]
            current_dist = dist_transform[current[0], current[1]]

            scores = []
            for neighbor in unvisited:
                new_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                straight_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
                right_dir = (-prev_dir[1], prev_dir[0])
                right_score = right_dir[0] * new_dir[0] + right_dir[1] * new_dir[1]
                neighbor_dist = dist_transform[neighbor[0], neighbor[1]]
                dist_score = neighbor_dist / (current_dist + 1.0)
                score = straight_score * 10.0 + dist_score * 3.0 + right_score * 1.5

                n_world_x = origin[0] + neighbor[1] * resolution
                n_world_y = origin[1] + neighbor[0] * resolution
                scores.append({
                    'neighbor': neighbor,
                    'world': (n_world_x, n_world_y),
                    'straight': straight_score,
                    'right': right_score,
                    'dist': dist_score,
                    'total': score
                })

                if score > best_score:
                    best_score = score
                    next_point = neighbor

            junction_decisions.append({
                'position': (world_x, world_y),
                'choices': scores,
                'chosen': next_point,
                'path_index': len(ordered)
            })

        prev_dir = (next_point[0] - current[0], next_point[1] - current[1])
        visited.add(next_point)
        ordered.append(next_point)
        current = next_point

    print(f"\nTotal ordered points: {len(ordered)}")
    print(f"Total skeleton points: {np.sum(skeleton)}")
    print(f"Unvisited skeleton points: {np.sum(skeleton) - len(ordered)}")

    print(f"\n=== Junction Decisions ===")
    for jd in junction_decisions:
        print(f"\nJunction at ({jd['position'][0]:.1f}, {jd['position'][1]:.1f}) - path index {jd['path_index']}:")
        for choice in jd['choices']:
            marker = "CHOSEN" if choice['neighbor'] == jd['chosen'] else "      "
            print(f"  {marker} -> ({choice['world'][0]:.1f}, {choice['world'][1]:.1f}): "
                  f"straight={choice['straight']:.1f}, right={choice['right']:.1f}, "
                  f"dist={choice['dist']:.2f}, total={choice['total']:.1f}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 12))

    rows, cols = navigable.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]
    display[skeleton] = [0.3, 0.3, 0.3]

    ax.imshow(display, origin='lower', extent=extent)

    # Plot the path with color gradient
    ordered_arr = np.array(ordered)
    path_x = origin[0] + ordered_arr[:, 1] * resolution
    path_y = origin[1] + ordered_arr[:, 0] * resolution

    colors = plt.cm.viridis(np.linspace(0, 1, len(path_x)))
    for i in range(len(path_x) - 1):
        ax.plot(path_x[i:i+2], path_y[i:i+2], color=colors[i], linewidth=2)

    # Mark start and end
    ax.plot(path_x[0], path_y[0], 'go', markersize=15, label='Start')
    ax.plot(path_x[-1], path_y[-1], 'ro', markersize=15, label='End')

    # Mark all endpoints
    for i, (row, col) in enumerate(endpoint_coords):
        ex = origin[0] + col * resolution
        ey = origin[1] + row * resolution
        ax.plot(ex, ey, 'm^', markersize=10)
        ax.annotate(f'E{i}', (ex, ey), fontsize=8, color='magenta')

    # Mark junction decisions
    for jd in junction_decisions:
        ax.plot(jd['position'][0], jd['position'][1], 'y*', markersize=8)

    ax.set_title(f"Path Debug\nOrdered: {len(ordered)}, Skeleton: {np.sum(skeleton)}")
    ax.legend()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    output_path = os.path.join(os.path.dirname(bag_path), "path_debug.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    debug_path(bag_path)
