#!/usr/bin/env python3
"""Analyze unvisited skeleton points to find the return path."""

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


def analyze_unvisited(bag_path: str, resolution: float = 0.5):
    """Analyze skeleton to find all connected components."""

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

    # Skeletonize
    skeleton = skeletonize(navigable)

    # Find connected components in skeleton
    labeled, num_components = ndimage.label(skeleton)
    print(f"Skeleton has {num_components} connected components")

    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labeled == i)
        coords = np.argwhere(labeled == i)
        min_y = coords[:, 0].min()
        max_y = coords[:, 0].max()
        min_x = coords[:, 1].min()
        max_x = coords[:, 1].max()

        # Convert to world
        world_min_x = origin[0] + min_x * resolution
        world_max_x = origin[0] + max_x * resolution
        world_min_y = origin[1] + min_y * resolution
        world_max_y = origin[1] + max_y * resolution

        component_sizes.append({
            'id': i,
            'size': size,
            'world_bounds': (world_min_x, world_max_x, world_min_y, world_max_y)
        })
        print(f"  Component {i}: {size} pixels, X:[{world_min_x:.1f}, {world_max_x:.1f}], Y:[{world_min_y:.1f}, {world_max_y:.1f}]")

    # Visualize components in different colors
    rows, cols = navigable.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    fig, ax = plt.subplots(figsize=(14, 12))

    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]

    # Color each component differently
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_components, 1)))
    for i in range(1, num_components + 1):
        mask = labeled == i
        display[mask] = colors[i-1][:3]

    ax.imshow(display, origin='lower', extent=extent)

    # Add legend
    for i, comp in enumerate(component_sizes):
        ax.plot([], [], 'o', color=colors[i][:3],
                label=f"Component {comp['id']}: {comp['size']} px")

    ax.set_title(f"Skeleton Connected Components ({num_components} total)")
    ax.legend(loc='upper right')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    output_path = os.path.join(os.path.dirname(bag_path), "skeleton_components.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    plt.close()

    # Now check if we can connect components by finding closest points
    if num_components > 1:
        print("\nFinding gaps between components:")
        for i in range(1, num_components + 1):
            for j in range(i + 1, num_components + 1):
                coords_i = np.argwhere(labeled == i)
                coords_j = np.argwhere(labeled == j)

                # Find minimum distance between components
                min_dist = float('inf')
                closest_pair = None

                # Sample points for efficiency
                sample_i = coords_i[::max(1, len(coords_i)//100)]
                sample_j = coords_j[::max(1, len(coords_j)//100)]

                for pi in sample_i:
                    for pj in sample_j:
                        d = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)
                        if d < min_dist:
                            min_dist = d
                            closest_pair = (pi, pj)

                if closest_pair:
                    p1, p2 = closest_pair
                    w1 = (origin[0] + p1[1] * resolution, origin[1] + p1[0] * resolution)
                    w2 = (origin[0] + p2[1] * resolution, origin[1] + p2[0] * resolution)
                    dist_m = min_dist * resolution
                    print(f"  Component {i} <-> {j}: {dist_m:.1f}m apart")
                    print(f"    Point 1: ({w1[0]:.1f}, {w1[1]:.1f})")
                    print(f"    Point 2: ({w2[0]:.1f}, {w2[1]:.1f})")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_unvisited(bag_path)
