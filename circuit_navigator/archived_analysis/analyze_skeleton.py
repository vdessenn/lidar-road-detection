#!/usr/bin/env python3
"""
Analyze skeleton structure to understand roundabout junctions.
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


def analyze_skeleton(bag_path: str, resolution: float = 0.5):
    """Analyze skeleton structure to find junctions."""

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

    # Count neighbors for each skeleton pixel
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), neighbor_kernel, mode='constant')

    # Find different types of points
    endpoints = skeleton & (neighbor_count == 1)  # Dead ends
    junctions = skeleton & (neighbor_count >= 3)   # Branch points
    path_points = skeleton & (neighbor_count == 2) # Regular path

    endpoint_coords = np.argwhere(endpoints)
    junction_coords = np.argwhere(junctions)

    print(f"Skeleton analysis:")
    print(f"  Total skeleton pixels: {np.sum(skeleton)}")
    print(f"  Endpoints (1 neighbor): {len(endpoint_coords)}")
    print(f"  Junctions (3+ neighbors): {len(junction_coords)}")
    print(f"  Path points (2 neighbors): {np.sum(path_points)}")

    # Convert to world coordinates for display
    rows, cols = navigable.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Skeleton with junction types highlighted
    ax1 = axes[0]
    display = np.zeros((*skeleton.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]  # Gray navigable
    display[skeleton] = [0.3, 0.3, 1.0]    # Blue skeleton
    display[junctions] = [1.0, 0.0, 0.0]   # Red junctions
    display[endpoints] = [0.0, 1.0, 0.0]   # Green endpoints

    ax1.imshow(display, origin='lower', extent=extent)
    ax1.set_title(f"Skeleton Structure\nBlue=path, Red=junctions ({len(junction_coords)}), Green=endpoints ({len(endpoint_coords)})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Label junctions with their coordinates
    for i, (row, col) in enumerate(junction_coords):
        world_x = origin[0] + col * resolution
        world_y = origin[1] + row * resolution
        ax1.annotate(f'J{i}', (world_x, world_y), fontsize=8, color='red',
                    ha='center', va='bottom')

    # Panel 2: Zoomed view of first roundabout area
    ax2 = axes[1]
    ax2.imshow(display, origin='lower', extent=extent)
    ax2.set_xlim(-180, -155)  # Zoom to first roundabout
    ax2.set_ylim(40, 85)
    ax2.set_title("First Roundabout Area (zoomed)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Label junctions in zoomed area
    for i, (row, col) in enumerate(junction_coords):
        world_x = origin[0] + col * resolution
        world_y = origin[1] + row * resolution
        if -180 < world_x < -155 and 40 < world_y < 85:
            ax2.annotate(f'J{i}\n({world_x:.0f},{world_y:.0f})',
                        (world_x, world_y), fontsize=8, color='white',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(bag_path), "skeleton_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis to {output_path}")
    plt.close()

    # Print junction details
    print(f"\nJunction coordinates (world):")
    for i, (row, col) in enumerate(junction_coords):
        world_x = origin[0] + col * resolution
        world_y = origin[1] + row * resolution
        num_neighbors = neighbor_count[row, col]
        print(f"  J{i}: ({world_x:.1f}, {world_y:.1f}) - {num_neighbors} neighbors")

    return {
        'skeleton': skeleton,
        'junctions': junction_coords,
        'endpoints': endpoint_coords,
        'navigable': navigable,
        'origin': origin,
        'resolution': resolution
    }


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_skeleton(bag_path)
