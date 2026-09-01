#!/usr/bin/env python3
"""Analyze gaps in the skeleton to understand why path doesn't complete."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, skeletonize
from scipy.ndimage import label

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def analyze_skeleton_gaps(bag_path: str, resolution: float = 0.5):
    """Analyze skeleton gaps."""

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

    # Find connected components
    labeled, num_features = label(skeleton)
    print(f"Skeleton connected components: {num_features}")

    # For each component, find its size and location
    components = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        coords = np.argwhere(mask)
        size = len(coords)
        center_row = np.mean(coords[:, 0])
        center_col = np.mean(coords[:, 1])
        center_x = origin[0] + center_col * resolution
        center_y = origin[1] + center_row * resolution
        components.append({
            'id': i,
            'size': size,
            'center': (center_x, center_y),
            'row_range': (coords[:, 0].min(), coords[:, 0].max()),
            'col_range': (coords[:, 1].min(), coords[:, 1].max())
        })

    # Sort by size
    components.sort(key=lambda c: -c['size'])

    print("\nLargest components:")
    for c in components[:10]:
        print(f"  Component {c['id']}: {c['size']} pixels, center=({c['center'][0]:.1f}, {c['center'][1]:.1f})")

    # Find path end point (pixel=(272, 115))
    end_row, end_col = 272, 115
    end_x = origin[0] + end_col * resolution
    end_y = origin[1] + end_row * resolution
    print(f"\nPath end: pixel=({end_row}, {end_col}), world=({end_x:.1f}, {end_y:.1f})")

    # Find nearest unvisited skeleton point from the path end
    skeleton_coords = np.argwhere(skeleton)
    end_component = labeled[end_row, end_col]
    print(f"Path end is in component: {end_component}")

    # Find distance from end point to each skeleton point in other components
    min_dist = float('inf')
    nearest = None
    for coord in skeleton_coords:
        if labeled[coord[0], coord[1]] != end_component:
            dist = np.sqrt((coord[0] - end_row)**2 + (coord[1] - end_col)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = coord

    if nearest is not None:
        nearest_x = origin[0] + nearest[1] * resolution
        nearest_y = origin[1] + nearest[0] * resolution
        print(f"Nearest point in other component: pixel=({nearest[0]}, {nearest[1]}), world=({nearest_x:.1f}, {nearest_y:.1f})")
        print(f"Distance: {min_dist:.1f} pixels = {min_dist * resolution:.1f}m")
        print(f"That component: {labeled[nearest[0], nearest[1]]}")

    # Visualize
    rows, cols = navigable.shape
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create colored skeleton by component
    display = np.zeros((*skeleton.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]

    # Assign colors to components
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_features)))
    for i, c in enumerate(components[:10]):
        mask = labeled == c['id']
        display[mask] = colors[i, :3]

    ax.imshow(display, origin='lower', extent=extent)
    ax.plot(end_x, end_y, 'r*', markersize=20, label=f'Path end ({end_x:.1f}, {end_y:.1f})')
    if nearest is not None:
        ax.plot(nearest_x, nearest_y, 'g*', markersize=20, label=f'Nearest other ({nearest_x:.1f}, {nearest_y:.1f})')
        ax.plot([end_x, nearest_x], [end_y, nearest_y], 'k--', linewidth=2, label=f'Gap: {min_dist * resolution:.1f}m')

    ax.set_title(f"Skeleton Components (colored by component)\n{num_features} total components")
    ax.legend(loc='upper left')

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "skeleton_gaps_analysis.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_skeleton_gaps(bag_path)
