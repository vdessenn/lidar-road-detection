#!/usr/bin/env python3
"""Visualize the track to understand where the gaps are."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, opening

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def visualize_gaps(bag_path: str, resolution: float = 0.5):
    """Visualize track to find gaps."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Find connected components in raw data
    labeled_raw, num_raw = ndimage.label(has_point)
    print(f"Raw point cloud: {num_raw} connected components")

    # Show sizes
    sizes = [(i, np.sum(labeled_raw == i)) for i in range(1, num_raw + 1)]
    sizes.sort(key=lambda x: -x[1])
    for comp_id, size in sizes[:10]:
        coords = np.argwhere(labeled_raw == comp_id)
        min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
        min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
        print(f"  Component {comp_id}: {size} cells, rows [{min_y},{max_y}], cols [{min_x},{max_x}]")

    # Visualize
    rows, cols = has_point.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Raw points
    axes[0].imshow(has_point.astype(float), origin='lower', extent=extent, cmap='Blues')
    axes[0].set_title(f"Raw point locations\n{np.sum(has_point)} cells in {num_raw} components")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    # Connected components colored
    display = np.zeros((*has_point.shape, 3))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (comp_id, size) in enumerate(sizes[:20]):
        mask = labeled_raw == comp_id
        display[mask] = colors[i % 20][:3]
    axes[1].imshow(display, origin='lower', extent=extent)
    axes[1].set_title(f"Top 20 connected components\n(colored by size)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")

    # After closing
    closing_radius_px = max(1, int(5.0 / resolution))  # Try larger closing
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)
    labeled_closed, num_closed = ndimage.label(closed)
    print(f"\nAfter closing (5m radius): {num_closed} connected components")

    axes[2].imshow(closed.astype(float), origin='lower', extent=extent, cmap='Greens')
    axes[2].set_title(f"After morphological closing (5m)\n{np.sum(closed)} cells in {num_closed} components")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "track_gaps_analysis.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    visualize_gaps(bag_path)
