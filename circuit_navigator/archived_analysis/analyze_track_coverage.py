#!/usr/bin/env python3
"""Analyze track data coverage to understand gaps."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def analyze_track_coverage(bag_path: str, resolution: float = 0.5):
    """Analyze where we have track data."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin
    rows, cols = raw_grid.shape

    # Where do we have point data?
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Create extent for plotting
    extent = [origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution]

    # Analyze coverage by Y position
    print("Track data coverage by Y position:")
    print("-" * 40)

    # Check Y ranges of interest
    for y_start, y_end, name in [
        (0, 50, "Bottom (start area)"),
        (50, 100, "Lower middle (return path?)"),
        (100, 130, "Upper middle (second roundabout approach)"),
        (130, 160, "Top (second roundabout)")
    ]:
        row_start = int((y_start - origin[1]) / resolution)
        row_end = int((y_end - origin[1]) / resolution)
        row_start = max(0, row_start)
        row_end = min(rows, row_end)

        region = has_point[row_start:row_end, :]
        coverage = np.sum(region)
        total = region.size
        pct = 100 * coverage / total if total > 0 else 0

        print(f"  Y={y_start}-{y_end} ({name}): {coverage} points ({pct:.1f}%)")

    # Check X ranges at different Y levels
    print("\nCoverage by X position at different Y levels:")
    for y in [50, 60, 70, 80, 90, 100, 110, 120]:
        row = int((y - origin[1]) / resolution)
        if 0 <= row < rows:
            row_data = has_point[row, :]
            x_min = None
            x_max = None
            for c in range(cols):
                if row_data[c]:
                    x = origin[0] + c * resolution
                    if x_min is None:
                        x_min = x
                    x_max = x
            if x_min is not None:
                print(f"  Y={y}: X from {x_min:.1f} to {x_max:.1f}")
            else:
                print(f"  Y={y}: NO DATA")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Raw points
    axes[0].imshow(has_point.astype(float), origin='lower', extent=extent, cmap='Greens')
    axes[0].set_title(f"Raw point data coverage\n({np.sum(has_point)} cells with data)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    # Add horizontal lines at key Y levels
    for y in [50, 75, 100, 125]:
        axes[0].axhline(y=y, color='red', linestyle='--', alpha=0.5)
        axes[0].text(origin[0] + 2, y + 2, f'Y={y}', color='red')

    # After closing
    closing_radius_px = max(1, int(2.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    axes[1].imshow(closed.astype(float), origin='lower', extent=extent, cmap='Greens')
    axes[1].set_title(f"After morphological closing (2.0m radius)\n({np.sum(closed)} cells)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")

    for y in [50, 75, 100, 125]:
        axes[1].axhline(y=y, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "track_coverage_analysis.png")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    analyze_track_coverage(bag_path)
