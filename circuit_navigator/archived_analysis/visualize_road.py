#!/usr/bin/env python3
"""
Visualize Road Extraction

This script extracts the road layer from the GridMap MCAP bag and displays it
to validate the data before building the full navigation system.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects, label

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def visualize_road_extraction(bag_path: str, resolution: float = 0.5):
    """
    Extract and visualize road data from PointCloud2.

    Shows:
    1. Raw grid data (with NaN areas)
    2. Point locations (has_point mask)
    3. After morphological closing
    4. Final filled road area
    """
    print("=" * 60)
    print("Road Extraction Visualization")
    print("=" * 60)

    # Extract grid data from /road_whole PointCloud2
    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    resolution = data.resolution
    origin = data.origin

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Road Extraction from GridMap\n"
                 f"Resolution: {resolution}m/cell, Origin: ({origin[0]:.1f}, {origin[1]:.1f})",
                 fontsize=14)

    # Calculate world extent for proper axis labels
    rows, cols = raw_grid.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    # --- Panel 1: Raw grid data ---
    ax1 = axes[0, 0]
    # Replace NaN with a sentinel value for visualization
    display_grid = raw_grid.copy()
    nan_mask = np.isnan(display_grid)
    display_grid[nan_mask] = np.nanmin(raw_grid) - 1  # Below min for NaN areas

    im1 = ax1.imshow(display_grid, origin='lower', extent=extent, cmap='viridis')
    ax1.set_title(f"Raw Grid Data (road layer)\nNaN count: {np.sum(nan_mask)}")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax1, label='Value')

    # --- Panel 2: Has-point mask (non-NaN locations) ---
    ax2 = axes[0, 1]
    has_point = ~np.isnan(raw_grid)
    # Also check for zero values if needed
    has_point = has_point & (raw_grid != 0)

    ax2.imshow(has_point.astype(int), origin='lower', extent=extent, cmap='gray')
    ax2.set_title(f"Points Detected\n({np.sum(has_point)} points = {100*np.mean(has_point):.1f}%)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # --- Panel 3: Point cloud scatter ---
    ax3 = axes[0, 2]
    if data.point_cloud is not None:
        point_coords = data.point_cloud[:, :2]
        point_values = data.point_cloud[:, 2] if data.point_cloud.shape[1] > 2 else np.ones(len(data.point_cloud))

        scatter = ax3.scatter(point_coords[:, 0], point_coords[:, 1],
                             c=point_values, s=0.5, cmap='viridis', alpha=0.7)
        ax3.set_title(f"Point Cloud View\n({len(point_coords)} points)")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_aspect('equal')
        plt.colorbar(scatter, ax=ax3, label='Z value')
    else:
        ax3.text(0.5, 0.5, "No point cloud data", ha='center', va='center')
        ax3.set_title("Point Cloud View")

    # --- Panel 4: Morphological closing ---
    ax4 = axes[1, 0]

    # Calculate kernel size based on typical gap size
    # Try different closing radii
    closing_radius_m = 2.0  # 2 meter radius to fill gaps
    closing_radius_px = max(1, int(closing_radius_m / resolution))

    print(f"\nMorphological Processing:")
    print(f"  Closing radius: {closing_radius_m}m = {closing_radius_px} pixels")

    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    ax4.imshow(closed.astype(int), origin='lower', extent=extent, cmap='gray')
    ax4.set_title(f"After Closing (radius={closing_radius_m}m)\n{np.sum(closed)} cells")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")

    # --- Panel 5: Remove small objects (noise) - DO NOT fill holes (grass zones) ---
    ax5 = axes[1, 1]

    # Remove small disconnected components (noise at bottom)
    # Keep only the largest connected component(s)
    min_size_m2 = 50  # Minimum area in m^2 to keep
    min_size_pixels = int(min_size_m2 / (resolution * resolution))

    # Remove small objects
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    ax5.imshow(cleaned.astype(int), origin='lower', extent=extent, cmap='gray')
    ax5.set_title(f"After Noise Removal\n(min area={min_size_m2}m²)\n{np.sum(cleaned)} cells")
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")

    # --- Panel 6: Final navigable area with 50cm safety margin ---
    ax6 = axes[1, 2]

    safety_margin_m = 0.50
    safety_radius_px = max(1, int(safety_margin_m / resolution))

    print(f"  Safety margin: {safety_margin_m}m = {safety_radius_px} pixels")

    safety_kernel = disk(safety_radius_px)
    eroded = ndimage.binary_erosion(cleaned, safety_kernel)

    # Show eroded area in green, original in light gray
    combined = np.zeros((*cleaned.shape, 3))
    combined[cleaned] = [0.8, 0.8, 0.8]  # Light gray for original
    combined[eroded] = [0.2, 0.8, 0.2]  # Green for safe area

    ax6.imshow(combined, origin='lower', extent=extent)
    ax6.set_title(f"Navigable Area (green)\nwith {safety_margin_m}m safety margin")
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(bag_path), "road_extraction_result.png"),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to road_extraction_result.png")

    # Don't call plt.show() - save to file only
    plt.close()

    # Return processed grids for further use
    return {
        'raw_grid': raw_grid,
        'has_point': has_point,
        'closed': closed,
        'cleaned': cleaned,  # No holes filled, noise removed
        'navigable': eroded,
        'resolution': resolution,
        'origin': origin,
        'extent': extent
    }


def analyze_road_shape(result: dict):
    """Analyze the extracted road shape."""
    cleaned = result['cleaned']
    navigable = result['navigable']
    resolution = result['resolution']

    print("\n" + "=" * 60)
    print("Road Shape Analysis")
    print("=" * 60)

    # Calculate areas
    cleaned_area_m2 = np.sum(cleaned) * resolution * resolution
    navigable_area_m2 = np.sum(navigable) * resolution * resolution

    print(f"Total road area: {cleaned_area_m2:.1f} m^2")
    print(f"Navigable area (with safety): {navigable_area_m2:.1f} m^2")
    print(f"Area reduction: {100*(1 - navigable_area_m2/cleaned_area_m2):.1f}%")

    # Estimate track length (very rough)
    # Using skeleton would be more accurate
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(navigable)
        track_length_m = np.sum(skeleton) * resolution
        print(f"Estimated track length: {track_length_m:.1f} m")
    except ImportError:
        print("Install scikit-image for track length estimation")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"

    result = visualize_road_extraction(bag_path)
    analyze_road_shape(result)
