#!/usr/bin/env python3
"""
Visualize Centerline Extraction

This script extracts the centerline from the road grid and visualizes it
to validate the skeletonization and point ordering.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor
from circuit_navigator.map_processing.centerline import CenterlineExtractor


def extract_navigable_grid(bag_path: str, resolution: float = 0.5):
    """Extract and process road data to get navigable grid."""
    print("=" * 60)
    print("Extracting Navigable Grid")
    print("=" * 60)

    # Extract grid data from /road_whole PointCloud2
    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    resolution = data.resolution
    origin = data.origin

    # Create has_point mask (non-NaN and non-zero locations)
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Morphological closing to fill small gaps
    closing_radius_m = 2.0
    closing_radius_px = max(1, int(closing_radius_m / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    # Remove small disconnected components (noise)
    min_size_m2 = 50
    min_size_pixels = int(min_size_m2 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    # Apply safety margin (50cm erosion)
    safety_margin_m = 0.50
    safety_radius_px = max(1, int(safety_margin_m / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    print(f"Navigable cells: {np.sum(navigable)}")

    return {
        'navigable': navigable,
        'cleaned': cleaned,
        'resolution': resolution,
        'origin': origin
    }


def visualize_centerline(bag_path: str, resolution: float = 0.5):
    """Extract and visualize centerline from road data."""

    # Get navigable grid
    result = extract_navigable_grid(bag_path, resolution)
    navigable = result['navigable']
    cleaned = result['cleaned']
    grid_resolution = result['resolution']
    origin = result['origin']

    print("\n" + "=" * 60)
    print("Extracting Centerline")
    print("=" * 60)

    # Extract centerline
    extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = extractor.extract(navigable, grid_resolution, origin, method="skeleton")

    print(f"\nCenterline Results:")
    print(f"  Points: {len(centerline.points_world)}")
    print(f"  Length: {centerline.length_m:.1f} m")
    print(f"  Closed loop: {centerline.is_closed}")

    # Resample to uniform spacing
    print("\nResampling to 1m spacing...")
    resampled = extractor.resample_by_distance(centerline, spacing_m=1.0)
    print(f"  Resampled points: {len(resampled.points_world)}")

    # Calculate world extent for proper axis labels
    rows, cols = navigable.shape
    extent = [
        origin[0],
        origin[0] + cols * grid_resolution,
        origin[1],
        origin[1] + rows * grid_resolution
    ]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Centerline Extraction\n"
                 f"Length: {centerline.length_m:.1f}m, Points: {len(centerline.points_world)}, "
                 f"Closed: {centerline.is_closed}",
                 fontsize=14)

    # --- Panel 1: Navigable area with skeleton ---
    ax1 = axes[0, 0]

    # Show navigable area in gray
    display = np.zeros((*navigable.shape, 3))
    display[navigable] = [0.8, 0.8, 0.8]

    # Overlay skeleton in red
    if extractor.skeleton is not None:
        display[extractor.skeleton] = [1.0, 0.0, 0.0]

    ax1.imshow(display, origin='lower', extent=extent)
    ax1.set_title("Skeleton (red) on Navigable Area (gray)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # --- Panel 2: Ordered centerline points ---
    ax2 = axes[0, 1]

    # Show navigable area
    ax2.imshow(navigable.astype(float), origin='lower', extent=extent,
               cmap='gray', alpha=0.3)

    # Plot centerline with color gradient showing order
    points = centerline.points_world
    if len(points) > 0:
        colors = np.linspace(0, 1, len(points))
        scatter = ax2.scatter(points[:, 0], points[:, 1],
                             c=colors, cmap='viridis', s=2, alpha=0.8)
        plt.colorbar(scatter, ax=ax2, label='Point order (0=start)')

        # Mark start and end
        ax2.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start')
        ax2.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, label='End')
        ax2.legend()

    ax2.set_title(f"Ordered Centerline Points\n({len(points)} points)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect('equal')

    # --- Panel 3: Resampled centerline ---
    ax3 = axes[1, 0]

    ax3.imshow(navigable.astype(float), origin='lower', extent=extent,
               cmap='gray', alpha=0.3)

    # Plot resampled centerline
    resampled_pts = resampled.points_world
    if len(resampled_pts) > 0:
        ax3.plot(resampled_pts[:, 0], resampled_pts[:, 1],
                'b-', linewidth=1.5, alpha=0.8, label='Centerline')
        ax3.scatter(resampled_pts[::10, 0], resampled_pts[::10, 1],
                   c='blue', s=20, zorder=5)  # Mark every 10th point

        # Mark start
        ax3.plot(resampled_pts[0, 0], resampled_pts[0, 1], 'go',
                markersize=12, label='Start', zorder=10)

    ax3.set_title(f"Resampled Centerline (1m spacing)\n({len(resampled_pts)} points)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_aspect('equal')
    ax3.legend()

    # --- Panel 4: Full track view with both boundaries ---
    ax4 = axes[1, 1]

    # Show original cleaned area (before safety margin)
    combined = np.zeros((*cleaned.shape, 3))
    combined[cleaned] = [0.9, 0.9, 0.9]  # Light gray for full track
    combined[navigable] = [0.7, 0.9, 0.7]  # Light green for navigable

    ax4.imshow(combined, origin='lower', extent=extent)

    # Plot centerline
    if len(resampled_pts) > 0:
        ax4.plot(resampled_pts[:, 0], resampled_pts[:, 1],
                'b-', linewidth=2, label='Centerline')

        # Add direction arrows every 20 points
        for i in range(0, len(resampled_pts) - 1, 20):
            dx = resampled_pts[i + 1, 0] - resampled_pts[i, 0]
            dy = resampled_pts[i + 1, 1] - resampled_pts[i, 1]
            ax4.arrow(resampled_pts[i, 0], resampled_pts[i, 1],
                     dx * 3, dy * 3, head_width=1.5, head_length=0.8,
                     fc='blue', ec='blue', alpha=0.7)

    ax4.set_title("Track Overview\n(Gray: full track, Green: navigable with 50cm margin)")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_aspect('equal')
    ax4.legend()

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(bag_path), "centerline_extraction_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")

    plt.close()

    return {
        'centerline': centerline,
        'resampled': resampled,
        'extractor': extractor,
        'navigable': navigable,
        'resolution': grid_resolution,
        'origin': origin
    }


def analyze_centerline(result: dict):
    """Analyze the extracted centerline for track characteristics."""
    centerline = result['centerline']
    resampled = result['resampled']

    print("\n" + "=" * 60)
    print("Centerline Analysis")
    print("=" * 60)

    points = resampled.points_world

    if len(points) < 3:
        print("Not enough points for analysis")
        return

    # Calculate curvature along the path
    # Using finite differences: kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^1.5
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature
    speed_sq = dx**2 + dy**2
    curvature = np.abs(dx * ddy - dy * ddx) / (speed_sq**1.5 + 1e-10)

    print(f"Curvature statistics:")
    print(f"  Min: {curvature.min():.4f} 1/m")
    print(f"  Max: {curvature.max():.4f} 1/m (radius = {1/curvature.max():.1f}m)")
    print(f"  Mean: {curvature.mean():.4f} 1/m")

    # Find high curvature sections (potential roundabouts/tight curves)
    high_curvature_threshold = 0.05  # 1/20m radius
    high_curvature_mask = curvature > high_curvature_threshold
    high_curvature_segments = []

    in_segment = False
    segment_start = 0

    for i, is_high in enumerate(high_curvature_mask):
        if is_high and not in_segment:
            in_segment = True
            segment_start = i
        elif not is_high and in_segment:
            in_segment = False
            if i - segment_start > 5:  # At least 5m segment
                high_curvature_segments.append((segment_start, i))

    if in_segment:
        high_curvature_segments.append((segment_start, len(curvature)))

    print(f"\nHigh curvature segments (likely curves/roundabouts):")
    for start, end in high_curvature_segments:
        segment_length = end - start  # Since 1m spacing
        center_idx = (start + end) // 2
        center_pos = points[center_idx]
        avg_curv = curvature[start:end].mean()
        avg_radius = 1 / avg_curv if avg_curv > 0 else float('inf')
        print(f"  Index {start}-{end}: {segment_length}m long, "
              f"center at ({center_pos[0]:.1f}, {center_pos[1]:.1f}), "
              f"avg radius {avg_radius:.1f}m")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"

    result = visualize_centerline(bag_path)
    analyze_centerline(result)
