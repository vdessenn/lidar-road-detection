#!/usr/bin/env python3
"""
Extract centerline using boundary-based approach:
1. Find the boundary contours of the navigable area
2. For each point on the boundary, find the nearest point on the opposite boundary
3. The centerline is the midpoint between boundary pairs
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects
from skimage import measure

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def extract_boundary_centerline(bag_path: str, resolution: float = 0.5):
    """Extract centerline using distance transform ridge."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Processing
    closing_radius_px = max(1, int(3.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    min_size_pixels = int(100 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    # Keep only largest connected component
    labeled, num = ndimage.label(cleaned)
    sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
    sizes.sort(key=lambda x: -x[1])
    largest_id = sizes[0][0]
    track = labeled == largest_id

    print(f"Track: {np.sum(track)} cells (one connected component)")

    # Compute distance transform
    dist_transform = ndimage.distance_transform_edt(track)
    max_dist = dist_transform.max()
    print(f"Max distance from boundary: {max_dist * resolution:.2f}m")

    # Find ridge points (local maxima of distance transform)
    # Use a larger neighborhood for smoothness
    from scipy.ndimage import maximum_filter, minimum_filter

    neighborhood_size = 5
    local_max = maximum_filter(dist_transform, size=neighborhood_size)

    # Ridge points: where distance equals local maximum and distance > threshold
    min_track_width = 1.0  # meters
    min_dist_threshold = min_track_width / (2 * resolution)

    ridge_mask = (dist_transform == local_max) & (dist_transform >= min_dist_threshold)
    ridge_coords = np.argwhere(ridge_mask)

    print(f"Ridge points: {len(ridge_coords)}")

    # Check connectivity of ridge
    labeled_ridge, num_ridge = ndimage.label(ridge_mask)
    print(f"Ridge components: {num_ridge}")

    if num_ridge > 0:
        sizes_ridge = [(i, np.sum(labeled_ridge == i)) for i in range(1, num_ridge + 1)]
        sizes_ridge.sort(key=lambda x: -x[1])
        print(f"Largest ridge components:")
        for comp_id, size in sizes_ridge[:5]:
            print(f"  Component {comp_id}: {size} pixels")

    # Now try to connect ridge fragments into a path
    # Use a greedy nearest-neighbor traversal with gap jumping
    if len(ridge_coords) > 0:
        ridge_set = {tuple(c) for c in ridge_coords}

        # Start from the point with minimum row (bottom of track)
        start_idx = np.argmin(ridge_coords[:, 0])
        start = tuple(ridge_coords[start_idx])

        ordered = [start]
        remaining = ridge_set - {start}
        current = start

        while remaining:
            # Find nearest point in remaining
            min_dist = float('inf')
            nearest = None
            for p in remaining:
                d = np.sqrt((p[0] - current[0])**2 + (p[1] - current[1])**2)
                if d < min_dist:
                    min_dist = d
                    nearest = p

            if nearest is None or min_dist > 20:  # Max gap of 20 pixels = 10m at 0.5m
                # Too far, try to find another nearby cluster
                print(f"  Gap at {len(ordered)} points, min_dist={min_dist*resolution:.1f}m")
                if min_dist > 50:
                    break
            if nearest:
                ordered.append(nearest)
                remaining.discard(nearest)
                current = nearest

        print(f"\nOrdered ridge path: {len(ordered)} points")
        ordered_arr = np.array(ordered)

        # Convert to world coordinates
        world_x = origin[0] + ordered_arr[:, 1] * resolution
        world_y = origin[1] + ordered_arr[:, 0] * resolution

        # Calculate path length
        if len(ordered) > 1:
            diffs = np.diff(np.column_stack([world_x, world_y]), axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            print(f"Path length: {path_length:.1f}m")

    # Visualize
    rows, cols = track.shape
    extent = [
        origin[0],
        origin[0] + cols * resolution,
        origin[1],
        origin[1] + rows * resolution
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Track
    axes[0, 0].imshow(track.astype(float), origin='lower', extent=extent, cmap='Greys')
    axes[0, 0].set_title("Track (largest component)")

    # Distance transform
    im1 = axes[0, 1].imshow(dist_transform, origin='lower', extent=extent, cmap='hot')
    axes[0, 1].set_title("Distance transform (track width)")
    plt.colorbar(im1, ax=axes[0, 1])

    # Ridge mask
    display = np.zeros((*track.shape, 3))
    display[track] = [0.8, 0.8, 0.8]
    display[ridge_mask] = [0, 0, 1]
    axes[1, 0].imshow(display, origin='lower', extent=extent)
    axes[1, 0].set_title(f"Ridge points: {len(ridge_coords)} in {num_ridge} components")

    # Ordered path
    display2 = np.zeros((*track.shape, 3))
    display2[track] = [0.8, 0.8, 0.8]
    if len(ordered) > 1:
        axes[1, 1].imshow(display2, origin='lower', extent=extent)
        axes[1, 1].plot(world_x, world_y, 'b-', linewidth=1, alpha=0.7)
        axes[1, 1].plot(world_x[0], world_y[0], 'go', markersize=10, label='Start')
        axes[1, 1].plot(world_x[-1], world_y[-1], 'ro', markersize=10, label='End')
        axes[1, 1].legend()
        axes[1, 1].set_title(f"Ordered path: {len(ordered)} points, {path_length:.1f}m")
    else:
        axes[1, 1].text(0.5, 0.5, "No path extracted", ha='center', va='center')

    plt.tight_layout()
    output = os.path.join(os.path.dirname(bag_path), "boundary_centerline_test.png")
    plt.savefig(output, dpi=150)
    print(f"\nSaved to {output}")
    plt.close()


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    extract_boundary_centerline(bag_path)
