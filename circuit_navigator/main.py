#!/usr/bin/env python3
"""
Circuit Navigator - Main Simulation

Autonomous vehicle navigation around a circuit using:
- GridMap extraction from ROS2 MCAP bag
- Skeletonization-based centerline extraction
- Pure Pursuit path following
- Kinematic bicycle model simulation
- Real-time matplotlib visualization
"""

import logging
import numpy as np
import time
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor
from circuit_navigator.map_processing.centerline import CenterlineExtractor, Centerline
from shared.navigation import PurePursuitController, PurePursuitConfig
from shared.simulation import BicycleModel, VehicleState
from circuit_navigator.visualization.visualizer import CircuitVisualizer, VisualizerConfig


# =============================================================================
# Configuration
# =============================================================================

# Map processing
GRID_RESOLUTION = 0.5           # meters per cell
SAFETY_MARGIN_M = 0.50          # 50cm safety margin from track edges
CLOSING_RADIUS_M = 2.0          # Morphological closing radius

# Path following
VISIBLE_HORIZON_M = 35.0        # Lookahead visibility distance

# Simulation
SIMULATION_HZ = 10              # Update frequency
DT = 1.0 / SIMULATION_HZ        # Time step

# Speed control (simple profile)
SPEED_DEFAULT_MS = 5.0          # Default cruise speed (m/s)
SPEED_CURVE_MS = 3.5            # Speed in curves (m/s)
CURVATURE_THRESHOLD = 0.05     # Curvature to trigger slow down (1/m)


# =============================================================================
# Utility Functions
# =============================================================================

def process_raw_grid(raw_grid: np.ndarray, resolution: float) -> np.ndarray:
    """
    Convert sparse road points to navigable binary grid.

    Steps:
    1. Mark cells with valid data
    2. Morphological closing to fill gaps
    3. Remove small isolated regions
    4. Apply safety margin via erosion
    """
    # Mark cells with data (not NaN and not zero)
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # Morphological closing to fill gaps in sparse data
    closing_radius_px = max(1, int(CLOSING_RADIUS_M / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    # Remove small isolated regions
    min_size_pixels = int(50 / (resolution * resolution))  # ~50 sq meters
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    # Apply safety margin (erode by 50cm)
    safety_radius_px = max(1, int(SAFETY_MARGIN_M / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    return navigable


def get_visible_path(
    car_x: float,
    car_y: float,
    centerline: np.ndarray,
    current_idx: int,
    horizon_m: float = VISIBLE_HORIZON_M
) -> tuple:
    """
    Get visible portion of path ahead of vehicle.

    Args:
        car_x, car_y: Vehicle position
        centerline: Full path
        current_idx: Current progress index (advances monotonically)
        horizon_m: How far ahead to look

    Returns:
        (visible_path, new_current_idx)
    """
    n = len(centerline)

    # Search for nearest point only in a forward window from current_idx
    # This prevents jumping back on out-and-back paths
    search_start = max(0, current_idx - 5)  # Allow small backward correction
    search_end = min(n, current_idx + 100)  # Look ahead up to 50m

    if search_start >= search_end:
        search_start = max(0, n - 10)
        search_end = n

    search_range = centerline[search_start:search_end]
    dx = search_range[:, 0] - car_x
    dy = search_range[:, 1] - car_y
    distances = np.sqrt(dx*dx + dy*dy)

    local_idx = np.argmin(distances)
    new_idx = search_start + local_idx

    # Only advance index, never go backward significantly
    if new_idx >= current_idx - 3:
        current_idx = new_idx

    # Get path points within horizon
    visible_points = []
    accumulated_dist = 0.0
    prev_point = centerline[current_idx]

    for i in range(current_idx, n):
        point = centerline[i]
        segment_dist = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
        accumulated_dist += segment_dist

        if accumulated_dist > horizon_m:
            break

        visible_points.append(point)
        prev_point = point

    if len(visible_points) == 0:
        visible_points = [centerline[min(current_idx, n-1)]]

    return np.array(visible_points), current_idx


def compute_target_speed(
    path_points: np.ndarray,
    default_speed: float = SPEED_DEFAULT_MS
) -> float:
    """
    Compute target speed based on path curvature.

    Lower speed when high curvature ahead.
    """
    if len(path_points) < 5:
        return default_speed

    # Compute curvature at multiple points
    curvatures = []
    for i in range(2, len(path_points) - 2):
        # Use 5-point curvature estimation
        p0 = path_points[i - 2]
        p1 = path_points[i]
        p2 = path_points[i + 2]

        # Vectors
        v1 = p1 - p0
        v2 = p2 - p1

        # Angle change
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle_diff = angle2 - angle1

        # Normalize
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Arc length
        arc_length = np.linalg.norm(v1) + np.linalg.norm(v2)
        if arc_length > 0:
            curvature = abs(angle_diff) / arc_length
            curvatures.append(curvature)

    if not curvatures:
        return default_speed

    # Use max curvature in visible path
    max_curvature = max(curvatures)

    if max_curvature > CURVATURE_THRESHOLD:
        # Reduce speed for curves
        return SPEED_CURVE_MS
    else:
        return default_speed


def find_start_position(centerline: np.ndarray) -> tuple:
    """
    Find starting position and heading from centerline.

    Returns:
        (x, y, yaw)
    """
    # Start at first point
    x, y = centerline[0]

    # Heading from first few points
    if len(centerline) > 5:
        dx = centerline[5, 0] - centerline[0, 0]
        dy = centerline[5, 1] - centerline[0, 1]
    else:
        dx = centerline[1, 0] - centerline[0, 0]
        dy = centerline[1, 1] - centerline[0, 1]

    yaw = np.arctan2(dy, dx)

    return x, y, yaw


# =============================================================================
# Main Simulation
# =============================================================================

def run_simulation(
    bag_path: str,
    max_laps: int = 2,
    headless: bool = False
):
    """
    Run the circuit navigation simulation.

    Args:
        bag_path: Path to ROS2 bag with road data
        max_laps: Number of laps to complete (0 = infinite)
        headless: If True, run without visualization
    """
    print("=" * 60)
    print("Circuit Navigator - Starting Simulation")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Extract map data
    # -------------------------------------------------------------------------
    print("\n[1/4] Extracting map from MCAP bag...")
    extractor = GridMapExtractor(bag_path, resolution=GRID_RESOLUTION)
    map_data = extractor.extract()

    print(f"  Grid size: {map_data.dimensions}")
    print(f"  Resolution: {map_data.resolution} m/cell")
    print(f"  Origin: ({map_data.origin[0]:.1f}, {map_data.origin[1]:.1f})")

    # -------------------------------------------------------------------------
    # Step 2: Process grid to navigable area
    # -------------------------------------------------------------------------
    print("\n[2/4] Processing navigable area...")
    navigable_grid = process_raw_grid(map_data.raw_data, map_data.resolution)

    nav_cells = np.sum(navigable_grid)
    total_cells = navigable_grid.size
    print(f"  Navigable cells: {nav_cells} ({100*nav_cells/total_cells:.1f}%)")

    # -------------------------------------------------------------------------
    # Step 3: Extract centerline
    # -------------------------------------------------------------------------
    print("\n[3/4] Extracting centerline...")
    cl_extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = cl_extractor.extract(
        navigable_grid,
        map_data.resolution,
        map_data.origin,
        method="skeleton"
    )

    print(f"  Centerline points: {len(centerline.points_world)}")
    print(f"  Centerline length: {centerline.length_m:.1f} m")
    print(f"  Closed loop: {centerline.is_closed}")

    # Resample for smoother following
    centerline = cl_extractor.resample_by_distance(centerline, spacing_m=0.5)
    print(f"  After resampling: {len(centerline.points_world)} points")

    # -------------------------------------------------------------------------
    # Step 4: Initialize simulation components
    # -------------------------------------------------------------------------
    print("\n[4/4] Initializing simulation...")

    # Find start position from centerline
    start_x, start_y, start_yaw = find_start_position(centerline.points_world)
    print(f"  Start position: ({start_x:.1f}, {start_y:.1f})")
    print(f"  Start heading: {np.degrees(start_yaw):.1f} deg")

    # Initialize vehicle state
    vehicle_state = VehicleState(
        x=start_x,
        y=start_y,
        yaw=start_yaw,
        speed=0.0,
        steering=0.0
    )

    # Initialize controllers
    pp_config = PurePursuitConfig(
        wheelbase_m=2.5,
        min_lookahead_m=4.0,
        max_lookahead_m=12.0,
        lookahead_speed_gain=0.5,
        max_steer_rad=0.40
    )
    pure_pursuit = PurePursuitController(pp_config)

    bicycle_model = BicycleModel(
        wheelbase=2.5,
        max_steer=0.40,
        max_speed=10.0,
        max_accel=2.0,
        max_decel=4.0
    )

    # Initialize visualization
    visualizer = None
    if not headless:
        print("\n  Initializing visualization...")
        viz_config = VisualizerConfig(
            vehicle_length=4.0,
            vehicle_width=2.0,
            trail_length=200
        )
        visualizer = CircuitVisualizer(
            navigable_grid=navigable_grid,
            centerline_world=centerline.points_world,
            origin=map_data.origin,
            resolution=map_data.resolution,
            config=viz_config
        )

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting simulation loop (Ctrl+C to stop)")
    print("=" * 60 + "\n")

    path_points = centerline.points_world
    lap_count = 0
    current_idx = 0
    step = 0
    path_complete = False

    try:
        while True:
            # Get visible path (current_idx advances monotonically)
            visible_path, current_idx = get_visible_path(
                vehicle_state.x, vehicle_state.y, path_points, current_idx
            )

            # Detect path completion (reached end of path)
            if current_idx >= len(path_points) - 10 and not path_complete:
                lap_count += 1
                print(f"\n*** Path traversal {lap_count} complete! ***\n")
                path_complete = True
                if max_laps > 0 and lap_count >= max_laps:
                    print("Target traversals reached. Stopping.")
                    break
                # Reset to start for another lap
                current_idx = 0
                path_complete = False
                vehicle_state = VehicleState(
                    x=path_points[0, 0],
                    y=path_points[0, 1],
                    yaw=find_start_position(path_points)[2],
                    speed=0.0,
                    steering=0.0
                )

            # Compute lap progress
            lap_progress = current_idx / len(path_points)

            # Compute target speed based on curvature
            target_speed = compute_target_speed(visible_path)

            # Pure Pursuit steering
            steering, target_point, _ = pure_pursuit.compute_steering(
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw,
                vehicle_state.speed,
                visible_path
            )

            # Update vehicle dynamics
            vehicle_state = bicycle_model.update(
                vehicle_state,
                steering,
                target_speed,
                DT
            )

            # Update visualization
            if visualizer is not None:
                visualizer.update(
                    car_x=vehicle_state.x,
                    car_y=vehicle_state.y,
                    car_yaw=vehicle_state.yaw,
                    car_speed=vehicle_state.speed,
                    steering=vehicle_state.steering,
                    target_point=target_point,
                    visible_path=visible_path,
                    path_idx=current_idx,
                    lap_progress=lap_progress
                )

            # Console output every second
            if step % SIMULATION_HZ == 0:
                print(
                    f"Pos: ({vehicle_state.x:6.1f}, {vehicle_state.y:6.1f}) | "
                    f"Speed: {vehicle_state.speed:4.1f} m/s | "
                    f"Steer: {np.degrees(vehicle_state.steering):5.1f}° | "
                    f"Progress: {lap_progress*100:5.1f}% | "
                    f"Lap: {lap_count}"
                )

            # Wait for next timestep
            time.sleep(DT)
            step += 1

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    finally:
        if visualizer is not None:
            print("Closing visualization...")
            visualizer.close()

    print("\nSimulation complete!")
    print(f"  Total steps: {step}")
    print(f"  Laps completed: {lap_count}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Circuit Navigator Simulation")
    parser.add_argument(
        "--bag",
        type=str,
        default="sy27_road_ground_truth2/",
        help="Path to ROS2 bag with road data"
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=2,
        help="Number of laps to complete (0 = infinite)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization"
    )

    args = parser.parse_args()

    run_simulation(
        bag_path=args.bag,
        max_laps=args.laps,
        headless=args.headless
    )
