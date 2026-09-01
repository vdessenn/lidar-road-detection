#!/usr/bin/env python3
"""
Real-Time Centerline Navigator - Main Simulation

Autonomous vehicle navigation using real-time centerline computation from
simulated lidar edge detection.

Key difference from circuit_navigator/:
- Centerline computed in real-time from lidar edges (not pre-computed)
- Configurable sensor range
- Quality metrics comparing real-time vs reference centerline
"""

import logging
import numpy as np
import time
import sys
import os
from typing import Optional, Tuple

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
from circuit_navigator.map_processing.centerline import CenterlineExtractor

from circuit_navigator_realtime.sensor.lidar_simulator import LidarEdgeSimulator, LidarScan
from circuit_navigator_realtime.filtering.edge_smoother import EdgeSmoother, SmootherConfig
from circuit_navigator_realtime.planning.local_centerline import LocalCenterlinePlanner, CenterlinePlannerConfig
from circuit_navigator_realtime.evaluation.centerline_compare import CenterlineEvaluator, EvaluationMetrics, format_metrics
from shared.navigation import PurePursuitController, PurePursuitConfig
from shared.simulation import BicycleModel, VehicleState
from circuit_navigator_realtime.visualization.visualizer import RealtimeVisualizer, VisualizerConfig


# =============================================================================
# Configuration
# =============================================================================

# Map processing
GRID_RESOLUTION = 0.5           # meters per cell
SAFETY_MARGIN_M = 0.50          # 50cm safety margin from track edges
CLOSING_RADIUS_M = 2.0          # Morphological closing radius

# Lidar sensor configuration (CONFIGURABLE)
SENSOR_RANGE_M = 20.0           # Lidar detection range (20m is optimal for this track)
NUM_LIDAR_RAYS = 180            # Angular resolution
LIDAR_FOV = np.pi               # 180 degrees forward-facing

# Edge smoothing
RANSAC_ENABLED = True
SAVGOL_ENABLED = True
KALMAN_ENABLED = True

# Centerline planner
MIN_FORWARD_DIST = 2.0          # Minimum forward distance
MAX_FORWARD_DIST = 40.0         # Maximum forward distance (matches sensor range)
CENTERLINE_SAFETY_MARGIN = 0.5  # Safety margin from edges

# Roundabout navigation (counterclockwise - European style)
# Optimized for Renault Zoe: center on normal track, right bias at roundabouts
ROUNDABOUT_RIGHT_BIAS = 0.45            # Normal driving: near center (0.5=center)
ROUNDABOUT_WIDENING_THRESHOLD = 1.1     # Width ratio to trigger roundabout mode
BASELINE_TRACK_WIDTH = 6.0              # Normal track width (m)
ADAPTIVE_BIAS_ENABLED = True            # Enable adaptive bias for roundabout entry
MIN_TRACK_WIDTH_FOR_BIAS = 2.5          # Minimum track width to check for roundabout

# Simulation
SIMULATION_HZ = 10              # Update frequency
DT = 1.0 / SIMULATION_HZ        # Time step

# Speed control
SPEED_DEFAULT_MS = 5.0          # Default cruise speed (m/s)
SPEED_CURVE_MS = 3.5            # Speed in curves (m/s)
CURVATURE_THRESHOLD = 0.05      # Curvature to trigger slow down (1/m)


# =============================================================================
# Utility Functions
# =============================================================================

def process_raw_grid(raw_grid: np.ndarray, resolution: float) -> np.ndarray:
    """
    Convert sparse road points to navigable binary grid.
    """
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    closing_radius_px = max(1, int(CLOSING_RADIUS_M / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    min_size_pixels = int(50 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)

    safety_radius_px = max(1, int(SAFETY_MARGIN_M / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)

    return navigable


def compute_target_speed(path_points: np.ndarray, default_speed: float = SPEED_DEFAULT_MS) -> float:
    """
    Compute target speed based on path curvature.
    """
    if len(path_points) < 5:
        return default_speed

    curvatures = []
    for i in range(2, len(path_points) - 2):
        p0 = path_points[i - 2]
        p1 = path_points[i]
        p2 = path_points[i + 2]

        v1 = p1 - p0
        v2 = p2 - p1

        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle_diff = angle2 - angle1

        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        arc_length = np.linalg.norm(v1) + np.linalg.norm(v2)
        if arc_length > 0:
            curvature = abs(angle_diff) / arc_length
            curvatures.append(curvature)

    if not curvatures:
        return default_speed

    max_curvature = max(curvatures)

    if max_curvature > CURVATURE_THRESHOLD:
        return SPEED_CURVE_MS
    else:
        return default_speed


def find_start_position(centerline: np.ndarray) -> Tuple[float, float, float]:
    """
    Find starting position and heading from centerline.
    """
    x, y = centerline[0]

    if len(centerline) > 5:
        dx = centerline[5, 0] - centerline[0, 0]
        dy = centerline[5, 1] - centerline[0, 1]
    else:
        dx = centerline[1, 0] - centerline[0, 0]
        dy = centerline[1, 1] - centerline[0, 1]

    yaw = np.arctan2(dy, dx)

    return x, y, yaw


def get_current_path_index(
    car_x: float,
    car_y: float,
    reference: np.ndarray,
    current_idx: int
) -> int:
    """
    Get current position index on reference path (for progress tracking).
    """
    n = len(reference)
    search_start = max(0, current_idx - 5)
    search_end = min(n, current_idx + 100)

    if search_start >= search_end:
        search_start = max(0, n - 10)
        search_end = n

    search_range = reference[search_start:search_end]
    dx = search_range[:, 0] - car_x
    dy = search_range[:, 1] - car_y
    distances = np.sqrt(dx*dx + dy*dy)

    local_idx = np.argmin(distances)
    new_idx = search_start + local_idx

    if new_idx >= current_idx - 3:
        return new_idx
    return current_idx


# =============================================================================
# Main Simulation
# =============================================================================

def run_simulation(
    bag_path: str,
    sensor_range: float = SENSOR_RANGE_M,
    max_laps: int = 2,
    headless: bool = False,
    show_metrics: bool = True
):
    """
    Run the real-time centerline navigation simulation.

    Args:
        bag_path: Path to ROS2 bag with road data
        sensor_range: Lidar sensor range in meters (CONFIGURABLE)
        max_laps: Number of laps to complete (0 = infinite)
        headless: If True, run without visualization
        show_metrics: If True, show quality metrics vs reference
    """
    print("=" * 70)
    print("Real-Time Centerline Navigator - Starting Simulation")
    print("=" * 70)
    print(f"  Sensor range: {sensor_range}m")
    print(f"  Lidar rays: {NUM_LIDAR_RAYS}")
    print(f"  Lidar FOV: {np.degrees(LIDAR_FOV):.0f} degrees")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Extract map data (for lidar simulation background)
    # -------------------------------------------------------------------------
    print("\n[1/5] Extracting map from MCAP bag...")
    extractor = GridMapExtractor(bag_path, resolution=GRID_RESOLUTION)
    map_data = extractor.extract()

    print(f"  Grid size: {map_data.dimensions}")
    print(f"  Resolution: {map_data.resolution} m/cell")
    print(f"  Origin: ({map_data.origin[0]:.1f}, {map_data.origin[1]:.1f})")

    # -------------------------------------------------------------------------
    # Step 2: Process grid to navigable area
    # -------------------------------------------------------------------------
    print("\n[2/5] Processing navigable area...")
    navigable_grid = process_raw_grid(map_data.raw_data, map_data.resolution)

    nav_cells = np.sum(navigable_grid)
    total_cells = navigable_grid.size
    print(f"  Navigable cells: {nav_cells} ({100*nav_cells/total_cells:.1f}%)")

    # -------------------------------------------------------------------------
    # Step 3: Extract reference centerline (for comparison only)
    # -------------------------------------------------------------------------
    print("\n[3/5] Extracting REFERENCE centerline (for comparison)...")
    cl_extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = cl_extractor.extract(
        navigable_grid,
        map_data.resolution,
        map_data.origin,
        method="skeleton"
    )

    print(f"  Reference centerline: {len(centerline.points_world)} points")
    print(f"  Reference length: {centerline.length_m:.1f} m")

    centerline = cl_extractor.resample_by_distance(centerline, spacing_m=0.5)
    reference_path = centerline.points_world
    print(f"  After resampling: {len(reference_path)} points")

    # -------------------------------------------------------------------------
    # Step 4: Initialize simulation components
    # -------------------------------------------------------------------------
    print("\n[4/5] Initializing real-time components...")

    # Lidar simulator with reference path for track-relative edge classification
    # This prevents left/right edge confusion when car heading deviates from track
    lidar = LidarEdgeSimulator(
        navigable_grid=navigable_grid,
        origin=map_data.origin,
        resolution=map_data.resolution,
        sensor_range=sensor_range,
        num_rays=NUM_LIDAR_RAYS,
        fov=LIDAR_FOV,
        reference_path=reference_path  # Use reference for robust edge classification
    )
    print(f"  Lidar simulator: {sensor_range}m range, {NUM_LIDAR_RAYS} rays")
    print(f"  Using track-relative edge classification (reference path)")

    # Edge smoother
    smoother_config = SmootherConfig(
        ransac_enabled=RANSAC_ENABLED,
        savgol_enabled=SAVGOL_ENABLED,
        kalman_enabled=KALMAN_ENABLED
    )
    edge_smoother = EdgeSmoother(smoother_config)
    print(f"  Edge smoother: RANSAC={RANSAC_ENABLED}, Savgol={SAVGOL_ENABLED}, Kalman={KALMAN_ENABLED}")

    # Local centerline planner (with roundabout navigation support)
    planner_config = CenterlinePlannerConfig(
        min_forward_dist=MIN_FORWARD_DIST,
        max_forward_dist=min(MAX_FORWARD_DIST, sensor_range),
        safety_margin=CENTERLINE_SAFETY_MARGIN,
        # Roundabout navigation parameters
        right_bias_alpha=ROUNDABOUT_RIGHT_BIAS,
        track_widening_threshold=ROUNDABOUT_WIDENING_THRESHOLD,
        baseline_track_width=BASELINE_TRACK_WIDTH,
        adaptive_bias_enabled=ADAPTIVE_BIAS_ENABLED,
        min_track_width_for_bias=MIN_TRACK_WIDTH_FOR_BIAS
    )
    centerline_planner = LocalCenterlinePlanner(planner_config)
    print(f"  Centerline planner: forward range [{MIN_FORWARD_DIST}, {min(MAX_FORWARD_DIST, sensor_range)}]m")
    print(f"  Roundabout mode: right_bias={ROUNDABOUT_RIGHT_BIAS}, widening_threshold={ROUNDABOUT_WIDENING_THRESHOLD}")

    # Centerline evaluator (compares against reference)
    evaluator = CenterlineEvaluator(reference_path, match_radius=5.0)
    print(f"  Evaluator: comparing against reference centerline")

    # Pure Pursuit controller - optimized for Renault Zoe roundabout navigation
    pp_config = PurePursuitConfig(
        wheelbase_m=2.588,  # Renault Zoe wheelbase
        min_lookahead_m=2.0,    # Shorter lookahead for faster reaction
        max_lookahead_m=5.0,    # Reduced max for tighter control
        lookahead_speed_gain=0.4,
        max_steer_rad=0.6       # 34 degrees for tight roundabout turns
    )
    pure_pursuit = PurePursuitController(pp_config)

    # Bicycle model - Renault Zoe
    bicycle_model = BicycleModel(
        wheelbase=2.588,  # Renault Zoe wheelbase
        max_steer=0.6,    # 34 degrees for tight roundabout turns
        max_speed=10.0,
        max_accel=2.0,
        max_decel=4.0
    )

    # Find start position from reference (just for initial placement)
    start_x, start_y, start_yaw = find_start_position(reference_path)
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

    # -------------------------------------------------------------------------
    # Step 5: Initialize visualization
    # -------------------------------------------------------------------------
    visualizer = None
    if not headless:
        print("\n[5/5] Initializing visualization...")
        viz_config = VisualizerConfig(
            vehicle_length=4.087,  # Renault Zoe length
            vehicle_width=1.73,    # Renault Zoe width (without mirrors)
            trail_length=200,
            sensor_range=sensor_range
        )
        visualizer = RealtimeVisualizer(
            navigable_grid=navigable_grid,
            reference_centerline=reference_path,
            origin=map_data.origin,
            resolution=map_data.resolution,
            config=viz_config
        )
    else:
        print("\n[5/5] Skipping visualization (headless mode)")

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Starting REAL-TIME simulation loop (Ctrl+C to stop)")
    print("=" * 70 + "\n")

    lap_count = 0
    current_ref_idx = 0
    step = 0
    path_complete = False
    distance_traveled = 0.0
    prev_x, prev_y = start_x, start_y

    # Metrics accumulation
    all_mean_errors = []
    all_max_errors = []
    all_rms_errors = []
    all_coverages = []

    try:
        while True:
            # =================================================================
            # 1. SENSE: Lidar detects track edges
            # =================================================================
            scan = lidar.scan(vehicle_state.x, vehicle_state.y, vehicle_state.yaw)

            # =================================================================
            # 2. FILTER: Smooth noisy edge points
            # =================================================================
            left_smooth, right_smooth = edge_smoother.smooth_edges(
                scan.left_edge,
                scan.right_edge,
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw
            )

            # Create filtered scan (preserving island detection info)
            filtered_scan = LidarScan(
                left_edge=left_smooth,
                right_edge=right_smooth,
                all_points=np.vstack([left_smooth, right_smooth]) if len(left_smooth) > 0 and len(right_smooth) > 0 else scan.all_points,
                ranges=scan.ranges,
                angles=scan.angles,
                car_position=(vehicle_state.x, vehicle_state.y),
                car_yaw=vehicle_state.yaw,
                # Pass through island detection from original scan
                island_points=scan.island_points,
                has_island_ahead=scan.has_island_ahead,
                island_center_angle=scan.island_center_angle
            )

            # =================================================================
            # 3. PLAN: Compute centerline from edges (with island avoidance)
            # =================================================================
            computed_path = centerline_planner.compute(
                left_smooth,
                right_smooth,
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw,
                island_points=scan.island_points,
                has_island_ahead=scan.has_island_ahead
            )

            # =================================================================
            # 4. EVALUATE: Compare against reference (development only)
            # =================================================================
            metrics = evaluator.evaluate(
                computed_path,
                vehicle_state.x,
                vehicle_state.y
            )

            if metrics.mean_lateral_error != float('inf'):
                all_mean_errors.append(metrics.mean_lateral_error)
                all_max_errors.append(metrics.max_lateral_error)
                all_rms_errors.append(metrics.rms_error)
                all_coverages.append(metrics.coverage)

            # =================================================================
            # 5. CONTROL: Pure Pursuit on computed path
            # =================================================================
            if len(computed_path) >= 2:
                steering, target_point, _ = pure_pursuit.compute_steering(
                    vehicle_state.x,
                    vehicle_state.y,
                    vehicle_state.yaw,
                    vehicle_state.speed,
                    computed_path
                )
                target_speed = compute_target_speed(computed_path)
            else:
                # Fallback: slow down if no path
                steering = 0.0
                target_point = (vehicle_state.x, vehicle_state.y)
                target_speed = 2.0

            # =================================================================
            # 6. ACT: Update vehicle dynamics
            # =================================================================
            vehicle_state = bicycle_model.update(
                vehicle_state,
                steering,
                target_speed,
                DT
            )

            # Track distance traveled
            dx = vehicle_state.x - prev_x
            dy = vehicle_state.y - prev_y
            distance_traveled += np.sqrt(dx*dx + dy*dy)
            prev_x, prev_y = vehicle_state.x, vehicle_state.y

            # =================================================================
            # 7. VISUALIZE: Show edges, computed path, reference path
            # =================================================================
            if visualizer is not None:
                visualizer.update(
                    car_x=vehicle_state.x,
                    car_y=vehicle_state.y,
                    car_yaw=vehicle_state.yaw,
                    car_speed=vehicle_state.speed,
                    steering=vehicle_state.steering,
                    lidar_scan=filtered_scan,
                    computed_path=computed_path,
                    target_point=target_point,
                    metrics=metrics,
                    distance_traveled=distance_traveled
                )

            # =================================================================
            # Progress tracking (using reference path)
            # =================================================================
            current_ref_idx = get_current_path_index(
                vehicle_state.x, vehicle_state.y, reference_path, current_ref_idx
            )
            lap_progress = current_ref_idx / len(reference_path)

            # Detect path completion
            if current_ref_idx >= len(reference_path) - 10 and not path_complete:
                lap_count += 1
                print(f"\n*** Path traversal {lap_count} complete! ***")

                # Print lap statistics
                if len(all_mean_errors) > 0:
                    print(f"  Mean lateral error: {np.mean(all_mean_errors):.3f} m")
                    print(f"  Max lateral error: {np.max(all_max_errors):.3f} m")
                    print(f"  RMS error: {np.mean(all_rms_errors):.3f} m")
                    print(f"  Coverage: {np.mean(all_coverages)*100:.1f}%")

                path_complete = True

                if max_laps > 0 and lap_count >= max_laps:
                    print("\nTarget traversals reached. Stopping.")
                    break

                # Reset for another lap
                current_ref_idx = 0
                path_complete = False
                edge_smoother.reset()  # Reset Kalman filters
                lidar.reset_path_progress()  # Reset path tracking for new lap
                vehicle_state = VehicleState(
                    x=reference_path[0, 0],
                    y=reference_path[0, 1],
                    yaw=find_start_position(reference_path)[2],
                    speed=0.0,
                    steering=0.0
                )
                prev_x, prev_y = vehicle_state.x, vehicle_state.y

            # Console output every second
            if step % SIMULATION_HZ == 0 and show_metrics:
                print(
                    f"Pos: ({vehicle_state.x:6.1f}, {vehicle_state.y:6.1f}) | "
                    f"Speed: {vehicle_state.speed:4.1f} m/s | "
                    f"Steer: {np.degrees(vehicle_state.steering):5.1f}° | "
                    f"Lat Err: {metrics.mean_lateral_error:5.3f}m | "
                    f"Progress: {lap_progress*100:5.1f}%"
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

    # Print final statistics
    print("\n" + "=" * 70)
    print("Simulation Complete - Final Statistics")
    print("=" * 70)
    print(f"  Total steps: {step}")
    print(f"  Laps completed: {lap_count}")
    print(f"  Distance traveled: {distance_traveled:.1f} m")

    if len(all_mean_errors) > 0:
        print(f"\n  QUALITY METRICS vs REFERENCE:")
        print(f"    Mean lateral error: {np.mean(all_mean_errors):.3f} m")
        print(f"    Max lateral error: {np.max(all_max_errors):.3f} m")
        print(f"    RMS error: {np.mean(all_rms_errors):.3f} m")
        print(f"    Coverage: {np.mean(all_coverages)*100:.1f}%")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-Time Centerline Navigator Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --bag /path/to/bag --range 40.0
  python main.py --range 30.0 --laps 1
  python main.py --headless --no-metrics
        """
    )
    parser.add_argument(
        "--bag",
        type=str,
        default="sy27_road_ground_truth2/",
        help="Path to ROS2 bag with road data"
    )
    parser.add_argument(
        "--range",
        type=float,
        default=40.0,
        help="Lidar sensor range in meters (default: 40.0)"
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
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable console metrics output"
    )

    args = parser.parse_args()

    run_simulation(
        bag_path=args.bag,
        sensor_range=args.range,
        max_laps=args.laps,
        headless=args.headless,
        show_metrics=not args.no_metrics
    )
