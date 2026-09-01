#!/usr/bin/env python3
"""
Diagnostic Analysis - Collect centerline data and identify trajectory failures.

Compares real-time computed centerline against reference centerline from
the non-realtime circuit_navigator to identify divergence points.
"""

import numpy as np
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor
from circuit_navigator.map_processing.centerline import CenterlineExtractor

from circuit_navigator_realtime.sensor.lidar_simulator import LidarEdgeSimulator, LidarScan
from circuit_navigator_realtime.filtering.edge_smoother import EdgeSmoother, SmootherConfig
from circuit_navigator_realtime.planning.local_centerline import LocalCenterlinePlanner, CenterlinePlannerConfig
from circuit_navigator_realtime.navigation.pure_pursuit import PurePursuitController, PurePursuitConfig
from circuit_navigator_realtime.simulation.bicycle_model import BicycleModel, VehicleState


@dataclass
class DiagnosticFrame:
    """Data collected at each simulation step."""
    step: int
    timestamp: float
    car_x: float
    car_y: float
    car_yaw: float
    car_speed: float
    steering: float

    # Edge data
    left_edge: np.ndarray
    right_edge: np.ndarray
    num_left_points: int
    num_right_points: int

    # Computed centerline
    computed_path: np.ndarray

    # Reference comparison
    nearest_ref_idx: int
    lateral_error: float
    ref_x: float
    ref_y: float

    # Island detection
    has_island_ahead: bool
    island_points: np.ndarray


def process_raw_grid(raw_grid: np.ndarray, resolution: float) -> np.ndarray:
    """Convert sparse road points to navigable binary grid."""
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)
    closing_radius_px = max(1, int(2.0 / resolution))
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)
    min_size_pixels = int(50 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size_pixels)
    safety_radius_px = max(1, int(0.5 / resolution))
    safety_kernel = disk(safety_radius_px)
    navigable = ndimage.binary_erosion(cleaned, safety_kernel)
    return navigable


class SequentialRefTracker:
    """Track position along reference path sequentially to avoid jump issues at roundabouts."""

    def __init__(self, reference: np.ndarray):
        self.reference = reference
        self.progress_idx = 0
        self.n = len(reference)

    def find_nearest(self, car_x: float, car_y: float) -> Tuple[int, float, float, float]:
        """Find nearest point using sequential progress tracking."""
        # Search within a forward window from current progress
        search_start = max(0, self.progress_idx - 10)
        search_end = min(self.n, self.progress_idx + 100)

        search_range = self.reference[search_start:search_end]
        if len(search_range) == 0:
            return 0, 999.0, self.reference[0, 0], self.reference[0, 1]

        dx = search_range[:, 0] - car_x
        dy = search_range[:, 1] - car_y
        distances = np.sqrt(dx * dx + dy * dy)
        local_idx = np.argmin(distances)
        nearest_idx = search_start + local_idx

        # Update progress if we've moved forward
        if nearest_idx >= self.progress_idx:
            self.progress_idx = nearest_idx

        lateral_error = distances[local_idx]
        return nearest_idx, lateral_error, self.reference[nearest_idx, 0], self.reference[nearest_idx, 1]


def find_nearest_reference_point(car_x: float, car_y: float, reference: np.ndarray) -> Tuple[int, float, float, float]:
    """Find nearest point on reference path and compute lateral error."""
    dx = reference[:, 0] - car_x
    dy = reference[:, 1] - car_y
    distances = np.sqrt(dx*dx + dy*dy)
    nearest_idx = np.argmin(distances)
    lateral_error = distances[nearest_idx]
    return nearest_idx, lateral_error, reference[nearest_idx, 0], reference[nearest_idx, 1]


def run_diagnostic(
    bag_path: str,
    sensor_range: float = 10.0,
    max_steps: int = 500,
    output_file: str = "diagnostic_data.npz"
) -> List[DiagnosticFrame]:
    """
    Run simulation and collect diagnostic data.
    """
    print("=" * 70)
    print("Diagnostic Analysis - Collecting Centerline Data")
    print("=" * 70)
    print(f"  Sensor range: {sensor_range}m")
    print(f"  Max steps: {max_steps}")
    print("=" * 70)

    # Extract map
    print("\n[1/4] Extracting map...")
    extractor = GridMapExtractor(bag_path, resolution=0.5)
    map_data = extractor.extract()
    navigable_grid = process_raw_grid(map_data.raw_data, map_data.resolution)

    # Extract reference centerline
    print("\n[2/4] Extracting reference centerline...")
    cl_extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
    centerline = cl_extractor.extract(navigable_grid, map_data.resolution, map_data.origin, method="skeleton")
    centerline = cl_extractor.resample_by_distance(centerline, spacing_m=0.5)
    reference_path = centerline.points_world
    print(f"  Reference: {len(reference_path)} points, {centerline.length_m:.1f}m")

    # Initialize components
    print("\n[3/4] Initializing components...")
    lidar = LidarEdgeSimulator(
        navigable_grid=navigable_grid,
        origin=map_data.origin,
        resolution=map_data.resolution,
        sensor_range=sensor_range,
        num_rays=180,
        fov=np.pi,
        reference_path=reference_path  # Use reference for robust edge classification
    )
    lidar.reset_path_progress()  # Ensure clean start
    print(f"  Using track-relative edge classification with sequential progress")

    smoother_config = SmootherConfig(ransac_enabled=True, savgol_enabled=True, kalman_enabled=True)
    edge_smoother = EdgeSmoother(smoother_config)

    planner_config = CenterlinePlannerConfig(
        min_forward_dist=2.0,
        max_forward_dist=sensor_range,
        safety_margin=0.5,
        right_bias_alpha=0.45,            # Normal driving: near center
        track_widening_threshold=1.1,     # Width ratio to trigger roundabout mode
        baseline_track_width=6.0,
        adaptive_bias_enabled=True,       # Enable adaptive bias for roundabout entry
        min_track_width_for_bias=2.5      # Minimum track width to check for roundabout
    )
    centerline_planner = LocalCenterlinePlanner(planner_config)

    # Renault Zoe dimensions with VERY responsive steering for roundabout entry
    # Shorter lookahead = faster reaction to path changes
    pp_config = PurePursuitConfig(wheelbase_m=2.588, min_lookahead_m=2.0, max_lookahead_m=5.0, max_steer_rad=0.6)
    pure_pursuit = PurePursuitController(pp_config)

    # Renault Zoe bicycle model (0.6 rad = 34 degrees max steering for tight roundabout turns)
    bicycle_model = BicycleModel(wheelbase=2.588, max_steer=0.6, max_speed=10.0)

    # Start position
    start_x, start_y = reference_path[0]
    dx = reference_path[5, 0] - reference_path[0, 0]
    dy = reference_path[5, 1] - reference_path[0, 1]
    start_yaw = np.arctan2(dy, dx)

    vehicle_state = VehicleState(x=start_x, y=start_y, yaw=start_yaw, speed=0.0, steering=0.0)

    # Collect data
    print("\n[4/4] Running simulation and collecting data...")
    frames: List[DiagnosticFrame] = []
    dt = 0.1

    # Use sequential tracker to properly handle roundabout overlaps
    ref_tracker = SequentialRefTracker(reference_path)

    for step in range(max_steps):
        # Sense
        scan = lidar.scan(vehicle_state.x, vehicle_state.y, vehicle_state.yaw)

        # Filter
        left_smooth, right_smooth = edge_smoother.smooth_edges(
            scan.left_edge, scan.right_edge,
            vehicle_state.x, vehicle_state.y, vehicle_state.yaw
        )

        # Plan
        computed_path = centerline_planner.compute(
            left_smooth, right_smooth,
            vehicle_state.x, vehicle_state.y, vehicle_state.yaw,
            island_points=scan.island_points,
            has_island_ahead=scan.has_island_ahead
        )

        # Find reference comparison (using sequential tracking)
        nearest_idx, lateral_error, ref_x, ref_y = ref_tracker.find_nearest(
            vehicle_state.x, vehicle_state.y
        )

        # Collect frame data
        frame = DiagnosticFrame(
            step=step,
            timestamp=step * dt,
            car_x=vehicle_state.x,
            car_y=vehicle_state.y,
            car_yaw=vehicle_state.yaw,
            car_speed=vehicle_state.speed,
            steering=vehicle_state.steering,
            left_edge=left_smooth.copy() if len(left_smooth) > 0 else np.array([]).reshape(0, 2),
            right_edge=right_smooth.copy() if len(right_smooth) > 0 else np.array([]).reshape(0, 2),
            num_left_points=len(left_smooth),
            num_right_points=len(right_smooth),
            computed_path=computed_path.copy() if len(computed_path) > 0 else np.array([]).reshape(0, 2),
            nearest_ref_idx=nearest_idx,
            lateral_error=lateral_error,
            ref_x=ref_x,
            ref_y=ref_y,
            has_island_ahead=scan.has_island_ahead,
            island_points=scan.island_points if scan.island_points is not None else np.array([]).reshape(0, 2)
        )
        frames.append(frame)

        # Control
        if len(computed_path) >= 2:
            steering, target_point, _ = pure_pursuit.compute_steering(
                vehicle_state.x, vehicle_state.y, vehicle_state.yaw,
                vehicle_state.speed, computed_path
            )
            target_speed = 5.0
        else:
            steering = 0.0
            target_speed = 2.0

        # Update
        vehicle_state = bicycle_model.update(vehicle_state, steering, target_speed, dt)

        # Check if car is on navigable ground
        row = int((vehicle_state.y - map_data.origin[1]) / map_data.resolution)
        col = int((vehicle_state.x - map_data.origin[0]) / map_data.resolution)
        is_on_track = False
        if 0 <= row < navigable_grid.shape[0] and 0 <= col < navigable_grid.shape[1]:
            is_on_track = navigable_grid[row, col]

        # Progress
        if step % 50 == 0:
            status = "ON TRACK" if is_on_track else "OFF TRACK!"
            print(f"  Step {step}: pos=({vehicle_state.x:.1f}, {vehicle_state.y:.1f}), "
                  f"lat_err={lateral_error:.2f}m, edges=({len(left_smooth)}, {len(right_smooth)}) [{status}]")

        # Warn immediately if off track
        if not is_on_track:
            print(f"  *** STEP {step}: CAR OFF TRACK at ({vehicle_state.x:.1f}, {vehicle_state.y:.1f}) ***")

        # Check if car went off track (lateral error too high)
        if lateral_error > 10.0:
            print(f"\n  WARNING: Car went off track at step {step}! Lateral error: {lateral_error:.2f}m")
            break

    print(f"\nCollected {len(frames)} frames")

    # Save data
    print(f"\nSaving diagnostic data to {output_file}...")

    # Convert frames to arrays for saving
    car_positions = np.array([[f.car_x, f.car_y] for f in frames])
    car_yaws = np.array([f.car_yaw for f in frames])
    lateral_errors = np.array([f.lateral_error for f in frames])
    ref_indices = np.array([f.nearest_ref_idx for f in frames])
    num_left = np.array([f.num_left_points for f in frames])
    num_right = np.array([f.num_right_points for f in frames])
    has_island = np.array([f.has_island_ahead for f in frames])

    np.savez(output_file,
             reference_path=reference_path,
             car_positions=car_positions,
             car_yaws=car_yaws,
             lateral_errors=lateral_errors,
             ref_indices=ref_indices,
             num_left=num_left,
             num_right=num_right,
             has_island=has_island)

    return frames, reference_path


def analyze_failures(frames: List[DiagnosticFrame], reference_path: np.ndarray):
    """Analyze frames to identify failure points and causes."""
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    # Find high-error frames
    lateral_errors = np.array([f.lateral_error for f in frames])
    mean_error = np.mean(lateral_errors)
    max_error = np.max(lateral_errors)

    print(f"\nOverall Statistics:")
    print(f"  Mean lateral error: {mean_error:.3f}m")
    print(f"  Max lateral error: {max_error:.3f}m")
    print(f"  Std lateral error: {np.std(lateral_errors):.3f}m")

    # Identify failure events (where error exceeds threshold)
    failure_threshold = 2.0  # 2m from centerline
    failure_mask = lateral_errors > failure_threshold
    failure_indices = np.where(failure_mask)[0]

    if len(failure_indices) == 0:
        print(f"\nNo failures detected (threshold: {failure_threshold}m)")
        return

    print(f"\nFailure Events (error > {failure_threshold}m):")
    print(f"  Total failure frames: {len(failure_indices)}")

    # Group consecutive failures
    failure_groups = []
    current_group = [failure_indices[0]]
    for i in range(1, len(failure_indices)):
        if failure_indices[i] - failure_indices[i-1] <= 5:  # Within 5 steps
            current_group.append(failure_indices[i])
        else:
            failure_groups.append(current_group)
            current_group = [failure_indices[i]]
    failure_groups.append(current_group)

    print(f"  Number of failure episodes: {len(failure_groups)}")

    # Analyze each failure episode
    print("\nFailure Episode Details:")
    for i, group in enumerate(failure_groups[:5]):  # First 5 episodes
        start_frame = frames[group[0]]
        peak_idx = group[np.argmax([frames[j].lateral_error for j in group])]
        peak_frame = frames[peak_idx]

        # Get context from before failure
        context_start = max(0, group[0] - 10)
        context_frames = frames[context_start:group[0]]

        print(f"\n  Episode {i+1}:")
        print(f"    Start step: {start_frame.step}")
        print(f"    Peak error: {peak_frame.lateral_error:.2f}m at step {peak_frame.step}")
        print(f"    Position: ({peak_frame.car_x:.1f}, {peak_frame.car_y:.1f})")
        print(f"    Heading: {np.degrees(peak_frame.car_yaw):.1f}°")
        print(f"    Reference index: {peak_frame.nearest_ref_idx}")
        print(f"    Edge points: left={peak_frame.num_left_points}, right={peak_frame.num_right_points}")
        print(f"    Island ahead: {peak_frame.has_island_ahead}")

        # Check for edge detection issues
        if peak_frame.num_left_points == 0:
            print(f"    ISSUE: No left edge detected!")
        if peak_frame.num_right_points == 0:
            print(f"    ISSUE: No right edge detected!")

        # Check if computed path exists
        if len(peak_frame.computed_path) < 5:
            print(f"    ISSUE: Computed path too short ({len(peak_frame.computed_path)} points)")

        # Compare with reference path at this location
        ref_idx = peak_frame.nearest_ref_idx
        if ref_idx > 5 and ref_idx < len(reference_path) - 5:
            ref_direction = reference_path[ref_idx + 5] - reference_path[ref_idx - 5]
            ref_yaw = np.arctan2(ref_direction[1], ref_direction[0])
            yaw_diff = peak_frame.car_yaw - ref_yaw
            while yaw_diff > np.pi: yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi: yaw_diff += 2 * np.pi
            print(f"    Heading deviation from reference: {np.degrees(yaw_diff):.1f}°")

    # Analyze edge detection statistics
    print("\n" + "-" * 50)
    print("Edge Detection Statistics:")
    num_left = np.array([f.num_left_points for f in frames])
    num_right = np.array([f.num_right_points for f in frames])

    print(f"  Left edge - mean: {np.mean(num_left):.1f}, min: {np.min(num_left)}, "
          f"zeros: {np.sum(num_left == 0)}")
    print(f"  Right edge - mean: {np.mean(num_right):.1f}, min: {np.min(num_right)}, "
          f"zeros: {np.sum(num_right == 0)}")

    # Correlation between edge count and error
    edge_total = num_left + num_right
    correlation = np.corrcoef(edge_total, lateral_errors)[0, 1]
    print(f"  Correlation (edge count vs error): {correlation:.3f}")

    return failure_groups


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic Analysis")
    parser.add_argument("--bag", type=str,
                        default="/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/")
    parser.add_argument("--range", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="diagnostic_data.npz")

    args = parser.parse_args()

    frames, reference = run_diagnostic(
        bag_path=args.bag,
        sensor_range=args.range,
        max_steps=args.steps,
        output_file=args.output
    )

    analyze_failures(frames, reference)
