#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, "..")

from circuit_navigator_realtime.sensor.lidar_simulator import LidarEdgeSimulator
from circuit_navigator_realtime.planning.local_centerline import LocalCenterlinePlanner, CenterlinePlannerConfig
from circuit_navigator_realtime.filtering.edge_smoother import EdgeSmoother, SmootherConfig
from scipy import ndimage
from skimage.morphology import closing, disk, remove_small_objects
from circuit_navigator.map_processing.grid_extractor import GridMapExtractor

extractor = GridMapExtractor("/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/", resolution=0.5)
map_data = extractor.extract()

has_point = ~np.isnan(map_data.raw_data) & (map_data.raw_data != 0)
closed = closing(has_point, disk(4))
cleaned = remove_small_objects(closed, min_size=200)
navigable = ndimage.binary_erosion(cleaned, disk(1))

lidar = LidarEdgeSimulator(
    navigable_grid=navigable,
    origin=map_data.origin,
    resolution=map_data.resolution,
    sensor_range=10.0,
    num_rays=180,
    fov=np.pi
)

smoother = EdgeSmoother(SmootherConfig(ransac_enabled=True, savgol_enabled=True, kalman_enabled=True))

planner_config = CenterlinePlannerConfig(
    min_forward_dist=2.0,
    max_forward_dist=10.0,
    safety_margin=0.5,
    right_bias_alpha=0.45,  # More centered (0.5=center)
    track_widening_threshold=1.5,
    baseline_track_width=6.0,
    adaptive_bias_enabled=True
)
planner = LocalCenterlinePlanner(planner_config)

car_x, car_y, car_yaw = -170.0, 52.2, np.radians(130)
scan = lidar.scan(car_x, car_y, car_yaw)

left_smooth, right_smooth = smoother.smooth_edges(
    scan.left_edge, scan.right_edge, car_x, car_y, car_yaw
)

computed_path = planner.compute(
    left_smooth, right_smooth, car_x, car_y, car_yaw,
    island_points=scan.island_points, has_island_ahead=scan.has_island_ahead
)

print(f"Car at ({car_x:.1f}, {car_y:.1f}), heading {np.degrees(car_yaw):.0f} degrees")
print(f"Island ahead: {scan.has_island_ahead}")
print(f"Island points: {len(scan.island_points) if scan.island_points is not None else 0}")
if scan.island_points is not None and len(scan.island_points) > 0:
    island_local = lidar._to_car_frame(scan.island_points, car_x, car_y, car_yaw)
    print(f"Island X range: {island_local[:, 0].min():.1f} to {island_local[:, 0].max():.1f}")
    print(f"Island Y range: {island_local[:, 1].min():.1f} to {island_local[:, 1].max():.1f}")
print()

left_local = lidar._to_car_frame(left_smooth, car_x, car_y, car_yaw)
right_local = lidar._to_car_frame(right_smooth, car_x, car_y, car_yaw)
print(f"Left edge Y (car frame): {left_local[:, 1].min():.1f} to {left_local[:, 1].max():.1f}")
print(f"Right edge Y (car frame): {right_local[:, 1].min():.1f} to {right_local[:, 1].max():.1f}")

if len(computed_path) > 0:
    path_local = lidar._to_car_frame(computed_path, car_x, car_y, car_yaw)
    print(f"Centerline Y (car frame): {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
    print(f"Average lateral offset: {np.mean(path_local[:, 1]):.2f}m (negative = right-biased)")
else:
    print("No path computed!")

# Detailed analysis
print("\n--- Detailed Edge Analysis ---")
print(f"Left edge points: {len(left_smooth)}")
print(f"Right edge points: {len(right_smooth)}")

if len(left_local) > 0:
    print(f"\nLeft edge in car frame:")
    print(f"  X (forward) range: {left_local[:, 0].min():.1f} to {left_local[:, 0].max():.1f}")
    print(f"  Y (lateral) range: {left_local[:, 1].min():.1f} to {left_local[:, 1].max():.1f}")

if len(right_local) > 0:
    print(f"\nRight edge in car frame:")
    print(f"  X (forward) range: {right_local[:, 0].min():.1f} to {right_local[:, 0].max():.1f}")
    print(f"  Y (lateral) range: {right_local[:, 1].min():.1f} to {right_local[:, 1].max():.1f}")

# Check at specific forward distances
print("\n--- Track Width at Forward Distances ---")
from scipy.interpolate import interp1d

left_sorted = left_local[np.argsort(left_local[:, 0])]
right_sorted = right_local[np.argsort(right_local[:, 0])]

# Common forward range
x_min = max(left_sorted[0, 0], right_sorted[0, 0], 2.0)
x_max = min(left_sorted[-1, 0], right_sorted[-1, 0])

if x_max > x_min:
    left_interp = interp1d(left_sorted[:, 0], left_sorted[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    right_interp = interp1d(right_sorted[:, 0], right_sorted[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')

    for x in np.linspace(x_min, x_max, 5):
        y_left = float(left_interp(x))
        y_right = float(right_interp(x))
        width = y_left - y_right
        midpoint = (y_left + y_right) / 2
        biased = y_right + 0.35 * (y_left - y_right)
        print(f"  X={x:.1f}m: left_Y={y_left:.1f}, right_Y={y_right:.1f}, width={width:.1f}m, mid={midpoint:.1f}, biased(0.35)={biased:.1f}")
else:
    print(f"  No common forward range! Left: {left_sorted[0, 0]:.1f}-{left_sorted[-1, 0]:.1f}, Right: {right_sorted[0, 0]:.1f}-{right_sorted[-1, 0]:.1f}")
