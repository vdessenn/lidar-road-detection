#!/usr/bin/env python3
"""Test edge cases in centerline computation."""
import numpy as np
import sys
sys.path.insert(0, "..")

from circuit_navigator_realtime.planning.local_centerline import LocalCenterlinePlanner, CenterlinePlannerConfig

config = CenterlinePlannerConfig(
    min_forward_dist=2.0,
    max_forward_dist=10.0,
    safety_margin=0.5,
    right_bias_alpha=0.45,
    baseline_track_width=6.0,
    adaptive_bias_enabled=True
)
planner = LocalCenterlinePlanner(config)

car_x, car_y, car_yaw = 0.0, 0.0, 0.0  # Car at origin, facing East

# Test 1: Both edges available
print("Test 1: Both edges available (6m track width)")
left_edge = np.array([[5, 3], [10, 3], [15, 3]])  # 3m to the left
right_edge = np.array([[5, -3], [10, -3], [15, -3]])  # 3m to the right
path = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)
if len(path) > 0:
    path_local = planner._to_car_frame(path, car_x, car_y, car_yaw)
    print(f"  Path Y range: {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
    print(f"  Expected: ~0 (center) with slight right bias")
else:
    print("  No path computed!")

# Test 2: Only left edge
print("\nTest 2: Only left edge available")
left_edge = np.array([[5, 3], [10, 3], [15, 3]])
right_edge = np.array([])  # No right edge
path = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)
if len(path) > 0:
    path_local = planner._to_car_frame(path, car_x, car_y, car_yaw)
    print(f"  Path Y range: {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
    print(f"  Expected: ~{3 - 0.55*6:.1f} (left edge at 3, offset right by {0.55*6:.1f})")
else:
    print("  No path computed!")

# Test 3: Only right edge
print("\nTest 3: Only right edge available")
left_edge = np.array([])  # No left edge
right_edge = np.array([[5, -5], [10, -5], [15, -5]])  # Right edge at -5
path = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)
if len(path) > 0:
    path_local = planner._to_car_frame(path, car_x, car_y, car_yaw)
    print(f"  Path Y range: {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
    print(f"  Expected: ~{-5 + 0.45*6:.1f} (right edge at -5, offset left by {0.45*6:.1f})")
else:
    print("  No path computed!")

# Test 4: No edges
print("\nTest 4: No edges available")
left_edge = np.array([])
right_edge = np.array([])
path = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)
if len(path) > 0:
    path_local = planner._to_car_frame(path, car_x, car_y, car_yaw)
    print(f"  Path Y range: {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
    print(f"  Expected: ~0 (straight ahead)")
else:
    print("  No path computed!")

# Test 5: Very few edge points (edge case)
print("\nTest 5: Only 1 left edge point")
left_edge = np.array([[5, 3]])  # Only 1 point
right_edge = np.array([[5, -3], [10, -3], [15, -3]])
path = planner.compute(left_edge, right_edge, car_x, car_y, car_yaw)
if len(path) > 0:
    path_local = planner._to_car_frame(path, car_x, car_y, car_yaw)
    print(f"  Path Y range: {path_local[:, 1].min():.2f} to {path_local[:, 1].max():.2f}")
else:
    print("  No path computed (falls back to right edge only)")
