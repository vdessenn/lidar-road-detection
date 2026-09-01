#!/usr/bin/env python3
"""Check if there's any road data in the return path area."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_navigator.map_processing.grid_extractor import GridMapExtractor


def check_return_path(bag_path: str, resolution: float = 0.5):
    """Check for data in the return path area."""

    extractor = GridMapExtractor(bag_path, topic="/road_whole", resolution=resolution)
    data = extractor.extract()

    raw_grid = data.raw_data
    origin = data.origin
    rows, cols = raw_grid.shape

    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # The return path should be around X = -90 to -120 (between the two roundabouts)
    # at Y levels from about 50 to 130

    print("Checking for road data in RETURN PATH area:")
    print("Expected: X from -120 to -90, Y from 50 to 120")
    print("-" * 50)

    for x_target in [-120, -115, -110, -105, -100, -95, -90]:
        col = int((x_target - origin[0]) / resolution)
        if 0 <= col < cols:
            count = 0
            y_values = []
            for row in range(rows):
                if has_point[row, col]:
                    y = origin[1] + row * resolution
                    if 50 <= y <= 130:  # Return path Y range
                        count += 1
                        y_values.append(y)
            if count > 0:
                print(f"  X={x_target}: {count} points, Y range: {min(y_values):.1f} - {max(y_values):.1f}")
            else:
                print(f"  X={x_target}: NO DATA")

    # Also check the main track area for comparison
    print("\nFor comparison - MAIN TRACK (going up) area:")
    print("Expected: X from -175 to -150, Y from 50 to 100")
    print("-" * 50)

    for x_target in [-175, -170, -165, -160, -155, -150]:
        col = int((x_target - origin[0]) / resolution)
        if 0 <= col < cols:
            count = 0
            y_values = []
            for row in range(rows):
                if has_point[row, col]:
                    y = origin[1] + row * resolution
                    if 50 <= y <= 100:
                        count += 1
                        y_values.append(y)
            if count > 0:
                print(f"  X={x_target}: {count} points, Y range: {min(y_values):.1f} - {max(y_values):.1f}")
            else:
                print(f"  X={x_target}: NO DATA")


if __name__ == "__main__":
    bag_path = "/home/lara/Desktop/transfer_11310705_files_2760ab59/sy27_road_ground_truth2/"
    check_return_path(bag_path)
