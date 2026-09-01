#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys

# Check if NPZ files have ring field
validation_dir = Path('tools/validation_output')

# Find latest session
sessions = sorted(validation_dir.glob('session_*'), reverse=True)
if not sessions:
    print("No validation sessions found")
    sys.exit(1)

latest = sessions[0]
print(f"Checking session: {latest}")

# Check good frames
good_dir = latest / 'good'
if good_dir.exists():
    npz_files = list(good_dir.glob('*.npz'))
    if npz_files:
        print(f"\nFound {len(npz_files)} NPZ files in good/")

        # Load first file
        first_file = npz_files[0]
        print(f"\nInspecting: {first_file.name}")

        data = np.load(first_file)
        print(f"\nAvailable keys: {list(data.keys())}")

        # Check road points
        if 'road_x' in data:
            road_x = data['road_x']
            print(f"\nRoad points: {len(road_x)}")
            print(f"  X range: [{road_x.min():.2f}, {road_x.max():.2f}]")

            if 'road_y' in data:
                road_y = data['road_y']
                print(f"  Y range: [{road_y.min():.2f}, {road_y.max():.2f}]")

            # Check if ring field exists
            if 'road_ring' in data:
                road_ring = data['road_ring']
                print(f"\n✓ road_ring field EXISTS!")
                print(f"  Ring dtype: {road_ring.dtype}")
                print(f"  Unique rings: {np.unique(road_ring)}")
                print(f"  Ring value range: [{road_ring.min()}, {road_ring.max()}]")

                # Count points per ring
                unique_rings, counts = np.unique(road_ring, return_counts=True)
                print(f"\n  Points per ring:")
                for ring_id, count in zip(unique_rings, counts):
                    print(f"    Ring {ring_id}: {count} points")
            else:
                print(f"\n✗ road_ring field MISSING!")
                print("\nYou need to:")
                print("  1. Modify detection_validator.py to extract ring field")
                print("  2. Re-run detection_validator.py to regenerate NPZ files")
                print("  3. Label frames again (or copy old validation.json)")

        # Show example of what ring extraction should look like
        print(f"\n" + "="*60)
        print("EXPECTED PointCloud2 structure for Velodyne:")
        print("="*60)
        print("Offset | Size | Field")
        print("-------|------|-------")
        print("  0-3  |  4   | x (float32)")
        print("  4-7  |  4   | y (float32)")
        print("  8-11 |  4   | z (float32)")
        print(" 12-15 |  4   | (padding)")
        print(" 16-19 |  4   | intensity (float32)")
        print(" 20-21 |  2   | ring (uint16)  ← THIS IS WHAT WE NEED")
        print(" 22-23 |  2   | (padding)")

    else:
        print("No NPZ files found in good/")
else:
    print("No good/ directory found")
