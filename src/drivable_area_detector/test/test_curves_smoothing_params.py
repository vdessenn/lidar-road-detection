#!/usr/bin/env python3
"""
Simplified Parameter Tests for Curves Smoothing

Tests Savitzky-Golay smoothing parameters using MCAP rosbag data.
Generates graphical analysis for optimal parameter selection.
"""

import sys
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from scipy.signal import savgol_filter

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MCAP reader
from mcap_ros2.reader import read_ros2_messages


@dataclass
class SmoothingResult:
    """Single smoothing test result."""
    window: int
    polyorder: int
    mode: str
    smoothness: float  # Lower = smoother
    fidelity: float    # Distance from original (lower = better preserves shape)
    curvature_std: float  # Curvature variation


def read_mcap_frames(bag_path: str, topic: str = '/points', max_frames: int = 10) -> List[np.ndarray]:
    """Read point cloud frames from MCAP file."""
    print(f"Reading MCAP: {bag_path}")

    if not Path(bag_path).exists():
        print(f"ERROR: File not found: {bag_path}")
        return []

    frames = []
    for msg in read_ros2_messages(bag_path, topics=[topic]):
        pc_msg = msg.ros_msg
        point_step = pc_msg.point_step
        data = np.frombuffer(bytes(pc_msg.data), dtype=np.uint8)

        if len(data) < point_step:
            continue

        num_points = len(data) // point_step
        points = []

        for i in range(num_points):
            offset = i * point_step
            if offset + 16 > len(data):
                break
            x, y, z, intensity = struct.unpack('<4f', data[offset:offset+16])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z, intensity])

        if points:
            frames.append(np.array(points, dtype=np.float32))
            print(f"  Frame {len(frames)}: {len(points)} points")
            if len(frames) >= max_frames:
                break

    print(f"Total frames loaded: {len(frames)}")
    return frames


def extract_boundary(points: np.ndarray) -> np.ndarray:
    """Extract boundary points sorted by angle from centroid."""
    if len(points) < 10:
        return points

    # Use y extremes as boundary approximation
    y_sorted = points[np.argsort(points[:, 1])]
    n = len(y_sorted)
    left_boundary = y_sorted[:n//4]
    right_boundary = y_sorted[-n//4:]

    # Sort by x for a smooth curve
    left_sorted = left_boundary[np.argsort(left_boundary[:, 0])]
    right_sorted = right_boundary[np.argsort(right_boundary[:, 0])]

    # Return left boundary for testing
    return left_sorted[:, :2]  # Just x, y


def compute_smoothness(curve: np.ndarray) -> float:
    """Compute smoothness metric (second derivative magnitude)."""
    if len(curve) < 3:
        return 0

    # Second derivative approximation
    d2x = np.diff(curve[:, 0], n=2)
    d2y = np.diff(curve[:, 1], n=2) if curve.shape[1] > 1 else np.zeros_like(d2x)

    return np.mean(np.sqrt(d2x**2 + d2y**2))


def compute_curvature(curve: np.ndarray) -> np.ndarray:
    """Compute curvature along the curve."""
    if len(curve) < 3:
        return np.array([0])

    x, y = curve[:, 0], curve[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    denom = (dx**2 + dy**2)**1.5
    denom[denom < 1e-10] = 1e-10
    curvature = np.abs(dx * d2y - dy * d2x) / denom

    return curvature


def test_smoothing(curve: np.ndarray, window: int, polyorder: int, mode: str) -> SmoothingResult:
    """Test smoothing with given parameters."""
    if len(curve) < window:
        return SmoothingResult(window, polyorder, mode, 0, 0, 0)

    try:
        smoothed_x = savgol_filter(curve[:, 0], window, polyorder, mode=mode)
        smoothed_y = savgol_filter(curve[:, 1], window, polyorder, mode=mode)
        smoothed = np.column_stack([smoothed_x, smoothed_y])
    except Exception:
        return SmoothingResult(window, polyorder, mode, 0, 0, 0)

    # Metrics
    smoothness = compute_smoothness(smoothed)
    fidelity = np.mean(np.sqrt((curve[:, 0] - smoothed_x)**2 + (curve[:, 1] - smoothed_y)**2))
    curvature = compute_curvature(smoothed)
    curvature_std = np.std(curvature)

    return SmoothingResult(
        window=window,
        polyorder=polyorder,
        mode=mode,
        smoothness=smoothness,
        fidelity=fidelity,
        curvature_std=curvature_std
    )


def plot_window_polyorder_heatmap(results: List[SmoothingResult], output_dir: Path) -> str:
    """Generate heatmap of window_length vs polyorder."""
    # Extract unique values
    windows = sorted(set(r.window for r in results))
    polyorders = sorted(set(r.polyorder for r in results))

    # Create matrices
    smoothness_matrix = np.zeros((len(polyorders), len(windows)))
    fidelity_matrix = np.zeros((len(polyorders), len(windows)))
    score_matrix = np.zeros((len(polyorders), len(windows)))

    for r in results:
        wi = windows.index(r.window)
        pi = polyorders.index(r.polyorder)
        smoothness_matrix[pi, wi] = r.smoothness
        fidelity_matrix[pi, wi] = r.fidelity
        # Score: balance smoothness and fidelity (lower is better for both)
        max_smooth = max(rr.smoothness for rr in results) or 1
        max_fidelity = max(rr.fidelity for rr in results) or 1
        score_matrix[pi, wi] = (r.smoothness/max_smooth + r.fidelity/max_fidelity) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Smoothing Parameter Optimization: Window Length vs Polynomial Order', fontsize=12)

    # Smoothness heatmap
    im1 = axes[0].imshow(smoothness_matrix, cmap='Blues', aspect='auto')
    axes[0].set_xticks(range(len(windows)))
    axes[0].set_xticklabels(windows)
    axes[0].set_yticks(range(len(polyorders)))
    axes[0].set_yticklabels(polyorders)
    axes[0].set_xlabel('Window Length')
    axes[0].set_ylabel('Polynomial Order')
    axes[0].set_title('Smoothness (lower=smoother)')
    plt.colorbar(im1, ax=axes[0])

    # Fidelity heatmap
    im2 = axes[1].imshow(fidelity_matrix, cmap='Greens', aspect='auto')
    axes[1].set_xticks(range(len(windows)))
    axes[1].set_xticklabels(windows)
    axes[1].set_yticks(range(len(polyorders)))
    axes[1].set_yticklabels(polyorders)
    axes[1].set_xlabel('Window Length')
    axes[1].set_ylabel('Polynomial Order')
    axes[1].set_title('Fidelity (lower=closer to original)')
    plt.colorbar(im2, ax=axes[1])

    # Combined score heatmap
    im3 = axes[2].imshow(score_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_xticks(range(len(windows)))
    axes[2].set_xticklabels(windows)
    axes[2].set_yticks(range(len(polyorders)))
    axes[2].set_yticklabels(polyorders)
    axes[2].set_xlabel('Window Length')
    axes[2].set_ylabel('Polynomial Order')
    axes[2].set_title('Combined Score (lower=better)')
    plt.colorbar(im3, ax=axes[2])

    # Mark best
    best_idx = np.unravel_index(np.argmin(score_matrix), score_matrix.shape)
    axes[2].plot(best_idx[1], best_idx[0], 'k*', markersize=15)
    axes[2].annotate(f'Best: w={windows[best_idx[1]]}, p={polyorders[best_idx[0]]}',
                     xy=(best_idx[1], best_idx[0]), xytext=(best_idx[1]+0.5, best_idx[0]+0.5),
                     fontsize=9, color='black')

    plt.tight_layout()

    filename = output_dir / 'smoothing_heatmap.png'
    plt.savefig(filename, dpi=150)
    plt.close()

    return str(filename)


def plot_mode_comparison(results: Dict[str, List[SmoothingResult]], output_dir: Path) -> str:
    """Compare different smoothing modes."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Smoothing Mode Comparison', fontsize=12)

    modes = list(results.keys())
    smoothness = [np.mean([r.smoothness for r in results[m]]) for m in modes]
    fidelity = [np.mean([r.fidelity for r in results[m]]) for m in modes]
    curvature = [np.mean([r.curvature_std for r in results[m]]) for m in modes]

    axes[0].bar(modes, smoothness, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Avg Smoothness')
    axes[0].set_title('Smoothness by Mode')

    axes[1].bar(modes, fidelity, color='forestgreen', alpha=0.7)
    axes[1].set_ylabel('Avg Fidelity')
    axes[1].set_title('Fidelity by Mode')

    axes[2].bar(modes, curvature, color='coral', alpha=0.7)
    axes[2].set_ylabel('Curvature Std')
    axes[2].set_title('Curvature Stability by Mode')

    plt.tight_layout()

    filename = output_dir / 'smoothing_modes.png'
    plt.savefig(filename, dpi=150)
    plt.close()

    return str(filename)


def find_optimal(results: List[SmoothingResult]) -> SmoothingResult:
    """Find optimal parameters."""
    if not results:
        return None

    max_smooth = max(r.smoothness for r in results) or 1
    max_fidelity = max(r.fidelity for r in results) or 1

    scores = []
    for r in results:
        score = (r.smoothness/max_smooth + r.fidelity/max_fidelity) / 2
        scores.append(score)

    best_idx = np.argmin(scores)
    return results[best_idx]


def main():
    """Run all smoothing parameter tests."""
    print("=" * 60)
    print("CURVES SMOOTHING PARAMETER OPTIMIZATION")
    print("=" * 60)

    # Paths
    bag_path = Path('/home/ws/rosbag/spanlidar/spanlidar_0.mcap')
    output_dir = Path('/home/ws/doc/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frames
    print("\n1. Loading MCAP data...")
    frames = read_mcap_frames(str(bag_path), max_frames=10)

    if not frames:
        print("ERROR: No frames loaded. Check MCAP file.")
        return 1

    # Extract boundary curves from frames
    print("\n2. Extracting boundary curves...")
    curves = []
    for i, frame in enumerate(frames):
        boundary = extract_boundary(frame)
        if len(boundary) >= 10:
            curves.append(boundary)
            print(f"  Frame {i+1}: {len(boundary)} boundary points")

    if not curves:
        print("ERROR: No valid boundary curves extracted.")
        return 1

    # Test window_length vs polyorder
    print("\n3. Testing window_length vs polyorder...")
    window_lengths = [3, 5, 7, 9, 11, 15, 21]
    polyorders = [1, 2, 3, 4]

    all_results = []
    for window in window_lengths:
        for polyorder in polyorders:
            if polyorder >= window:
                continue  # polyorder must be < window

            frame_results = []
            for curve in curves:
                if len(curve) >= window:
                    result = test_smoothing(curve, window, polyorder, 'nearest')
                    frame_results.append(result)

            if frame_results:
                avg_result = SmoothingResult(
                    window=window,
                    polyorder=polyorder,
                    mode='nearest',
                    smoothness=np.mean([r.smoothness for r in frame_results]),
                    fidelity=np.mean([r.fidelity for r in frame_results]),
                    curvature_std=np.mean([r.curvature_std for r in frame_results])
                )
                all_results.append(avg_result)
                print(f"  window={window:2d}, poly={polyorder}: "
                      f"smooth={avg_result.smoothness:.4f}, fidelity={avg_result.fidelity:.4f}")

    # Generate heatmap
    print("\n4. Generating window/polyorder heatmap...")
    heatmap_path = plot_window_polyorder_heatmap(all_results, output_dir)
    print(f"  Saved: {heatmap_path}")

    # Test different modes
    print("\n5. Testing smoothing modes...")
    modes = ['nearest', 'wrap', 'mirror', 'constant']
    mode_results = {}

    for mode in modes:
        frame_results = []
        for curve in curves:
            if len(curve) >= 7:
                result = test_smoothing(curve, 7, 2, mode)
                frame_results.append(result)

        if frame_results:
            mode_results[mode] = frame_results
            avg_smooth = np.mean([r.smoothness for r in frame_results])
            avg_fidelity = np.mean([r.fidelity for r in frame_results])
            print(f"  mode={mode:10s}: smooth={avg_smooth:.4f}, fidelity={avg_fidelity:.4f}")

    # Generate mode comparison plot
    print("\n6. Generating mode comparison plot...")
    mode_path = plot_mode_comparison(mode_results, output_dir)
    print(f"  Saved: {mode_path}")

    # Find optimal parameters
    optimal = find_optimal(all_results)

    # Print recommendations
    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETER RECOMMENDATIONS")
    print("=" * 60)

    if optimal:
        print("\nRecommended params.yaml values:")
        print("```yaml")
        print("curves_smoothing_node:")
        print(f"  window_length: {optimal.window}  # Savitzky-Golay window")
        print(f"  polyorder: {optimal.polyorder}  # Polynomial order")
        print(f"  mode: 'nearest'  # Edge handling mode")
        print("```")

        print(f"\nMetrics:")
        print(f"  Smoothness: {optimal.smoothness:.4f}")
        print(f"  Fidelity:   {optimal.fidelity:.4f}")
        print(f"  Curvature Std: {optimal.curvature_std:.4f}")

    print(f"\nPlots saved to: {output_dir}")
    print("\nDONE.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
