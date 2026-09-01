#!/usr/bin/env python3
"""
Tuning tool for Road Points Extraction Node (01_road_points_extraction.py)

Analyzes /patchworkpp/ground to find optimal parameters:
- Intensity thresholds (min_road_intensity, max_road_intensity)
- Noise filtering effectiveness

Usage:
    # Live analysis (subscribe to topics)
    python3 tools/tune_road_extraction.py --live --duration 30

    # Bag file analysis
    python3 tools/tune_road_extraction.py --bag bags/spanlidar_0.mcap

    # Interactive mode (adjust thresholds in real-time)
    python3 tools/tune_road_extraction.py --interactive
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import yaml

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ROS2 imports for live mode
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# MCAP imports for bag analysis
try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    HAS_MCAP = True
except ImportError:
    HAS_MCAP = False


@dataclass
class FrameStats:
    """Statistics for a single frame."""
    n_ground: int = 0
    n_low_intensity: int = 0
    n_after_noise_filter: int = 0
    intensity_min: float = 0.0
    intensity_max: float = 0.0
    intensity_mean: float = 0.0
    intensity_std: float = 0.0


@dataclass
class AnalysisResult:
    """Aggregated analysis results."""
    frames: List[FrameStats] = field(default_factory=list)
    all_intensities: List[float] = field(default_factory=list)

    # Spatial distribution
    all_x: List[float] = field(default_factory=list)
    all_y: List[float] = field(default_factory=list)
    all_z: List[float] = field(default_factory=list)


def parse_pointcloud2(msg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PointCloud2 message to extract x, y, z, intensity."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    points = data.reshape(-1, msg.point_step)

    x = points[:, 0:4].view(np.float32).ravel()
    y = points[:, 4:8].view(np.float32).ravel()
    z = points[:, 8:12].view(np.float32).ravel()
    intensity = points[:, 12:16].view(np.float32).ravel()

    return x, y, z, intensity


def simulate_intensity_filter(intensity: np.ndarray,
                              min_intensity: float,
                              max_intensity: float) -> np.ndarray:
    """Simulate intensity filtering and return mask."""
    return (intensity >= min_intensity) & (intensity <= max_intensity)


def simulate_noise_filter(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          x_bin_size: float = 2.0,
                          y_bin_size: float = 0.6,
                          min_points_per_bin: int = 3,
                          max_z_variance: float = 0.35) -> np.ndarray:
    """
    Simulate geometric noise filtering.
    Returns indices of points that would pass the filter.
    """
    if len(x) == 0:
        return np.array([], dtype=np.int32)

    # Spatial binning
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
    y_bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)

    if len(x_bins) < 2 or len(y_bins) < 2:
        return np.arange(len(x))

    x_indices = np.clip(np.digitize(x, x_bins) - 1, 0, len(x_bins) - 2)
    y_indices = np.clip(np.digitize(y, y_bins) - 1, 0, len(y_bins) - 2)

    # Group points by bin
    bin_map = defaultdict(list)
    for i, (xi, yi) in enumerate(zip(x_indices, y_indices)):
        bin_map[(xi, yi)].append(i)

    # Filter bins by density and planarity
    valid_indices = []
    for (xi, yi), indices in bin_map.items():
        if len(indices) < min_points_per_bin:
            continue

        z_bin = z[indices]
        if np.var(z_bin) > max_z_variance:
            continue

        valid_indices.extend(indices)

    return np.array(valid_indices, dtype=np.int32)


def analyze_frame(x: np.ndarray, y: np.ndarray, z: np.ndarray, intensity: np.ndarray,
                  min_intensity: float = 0.0,
                  max_intensity: float = 20.0) -> FrameStats:
    """Analyze a single frame with given parameters."""
    stats = FrameStats()
    stats.n_ground = len(intensity)

    if len(intensity) == 0:
        return stats

    # Intensity filter
    intensity_mask = simulate_intensity_filter(intensity, min_intensity, max_intensity)
    stats.n_low_intensity = np.sum(intensity_mask)

    # Noise filter on intensity-filtered points
    if stats.n_low_intensity > 0:
        x_filtered = x[intensity_mask]
        y_filtered = y[intensity_mask]
        z_filtered = z[intensity_mask]

        valid_indices = simulate_noise_filter(x_filtered, y_filtered, z_filtered)
        stats.n_after_noise_filter = len(valid_indices)

    # Intensity statistics (on all ground points)
    stats.intensity_min = intensity.min()
    stats.intensity_max = intensity.max()
    stats.intensity_mean = intensity.mean()
    stats.intensity_std = intensity.std()

    return stats


def print_analysis_summary(result: AnalysisResult,
                           min_intensity: float = 0.0,
                           max_intensity: float = 20.0):
    """Print summary of analysis results."""
    if not result.frames:
        print("No frames analyzed!")
        return

    print(f"\n{'='*80}")
    print("ROAD POINTS EXTRACTION - PARAMETER TUNING ANALYSIS")
    print(f"{'='*80}\n")

    # Overall statistics
    n_frames = len(result.frames)
    total_ground = sum(f.n_ground for f in result.frames)
    total_intensity_filtered = sum(f.n_low_intensity for f in result.frames)
    total_noise_filtered = sum(f.n_after_noise_filter for f in result.frames)

    print(f"Frames analyzed: {n_frames}")
    print(f"Total ground points: {total_ground:,}")
    print(f"Average ground points/frame: {total_ground / n_frames:.0f}")

    print(f"\n--- Current Parameters ---")
    print(f"  min_road_intensity: {min_intensity}")
    print(f"  max_road_intensity: {max_intensity}")

    print(f"\n--- Filtering Results ---")
    print(f"  After intensity filter: {total_intensity_filtered:,} ({100*total_intensity_filtered/total_ground:.1f}%)")
    print(f"  After noise filter: {total_noise_filtered:,} ({100*total_noise_filtered/total_ground:.1f}%)")
    print(f"  Average road points/frame: {total_noise_filtered / n_frames:.0f}")

    # Intensity distribution
    if result.all_intensities:
        intensities = np.array(result.all_intensities)
        print(f"\n--- Intensity Distribution ---")
        print(f"  Min:    {intensities.min():.2f}")
        print(f"  Max:    {intensities.max():.2f}")
        print(f"  Mean:   {intensities.mean():.2f}")
        print(f"  Median: {np.median(intensities):.2f}")
        print(f"  Std:    {intensities.std():.2f}")

        print(f"\n  Percentiles:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(intensities, p)
            print(f"    {p:2d}th: {val:.2f}")

        # Histogram
        print(f"\n--- Intensity Histogram ---")
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
        hist, _ = np.histogram(intensities, bins=bins)
        total = len(intensities)

        print(f"  {'Range':>10s} | {'Count':>10s} | {'Pct':>8s} | Bar")
        print(f"  {'-'*50}")
        for i in range(len(bins)-1):
            count = hist[i]
            pct = 100 * count / total
            bar = '█' * int(pct / 2)
            print(f"  {bins[i]:4.0f}-{bins[i+1]:4.0f} | {count:10,d} | {pct:6.2f}% | {bar}")


def print_recommendations(result: AnalysisResult):
    """Print parameter recommendations based on analysis."""
    if not result.all_intensities:
        return

    intensities = np.array(result.all_intensities)
    n_frames = len(result.frames)
    avg_ground = sum(f.n_ground for f in result.frames) / n_frames

    print(f"\n{'='*80}")
    print("PARAMETER RECOMMENDATIONS")
    print(f"{'='*80}\n")

    # Target: 500-2500 road points per frame
    target_min, target_max = 500, 2500

    # Test different thresholds
    test_thresholds = [10, 15, 20, 25, 30, 40, 50]

    print(f"Target road points per frame: {target_min}-{target_max}")
    print(f"Average ground points per frame: {avg_ground:.0f}")
    print(f"\n  {'Threshold':>10s} | {'Points kept':>12s} | {'Pct':>8s} | {'Est. road pts':>15s} | Status")
    print(f"  {'-'*70}")

    best_threshold = 20.0
    best_diff = float('inf')

    for thresh in test_thresholds:
        n_kept = np.sum(intensities <= thresh)
        pct = 100 * n_kept / len(intensities)
        est_road = avg_ground * (n_kept / len(intensities)) * 0.5  # ~50% pass noise filter

        if target_min <= est_road <= target_max:
            status = "✓ GOOD"
        elif est_road < target_min:
            status = "⚠ Too few"
        else:
            status = "⚠ Too many"

        target_mid = (target_min + target_max) / 2
        diff = abs(est_road - target_mid)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh

        print(f"  {thresh:10.0f} | {n_kept:12,d} | {pct:6.2f}% | {est_road:15.0f} | {status}")

    print(f"\n--- Recommended Configuration ---")
    print(f"  road_points_extraction_node:")
    print(f"    min_road_intensity: 0.0")
    print(f"    max_road_intensity: {best_threshold:.1f}")

    # Additional recommendations
    print(f"\n--- Additional Tips ---")
    p90 = np.percentile(intensities, 90)
    if p90 < 30:
        print(f"  • Most points have low intensity (90th percentile = {p90:.1f})")
        print(f"    Consider using max_road_intensity: {p90:.0f}")

    if avg_ground < 3000:
        print(f"  • Low ground point count ({avg_ground:.0f})")
        print(f"    Check Patchwork++ parameters or LiDAR coverage")

    if avg_ground > 10000:
        print(f"  • High ground point count ({avg_ground:.0f})")
        print(f"    Consider stricter intensity filtering")


def plot_results(result: AnalysisResult, output_path: Optional[Path] = None,
                 current_threshold: float = 20.0):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available - skipping plots")
        return

    if not result.all_intensities:
        return

    intensities = np.array(result.all_intensities)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Intensity histogram
    ax1 = axes[0, 0]
    ax1.hist(intensities, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(current_threshold, color='r', linestyle='--',
                label=f'Current threshold ({current_threshold:.1f})')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Count')
    ax1.set_title('Intensity Distribution of Ground Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Points per frame over time
    ax2 = axes[0, 1]
    frame_numbers = range(len(result.frames))
    ground_counts = [f.n_ground for f in result.frames]
    road_counts = [f.n_after_noise_filter for f in result.frames]

    ax2.plot(frame_numbers, ground_counts, 'b-', alpha=0.7, label='Ground points')
    ax2.plot(frame_numbers, road_counts, 'g-', alpha=0.7, label='Road points (after filter)')
    ax2.axhline(500, color='r', linestyle=':', alpha=0.5, label='Target min (500)')
    ax2.axhline(2500, color='r', linestyle=':', alpha=0.5, label='Target max (2500)')
    ax2.fill_between(frame_numbers, 500, 2500, alpha=0.1, color='green')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Point Count')
    ax2.set_title('Points per Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Spatial distribution (top-down view)
    ax3 = axes[1, 0]
    if result.all_x and result.all_y:
        # Subsample for performance
        n_sample = min(50000, len(result.all_x))
        indices = np.random.choice(len(result.all_x), n_sample, replace=False)
        x_sample = np.array(result.all_x)[indices]
        y_sample = np.array(result.all_y)[indices]
        i_sample = np.array(result.all_intensities)[indices]

        scatter = ax3.scatter(x_sample, y_sample, c=i_sample, s=1, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax3, label='Intensity')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Spatial Distribution (colored by intensity)')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

    # 4. Threshold analysis
    ax4 = axes[1, 1]
    thresholds = np.arange(5, 60, 2)
    points_kept = [100 * np.sum(intensities <= t) / len(intensities) for t in thresholds]

    ax4.plot(thresholds, points_kept, 'b-', linewidth=2)
    ax4.axvline(current_threshold, color='r', linestyle='--',
                label=f'Current ({current_threshold:.1f})')
    ax4.set_xlabel('max_road_intensity threshold')
    ax4.set_ylabel('% of ground points kept')
    ax4.set_title('Effect of Intensity Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")
    else:
        plt.show()


def analyze_bag(bag_path: str, min_intensity: float, max_intensity: float,
                topic: str = '/patchworkpp/ground') -> AnalysisResult:
    """Analyze a bag file."""
    if not HAS_MCAP:
        print("ERROR: mcap libraries not found. Install with:")
        print("  pip install mcap mcap-ros2-support")
        sys.exit(1)

    result = AnalysisResult()
    path = Path(bag_path)

    # Find MCAP files
    if path.is_dir():
        mcap_files = sorted(path.glob("*.mcap"))
    else:
        mcap_files = [path]

    if not mcap_files:
        print(f"ERROR: No .mcap files found at {bag_path}")
        return result

    print(f"\nAnalyzing bag: {bag_path}")
    print(f"Found {len(mcap_files)} MCAP file(s)")
    print(f"Using topic: {topic}")

    for mcap_file in mcap_files:
        print(f"  Processing: {mcap_file.name}...", end=" ", flush=True)
        frame_count = 0

        try:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    # Process specified topic
                    if channel.topic != topic:
                        continue

                    x, y, z, intensity = parse_pointcloud2(ros_msg)
                    if len(intensity) == 0:
                        continue

                    # Analyze frame
                    stats = analyze_frame(x, y, z, intensity, min_intensity, max_intensity)
                    result.frames.append(stats)

                    # Store data for distribution analysis (subsample for memory)
                    if frame_count % 3 == 0:  # Every 3rd frame
                        result.all_intensities.extend(intensity.tolist())
                        result.all_x.extend(x.tolist())
                        result.all_y.extend(y.tolist())
                        result.all_z.extend(z.tolist())

                    frame_count += 1

            print(f"✓ ({frame_count} frames)")

        except Exception as e:
            print(f"⚠ Error: {e}")

    return result


def analyze_live(duration: float, min_intensity: float, max_intensity: float) -> AnalysisResult:
    """Analyze live ROS topics."""
    if not HAS_ROS2:
        print("ERROR: ROS2 not available")
        sys.exit(1)

    result = AnalysisResult()

    class AnalyzerNode(Node):
        def __init__(self):
            super().__init__('road_extraction_tuner')

            qos = QoSProfile(depth=1)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            qos.durability = DurabilityPolicy.VOLATILE

            self.subscription = self.create_subscription(
                PointCloud2,
                '/patchworkpp/ground',
                self.callback,
                qos
            )
            self.frame_count = 0

        def callback(self, msg):
            x, y, z, intensity = parse_pointcloud2(msg)
            if len(intensity) == 0:
                return

            stats = analyze_frame(x, y, z, intensity, min_intensity, max_intensity)
            result.frames.append(stats)

            # Store subset of data (every 5th frame to save memory)
            if self.frame_count % 5 == 0:
                result.all_intensities.extend(intensity.tolist())
                result.all_x.extend(x.tolist())
                result.all_y.extend(y.tolist())
                result.all_z.extend(z.tolist())

            self.frame_count += 1

            # Progress indicator
            if self.frame_count % 10 == 0:
                avg_road = sum(f.n_after_noise_filter for f in result.frames[-10:]) / 10
                print(f"\r  Frames: {self.frame_count}, Avg road pts: {avg_road:.0f}", end="", flush=True)

    rclpy.init()
    node = AnalyzerNode()

    print(f"\nAnalyzing live topics for {duration} seconds...")
    print(f"  Subscribing to: /patchworkpp/ground")

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print()  # New line after progress
        node.destroy_node()
        rclpy.shutdown()

    return result


def load_params_from_yaml() -> dict:
    """
    Load parameters from params.yaml file.
    Location: src/drivable_area_detector/config/params.yaml

    Returns dict with extracted parameters or defaults if file not found.
    """
    # Fixed path relative to this script
    script_dir = Path(__file__).parent
    params_file = script_dir.parent / 'src' / 'drivable_area_detector' / 'config' / 'params.yaml'

    # Default fallback values
    defaults = {
        'min_road_intensity': 0.0,
        'max_road_intensity': 20.0
    }

    if not params_file.exists():
        print(f"⚠ params.yaml not found at {params_file}, using defaults")
        return defaults

    try:
        with open(params_file, 'r') as f:
            config = yaml.safe_load(f)

        # Navigate to road_points_extraction_node parameters
        ros_params = config.get('/**', {}).get('ros__parameters', {})
        node_params = ros_params.get('road_points_extraction_node', {})

        min_intensity = node_params.get('min_road_intensity', defaults['min_road_intensity'])
        max_intensity = node_params.get('max_road_intensity', defaults['max_road_intensity'])

        print(f"✓ Loaded parameters from {params_file}")
        print(f"  min_road_intensity: {min_intensity}")
        print(f"  max_road_intensity: {max_intensity}")

        return {
            'min_road_intensity': min_intensity,
            'max_road_intensity': max_intensity
        }

    except Exception as e:
        print(f"⚠ Error loading params.yaml: {e}")
        print(f"  Using defaults: {defaults}")
        return defaults


def main():
    # Load parameters from params.yaml first
    yaml_params = load_params_from_yaml()

    parser = argparse.ArgumentParser(
        description='Tune parameters for Road Points Extraction Node',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze bag file (uses params.yaml values)
    python3 tools/tune_road_extraction.py --bag bags/spanlidar_0.mcap

    # Live analysis for 30 seconds
    python3 tools/tune_road_extraction.py --live --duration 30

    # Test with custom thresholds (override params.yaml)
    python3 tools/tune_road_extraction.py --bag bags/ --min-intensity 0 --max-intensity 25
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bag', type=str, help='Path to bag file or directory')
    mode.add_argument('--live', action='store_true', help='Analyze live topics')

    parser.add_argument('--duration', type=float, default=30.0,
                        help='Duration for live analysis (seconds)')
    parser.add_argument('--topic', type=str, default='/patchworkpp/ground',
                        help='Topic to analyze (default: /patchworkpp/ground, use /points for raw)')
    parser.add_argument('--min-intensity', type=float, default=None,
                        help=f'Min intensity threshold to test (default: from params.yaml = {yaml_params["min_road_intensity"]})')
    parser.add_argument('--max-intensity', type=float, default=None,
                        help=f'Max intensity threshold to test (default: from params.yaml = {yaml_params["max_road_intensity"]})')
    parser.add_argument('--plot', type=str, default="doc/output/road_extraction_analysis.png",
                        help='Save plot to file (e.g., /doc/output/analysis.png)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')

    args = parser.parse_args()

    # Use yaml_params as defaults if not overridden by command line
    min_intensity = args.min_intensity if args.min_intensity is not None else yaml_params['min_road_intensity']
    max_intensity = args.max_intensity if args.max_intensity is not None else yaml_params['max_road_intensity']

    print(f"\n{'='*80}")
    print("FINAL ANALYSIS PARAMETERS")
    print(f"{'='*80}")
    print(f"  min_road_intensity: {min_intensity} {'(from params.yaml)' if args.min_intensity is None else '(from command line)'}")
    print(f"  max_road_intensity: {max_intensity} {'(from params.yaml)' if args.max_intensity is None else '(from command line)'}")
    print(f"{'='*80}\n")

    # Run analysis
    if args.bag:
        result = analyze_bag(args.bag, min_intensity, max_intensity, args.topic)
    else:
        result = analyze_live(args.duration, min_intensity, max_intensity)

    # Print results
    print_analysis_summary(result, min_intensity, max_intensity)
    print_recommendations(result)

    # Generate plots
    if not args.no_plot:
        output_path = Path(args.plot) if args.plot else None
        plot_results(result, output_path, current_threshold=max_intensity)

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
