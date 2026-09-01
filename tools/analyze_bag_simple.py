#!/usr/bin/env python3
"""
Simple bag analyzer using mcap library directly.
Analyzes drivable area detector performance from ROS2 bag.
"""

import sys
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    print("ERROR: mcap libraries not found. Install with:")
    print("  pip install mcap mcap-ros2-support")
    sys.exit(1)


def analyze_bag(bag_dir):
    """Analyze MCAP bag files."""
    bag_path = Path(bag_dir)
    mcap_files = sorted(bag_path.glob("*.mcap"))

    if not mcap_files:
        print(f"ERROR: No .mcap files found in {bag_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Analyzing bag: {bag_path}")
    print(f"Found {len(mcap_files)} MCAP files")
    print(f"{'='*80}\n")

    stats = defaultdict(list)
    frame_data = {}

    # Process all MCAP files
    for mcap_file in mcap_files:
        print(f"Processing: {mcap_file.name}")

        with open(mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                topic = channel.topic

                # Only process PointCloud2 topics
                if topic not in ['/patchworkpp/ground', '/ground_segmentation_points',
                                '/road_surface', '/ring_boundaries']:
                    continue

                # Get frame timestamp
                frame_stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec / 1e9
                msg_time = message.log_time / 1e9

                if frame_stamp not in frame_data:
                    frame_data[frame_stamp] = {}

                # Extract point count
                n_points = ros_msg.width * ros_msg.height

                frame_data[frame_stamp][topic] = {
                    'n_points': n_points,
                    'timestamp': msg_time,
                }

                # Extract intensity if this is road_surface or ground
                if topic in ['/road_surface', '/patchworkpp/ground'] and hasattr(ros_msg, 'data'):
                    try:
                        data = np.frombuffer(ros_msg.data, dtype=np.uint8)
                        if len(data) > 0:
                            points = data.reshape(-1, ros_msg.point_step)
                            # Pandora format: intensity at offset 12
                            if ros_msg.point_step >= 16:
                                intensity = points[:, 12:16].view(np.float32).ravel()
                                frame_data[frame_stamp][topic]['intensity_min'] = float(intensity.min())
                                frame_data[frame_stamp][topic]['intensity_max'] = float(intensity.max())
                                frame_data[frame_stamp][topic]['intensity_mean'] = float(intensity.mean())
                                frame_data[frame_stamp][topic]['intensity_std'] = float(intensity.std())
                    except Exception as e:
                        pass

    # Analyze frames
    print(f"\nAnalyzing {len(frame_data)} frames...\n")

    sorted_frames = sorted(frame_data.keys())

    for i, frame_stamp in enumerate(sorted_frames):
        frame = frame_data[frame_stamp]

        n_ground = frame.get('/patchworkpp/ground', {}).get('n_points', 0)
        n_road = frame.get('/road_surface', {}).get('n_points', 0)
        n_boundaries = frame.get('/ring_boundaries', {}).get('n_points', 0)

        stats['n_ground'].append(n_ground)
        stats['n_road'].append(n_road)
        stats['n_boundaries'].append(n_boundaries)

        # Calculate latency
        if '/patchworkpp/ground' in frame and '/road_surface' in frame:
            t_input = frame['/patchworkpp/ground']['timestamp']
            t_output = frame['/road_surface']['timestamp']
            latency = (t_output - t_input) * 1000  # ms
            stats['latency'].append(latency)

        # Calculate reduction ratio
        if n_ground > 0:
            reduction_ratio = (n_road / n_ground) * 100
            stats['reduction_ratio'].append(reduction_ratio)

        # Store intensity stats
        if '/road_surface' in frame and 'intensity_mean' in frame['/road_surface']:
            stats['road_intensity_mean'].append(frame['/road_surface']['intensity_mean'])
            stats['road_intensity_std'].append(frame['/road_surface']['intensity_std'])
            stats['road_intensity_min'].append(frame['/road_surface']['intensity_min'])
            stats['road_intensity_max'].append(frame['/road_surface']['intensity_max'])

        # Print sample frames
        if i < 10 or i % 20 == 0:
            print(f"Frame {i:3d} | Ground: {n_ground:5d} → Road: {n_road:4d} → Boundaries: {n_boundaries:3d}", end="")
            if stats['latency']:
                print(f" | Latency: {stats['latency'][-1]:5.1f}ms", end="")
            if n_ground > 0:
                print(f" | Reduction: {reduction_ratio:5.1f}%", end="")
            print()

    # Generate statistical report
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total frames: {len(sorted_frames)}")

    if stats['n_ground']:
        print(f"\nPoint Counts:")
        print(f"  Ground (Patchwork++):")
        print(f"    Mean: {np.mean(stats['n_ground']):.0f} ± {np.std(stats['n_ground']):.0f}")
        print(f"    Range: [{np.min(stats['n_ground']):.0f}, {np.max(stats['n_ground']):.0f}]")

        print(f"\n  Road (after RANSAC):")
        print(f"    Mean: {np.mean(stats['n_road']):.0f} ± {np.std(stats['n_road']):.0f}")
        print(f"    Range: [{np.min(stats['n_road']):.0f}, {np.max(stats['n_road']):.0f}]")
        print(f"    Expected: 500-1500")

        if stats['n_boundaries']:
            print(f"\n  Boundaries:")
            print(f"    Mean: {np.mean(stats['n_boundaries']):.0f} ± {np.std(stats['n_boundaries']):.0f}")

    if stats['reduction_ratio']:
        print(f"\nReduction Ratio (road/ground):")
        print(f"  Mean: {np.mean(stats['reduction_ratio']):.1f}%")
        print(f"  Range: [{np.min(stats['reduction_ratio']):.1f}%, {np.max(stats['reduction_ratio']):.1f}%]")

    if stats['latency']:
        print(f"\nLatency (Patchwork++ → Road Surface):")
        print(f"  Mean: {np.mean(stats['latency']):.1f}ms ± {np.std(stats['latency']):.1f}ms")
        print(f"  Median: {np.median(stats['latency']):.1f}ms")
        print(f"  95th percentile: {np.percentile(stats['latency'], 95):.1f}ms")
        print(f"  Max: {np.max(stats['latency']):.1f}ms")
        print(f"  Target: <30ms for real-time")

    if stats['road_intensity_mean']:
        print(f"\nRoad Surface Intensity:")
        print(f"  Mean: {np.mean(stats['road_intensity_mean']):.1f} ± {np.std(stats['road_intensity_std']):.1f}")
        print(f"  Range: [{np.min(stats['road_intensity_min']):.1f}, {np.max(stats['road_intensity_max']):.1f}]")
        print(f"  Current filter: [0.0, 15.0]")

    # Performance assessment
    print(f"\n{'='*80}")
    print("PERFORMANCE ASSESSMENT & RECOMMENDATIONS")
    print(f"{'='*80}\n")

    recommendations = []

    # Latency assessment
    if stats['latency']:
        avg_latency = np.mean(stats['latency'])
        max_latency = np.max(stats['latency'])

        if avg_latency < 30:
            print(f"✅ LATENCY: Average {avg_latency:.1f}ms < 30ms target (PASS)")
        else:
            print(f"❌ LATENCY: Average {avg_latency:.1f}ms > 30ms target (FAIL)")
            recommendations.append("PERFORMANCE: Further optimization needed")
            recommendations.append("  - Consider reducing window_size further")
            recommendations.append("  - Reduce ransac_max_trials to 15")

        if max_latency > 50:
            print(f"⚠️  WARNING: Max latency {max_latency:.1f}ms (spikes detected)")

    # Point count assessment
    if stats['n_road']:
        avg_road = np.mean(stats['n_road'])
        min_road = np.min(stats['n_road'])
        max_road = np.max(stats['n_road'])

        if 500 <= avg_road <= 1500:
            print(f"✅ ROAD POINTS: Average {avg_road:.0f} within expected [500-1500] (PASS)")
        elif avg_road < 500:
            print(f"⚠️  ROAD POINTS: Average {avg_road:.0f} < 500 (TOO AGGRESSIVE)")
            recommendations.append("DETECTION: Filtering too aggressive")
            recommendations.append("  - Increase ransac_residual_threshold: 0.03 → 0.04")
            recommendations.append("  - Decrease min_window_points: 10 → 8")
            recommendations.append("  - Increase max_road_intensity: 15.0 → 18.0")
        else:
            print(f"⚠️  ROAD POINTS: Average {avg_road:.0f} > 1500 (TOO MUCH NOISE)")
            recommendations.append("DETECTION: Too much noise passing through")
            recommendations.append("  - Decrease ransac_residual_threshold: 0.03 → 0.025")
            recommendations.append("  - Decrease max_road_intensity: 15.0 → 12.0")
            recommendations.append("  - Increase min_window_points: 10 → 12")

        # Check variability
        std_road = np.std(stats['n_road'])
        if std_road > 300:
            print(f"⚠️  VARIABILITY: High std dev {std_road:.0f} (unstable detection)")
            recommendations.append("STABILITY: High frame-to-frame variability")
            recommendations.append("  - Consider adding temporal smoothing")

    # Intensity assessment
    if stats['road_intensity_mean']:
        avg_intensity = np.mean(stats['road_intensity_mean'])
        max_intensity_observed = np.max(stats['road_intensity_max'])

        if max_intensity_observed > 15:
            print(f"⚠️  INTENSITY: Observed max {max_intensity_observed:.1f} > filter max 15.0")
            recommendations.append("INTENSITY FILTER: Road markings may be included")
            recommendations.append(f"  - Observed intensity range: [0, {max_intensity_observed:.1f}]")
            recommendations.append("  - Consider if this is desired behavior")

    # Print recommendations
    if recommendations:
        print(f"\n{'='*80}")
        print("RECOMMENDED PARAMETER ADJUSTMENTS")
        print(f"{'='*80}\n")
        for rec in recommendations:
            print(rec)

    # Save JSON report
    report = {
        'summary': {
            'total_frames': len(sorted_frames),
            'avg_ground_points': float(np.mean(stats['n_ground'])) if stats['n_ground'] else 0,
            'avg_road_points': float(np.mean(stats['n_road'])) if stats['n_road'] else 0,
            'avg_latency_ms': float(np.mean(stats['latency'])) if stats['latency'] else 0,
            'max_latency_ms': float(np.max(stats['latency'])) if stats['latency'] else 0,
            'avg_reduction_ratio': float(np.mean(stats['reduction_ratio'])) if stats['reduction_ratio'] else 0,
        },
        'statistics': {k: [float(v) for v in vals] for k, vals in stats.items()},
        'recommendations': recommendations
    }

    report_path = bag_path / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Full report saved to: {report_path}")
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: analyze_bag_simple.py <bag_directory>")
        print("\nExample:")
        print("  python3 tools/analyze_bag_simple.py analysis_10s/")
        sys.exit(1)

    analyze_bag(sys.argv[1])
