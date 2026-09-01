#!/usr/bin/env python3
"""
Simplified Parameter Tests for Ring Boundary Extraction

Tests different parameters for 02_ring_boundary_extraction.py using MCAP rosbag data.
Generates graphical analysis for optimal parameter selection.
"""

import sys
import struct
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MCAP reader
from mcap_ros2.reader import read_ros2_messages

# Load boundary detector module
import importlib.util
MODULE_PATH = Path(__file__).parent.parent / 'drivable_area_detector' / '02_ring_boundary_extraction.py'
spec = importlib.util.spec_from_file_location("ring_boundary_extraction", MODULE_PATH)
ring_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ring_module)
BoundaryDetector = ring_module.BoundaryDetector

# Load params.yaml
PARAMS_PATH = Path(__file__).parent.parent / 'config' / 'params.yaml'


def load_params_from_yaml() -> Dict:
    """Load parameters from params.yaml configuration file."""
    with open(PARAMS_PATH, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Extract ring_boundary_node parameters
    ros_params = yaml_data.get('/**', {}).get('ros__parameters', {})
    ring_params = ros_params.get('ring_boundary_node', {})

    return {
        'discontinuity_threshold': ring_params.get('discontinuity_threshold', 0.052),
        'arc_continuity_threshold': ring_params.get('arc_continuity_threshold', 0.5),
        'radius_variance_threshold': ring_params.get('radius_variance_threshold', 0.1),
        'boundary_confidence_threshold': ring_params.get('boundary_confidence_threshold', 0.5),
        'min_boundary_width': ring_params.get('min_boundary_width', 1.5),
        'history_frames': ring_params.get('history_frames', 3),
    }


@dataclass
class TestResult:
    """Single parameter test result."""
    value: float
    detection_rate: float
    mean_width: float
    stability: float


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


def add_ring_ids(points: np.ndarray) -> np.ndarray:
    """Add synthetic ring IDs based on distance."""
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    ring_ids = np.clip((distances / 2.5).astype(int), 0, 11)
    return np.column_stack([points[:, :3], ring_ids])


def evaluate_parameter(frames: List[np.ndarray], param_name: str, param_value: float,
                       base_params: Dict) -> TestResult:
    """Test a single parameter value."""
    params = base_params.copy()
    params[param_name] = param_value

    detector = BoundaryDetector(
        discontinuity_threshold=params.get('discontinuity_threshold', 0.052),
        history_frames=params.get('history_frames', 3),
        arc_tolerance=params.get('arc_continuity_threshold', 0.5),
        radius_variance=params.get('radius_variance_threshold', 0.1),
        logger=None
    )

    valid_detections = 0
    widths = []

    for frame in frames:
        points_with_ring = add_ring_ids(frame)
        frame_y_min, frame_y_max = float('inf'), float('-inf')

        for ring_id in range(12):
            ring_mask = points_with_ring[:, 3] == ring_id
            ring_points = points_with_ring[ring_mask][:, :3]

            if len(ring_points) < 3:
                continue

            result = detector.detect_boundaries(ring_points, ring_id)

            if result.confidence >= params.get('boundary_confidence_threshold', 0.5):
                width = result.y_max - result.y_min
                if width >= params.get('min_boundary_width', 1.5):
                    frame_y_min = min(frame_y_min, result.y_min)
                    frame_y_max = max(frame_y_max, result.y_max)

        if frame_y_min < frame_y_max:
            valid_detections += 1
            widths.append(frame_y_max - frame_y_min)

    return TestResult(
        value=param_value,
        detection_rate=valid_detections / len(frames) if frames else 0,
        mean_width=np.mean(widths) if widths else 0,
        stability=np.std(widths) if len(widths) > 1 else 0
    )


def sweep_parameter(frames: List[np.ndarray], param_name: str,
                    values: List[float], base_params: Dict) -> List[TestResult]:
    """Sweep through parameter values."""
    results = []
    for val in values:
        result = evaluate_parameter(frames, param_name, val, base_params)
        results.append(result)
        print(f"  {param_name}={val:.3f}: det={result.detection_rate*100:.1f}%, "
              f"width={result.mean_width:.2f}m, stability={result.stability:.2f}m")
    return results


def plot_results(param_name: str, results: List[TestResult], output_dir: Path) -> str:
    """Generate plot for parameter sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Parameter Sweep: {param_name}', fontsize=12)

    values = [r.value for r in results]
    det_rates = [r.detection_rate * 100 for r in results]
    widths = [r.mean_width for r in results]
    stabilities = [r.stability for r in results]

    # Detection Rate
    axes[0].plot(values, det_rates, 'b-o', linewidth=2)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Detection Rate (%)')
    axes[0].set_title('Detection Rate')
    axes[0].grid(True, alpha=0.3)

    # Boundary Width
    axes[1].plot(values, widths, 'g-s', linewidth=2)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Mean Width (m)')
    axes[1].set_title('Road Width')
    axes[1].grid(True, alpha=0.3)

    # Stability (lower is better)
    axes[2].plot(values, stabilities, 'r-^', linewidth=2)
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel('Std Dev (m)')
    axes[2].set_title('Stability (lower=better)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    filename = output_dir / f'sweep_{param_name}.png'
    plt.savefig(filename, dpi=150)
    plt.close()

    return str(filename)


def find_optimal(results: List[TestResult]) -> TestResult:
    """Find optimal parameter value using combined score."""
    max_stab = max(r.stability for r in results) or 1

    scores = []
    for r in results:
        det_score = r.detection_rate
        stab_score = 1 - (r.stability / max_stab)
        width_factor = min(r.mean_width / 3.5, 1.5) if r.mean_width > 0 else 0
        scores.append(det_score * stab_score * width_factor)

    best_idx = np.argmax(scores)
    return results[best_idx]


def main():
    """Run all parameter tests."""
    print("=" * 60)
    print("RING BOUNDARY PARAMETER OPTIMIZATION")
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

    # Load base parameters from params.yaml
    print("\n2. Loading parameters from params.yaml...")
    base_params = load_params_from_yaml()
    print(f"   Loaded parameters: {base_params}")

    recommendations = {}

    # Test 1: Discontinuity threshold
    print("\n3. Testing discontinuity_threshold...")
    values = [0.02, 0.03, 0.04, 0.052, 0.08, 0.10, 0.15]  # Include current value (0.03)
    results = sweep_parameter(frames, 'discontinuity_threshold', values, base_params)
    plot_results('discontinuity_threshold', results, output_dir)
    recommendations['discontinuity_threshold'] = find_optimal(results)

    # Test 2: Arc continuity threshold
    print("\n4. Testing arc_continuity_threshold...")
    values = [0.2, 0.5, 1.0, 1.5, 2.0]
    results = sweep_parameter(frames, 'arc_continuity_threshold', values, base_params)
    plot_results('arc_continuity_threshold', results, output_dir)
    recommendations['arc_continuity_threshold'] = find_optimal(results)

    # Test 3: Confidence threshold
    print("\n5. Testing boundary_confidence_threshold...")
    values = [0.2, 0.4, 0.5, 0.6, 0.8]
    results = sweep_parameter(frames, 'boundary_confidence_threshold', values, base_params)
    plot_results('boundary_confidence_threshold', results, output_dir)
    recommendations['boundary_confidence_threshold'] = find_optimal(results)

    # Test 4: Min boundary width
    print("\n6. Testing min_boundary_width...")
    values = [0.5, 1.0, 1.5, 2.0, 2.5]
    results = sweep_parameter(frames, 'min_boundary_width', values, base_params)
    plot_results('min_boundary_width', results, output_dir)
    recommendations['min_boundary_width'] = find_optimal(results)

    # Test 5: History frames
    print("\n7. Testing history_frames...")
    values = [1, 2, 3, 4, 5]
    results = sweep_parameter(frames, 'history_frames', values, base_params)
    plot_results('history_frames', results, output_dir)
    recommendations['history_frames'] = find_optimal(results)

    # Print recommendations
    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETER RECOMMENDATIONS")
    print("=" * 60)
    print("\nRecommended params.yaml values:")
    print("```yaml")
    print("ring_boundary_node:")
    for param, result in recommendations.items():
        val = result.value
        if isinstance(val, float):
            print(f"  {param}: {val:.3f}  # det={result.detection_rate*100:.0f}%")
        else:
            print(f"  {param}: {val}  # det={result.detection_rate*100:.0f}%")
    print("```")

    print(f"\nPlots saved to: {output_dir}")
    print("\nDONE.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
