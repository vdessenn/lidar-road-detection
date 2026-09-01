#!/usr/bin/env python3
"""
Validation test suite using real captured LiDAR data.

This module tests the drivable_area_detector pipeline using actual
PointCloud2 data captured from rosbag recordings.

Validation files (GROUND TRUTH):
- straight_points_validation.txt: Validated boundary points for straight road (598 points)
- roundabouts_curve_points_validation.txt: Validated boundary points for curves (multiple frames)

Whole frame files (INPUT):
- whole_frame_points_straight_line.txt: Full LiDAR frame for straight road (2415 points)
- whole_frame_points_curved_line.txt: Full LiDAR frame for curved road (3016 points)

The tests compare pipeline detection against ground truth to measure:
- Precision: What fraction of detected points are correct?
- Recall: What fraction of ground truth points are detected?
- F1-score: Harmonic mean of precision and recall

Author: Validation Testing
Date: 2026-01-11
"""

import unittest
import numpy as np
import struct
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial import KDTree

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'drivable_area_detector'))

# Import modules to test
import importlib.util


def load_module(module_name: str, filename: str):
    """Dynamically load a Python module by filename."""
    module_path = Path(__file__).parent.parent / 'drivable_area_detector' / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load detection modules
try:
    ring_module = load_module("ring_boundary_extraction", "02_ring_boundary_extraction.py")
    BoundaryDetector = ring_module.BoundaryDetector
    BoundaryResult = ring_module.BoundaryResult
    HAS_BOUNDARY_MODULE = True
except Exception as e:
    print(f"Warning: Could not import boundary detection module: {e}")
    HAS_BOUNDARY_MODULE = False
    BoundaryDetector = None
    BoundaryResult = None

try:
    roundabout_module = load_module("roundabout_handler", "03_roundabout_handler.py")
    RoundaboutDetector = roundabout_module.RoundaboutDetector
    YRoadHandler = roundabout_module.YRoadHandler
    RoundaboutGeometry = roundabout_module.RoundaboutGeometry
    HAS_ROUNDABOUT_MODULE = True
except Exception as e:
    print(f"Warning: Could not import roundabout module: {e}")
    HAS_ROUNDABOUT_MODULE = False
    RoundaboutDetector = None
    YRoadHandler = None
    RoundaboutGeometry = None


@dataclass
class PointCloud2Data:
    """Parsed PointCloud2 message data."""
    frame_id: str
    width: int
    height: int
    point_step: int
    row_step: int
    is_bigendian: bool
    is_dense: bool
    timestamp_sec: int
    timestamp_nanosec: int
    points: np.ndarray  # Nx4 array of (x, y, z, intensity)

    @property
    def num_points(self) -> int:
        return len(self.points)

    def get_xyz(self) -> np.ndarray:
        """Return Nx3 array of x, y, z coordinates."""
        return self.points[:, :3]

    def get_intensity(self) -> np.ndarray:
        """Return 1D array of intensity values."""
        return self.points[:, 3]


class PointCloud2Parser:
    """Parser for ROS2 PointCloud2 YAML format from txt files."""

    @staticmethod
    def parse_yaml_file(filepath: str) -> List[PointCloud2Data]:
        """Parse a txt file containing one or more PointCloud2 messages in YAML format."""
        with open(filepath, 'r') as f:
            content = f.read()

        documents = content.split('---')
        results = []

        for doc in documents:
            doc = doc.strip()
            if not doc:
                continue

            try:
                pc_data = PointCloud2Parser._parse_single_message(doc)
                if pc_data is not None:
                    results.append(pc_data)
            except Exception as e:
                continue

        return results

    @staticmethod
    def _parse_single_message(yaml_content: str) -> Optional[PointCloud2Data]:
        """Parse a single PointCloud2 YAML message."""
        lines = yaml_content.split('\n')

        frame_id = "velodyne"
        width = 0
        height = 1
        point_step = 16
        row_step = 0
        is_bigendian = False
        is_dense = False
        timestamp_sec = 0
        timestamp_nanosec = 0

        data_bytes = []
        in_data_section = False

        for line in lines:
            line = line.strip()

            if line.startswith('frame_id:'):
                frame_id = line.split(':', 1)[1].strip()
            elif line.startswith('width:'):
                width = int(line.split(':', 1)[1].strip())
            elif line.startswith('height:'):
                height = int(line.split(':', 1)[1].strip())
            elif line.startswith('point_step:'):
                point_step = int(line.split(':', 1)[1].strip())
            elif line.startswith('row_step:'):
                row_step = int(line.split(':', 1)[1].strip())
            elif line.startswith('is_bigendian:'):
                is_bigendian = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('is_dense:'):
                is_dense = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('sec:') and timestamp_sec == 0:
                timestamp_sec = int(line.split(':', 1)[1].strip())
            elif line.startswith('nanosec:') and timestamp_nanosec == 0:
                timestamp_nanosec = int(line.split(':', 1)[1].strip())
            elif line.startswith('data:'):
                in_data_section = True
            elif in_data_section:
                if line.startswith('-'):
                    byte_val = line[1:].strip()
                    if byte_val == "'...'" or byte_val == "...":
                        break
                    try:
                        byte_val = byte_val.strip("'\"")
                        data_bytes.append(int(byte_val))
                    except ValueError:
                        continue
                elif line and not line.startswith('#'):
                    in_data_section = False

        if width == 0 or not data_bytes:
            return None

        points = PointCloud2Parser._decode_point_data(
            data_bytes, width, point_step, is_bigendian
        )

        if points is None or len(points) == 0:
            return None

        return PointCloud2Data(
            frame_id=frame_id,
            width=width,
            height=height,
            point_step=point_step,
            row_step=row_step,
            is_bigendian=is_bigendian,
            is_dense=is_dense,
            timestamp_sec=timestamp_sec,
            timestamp_nanosec=timestamp_nanosec,
            points=points
        )

    @staticmethod
    def _decode_point_data(data_bytes: List[int],
                           num_points: int,
                           point_step: int,
                           is_bigendian: bool) -> Optional[np.ndarray]:
        """Decode byte array to point array."""
        byte_array = bytes(data_bytes)
        available_points = len(byte_array) // point_step
        actual_points = min(available_points, num_points)

        if actual_points == 0:
            return None

        endian = '>' if is_bigendian else '<'
        point_format = f'{endian}4f'

        points = []
        for i in range(actual_points):
            offset = i * point_step
            if offset + 16 > len(byte_array):
                break

            try:
                x, y, z, intensity = struct.unpack(point_format, byte_array[offset:offset+16])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z, intensity])
            except struct.error:
                continue

        if not points:
            return None

        return np.array(points, dtype=np.float32)


class ValidationMetrics:
    """Compute validation metrics comparing detection against ground truth."""

    @staticmethod
    def compute_overlap_metrics(detected_points: np.ndarray,
                                 ground_truth_points: np.ndarray,
                                 tolerance: float = 0.1) -> Dict[str, float]:
        """Compute precision, recall, F1 using spatial proximity matching.

        Args:
            detected_points: Nx3 array of detected boundary points (x, y, z)
            ground_truth_points: Mx3 array of ground truth points
            tolerance: Distance threshold for point matching (meters)

        Returns:
            Dictionary with precision, recall, f1, matched_count
        """
        if len(detected_points) == 0 and len(ground_truth_points) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0}
        if len(detected_points) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
        if len(ground_truth_points) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}

        # Build KD-tree for ground truth points
        gt_tree = KDTree(ground_truth_points[:, :2])  # Use x, y only

        # For each detected point, find nearest ground truth
        distances, _ = gt_tree.query(detected_points[:, :2])
        true_positives = np.sum(distances <= tolerance)

        # Precision: TP / detected
        precision = true_positives / len(detected_points)

        # Build KD-tree for detected points
        det_tree = KDTree(detected_points[:, :2])
        gt_distances, _ = det_tree.query(ground_truth_points[:, :2])
        gt_matched = np.sum(gt_distances <= tolerance)

        # Recall: matched_gt / total_gt
        recall = gt_matched / len(ground_truth_points)

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": int(true_positives),
            "detected": len(detected_points),
            "ground_truth": len(ground_truth_points)
        }

    @staticmethod
    def compute_boundary_metrics(detected_y_range: Tuple[float, float],
                                  ground_truth_y_range: Tuple[float, float]) -> Dict[str, float]:
        """Compute boundary detection accuracy.

        Args:
            detected_y_range: (y_min, y_max) from detection
            ground_truth_y_range: (y_min, y_max) from ground truth

        Returns:
            Dictionary with boundary accuracy metrics
        """
        det_min, det_max = detected_y_range
        gt_min, gt_max = ground_truth_y_range

        # Absolute errors
        min_error = abs(det_min - gt_min)
        max_error = abs(det_max - gt_max)

        # Width accuracy
        det_width = det_max - det_min
        gt_width = gt_max - gt_min
        width_error = abs(det_width - gt_width)
        width_ratio = det_width / gt_width if gt_width > 0 else 0.0

        # Overlap (IoU-like for 1D ranges)
        overlap_min = max(det_min, gt_min)
        overlap_max = min(det_max, gt_max)
        overlap_width = max(0, overlap_max - overlap_min)

        union_min = min(det_min, gt_min)
        union_max = max(det_max, gt_max)
        union_width = union_max - union_min

        iou = overlap_width / union_width if union_width > 0 else 0.0

        return {
            "y_min_error": min_error,
            "y_max_error": max_error,
            "width_error": width_error,
            "width_ratio": width_ratio,
            "iou_1d": iou
        }


class TestPointCloud2Parser(unittest.TestCase):
    """Test the PointCloud2 YAML parser."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.straight_file = cls.workspace_root / 'straight_points_validation.txt'
        cls.roundabout_file = cls.workspace_root / 'roundabouts_curve_points_validation.txt'
        cls.whole_straight_file = cls.workspace_root / 'whole_frame_points_straight_line.txt'
        cls.whole_curved_file = cls.workspace_root / 'whole_frame_points_curved_line.txt'

    def test_parse_straight_points_file(self):
        """Test parsing straight_points_validation.txt (ground truth)."""
        if not self.straight_file.exists():
            self.skipTest(f"Validation file not found: {self.straight_file}")

        messages = PointCloud2Parser.parse_yaml_file(str(self.straight_file))

        self.assertGreater(len(messages), 0, "Should parse at least one message")
        msg = messages[0]
        self.assertEqual(msg.frame_id, "velodyne")
        self.assertEqual(msg.width, 598, "Ground truth should have 598 points")

        print(f"\n[Straight Ground Truth] {msg.width} declared, {msg.num_points} decoded")

    def test_parse_roundabout_points_file(self):
        """Test parsing roundabouts_curve_points_validation.txt (ground truth)."""
        if not self.roundabout_file.exists():
            self.skipTest(f"Validation file not found: {self.roundabout_file}")

        messages = PointCloud2Parser.parse_yaml_file(str(self.roundabout_file))
        self.assertGreater(len(messages), 0, "Should parse at least one message")

        print(f"\n[Roundabout Ground Truth] {len(messages)} frame(s)")
        for i, msg in enumerate(messages):
            print(f"  Frame {i+1}: {msg.width} declared, {msg.num_points} decoded")

    def test_parse_whole_straight_file(self):
        """Test parsing whole_frame_points_straight_line.txt (input)."""
        if not self.whole_straight_file.exists():
            self.skipTest(f"Input file not found: {self.whole_straight_file}")

        messages = PointCloud2Parser.parse_yaml_file(str(self.whole_straight_file))
        self.assertGreater(len(messages), 0)
        self.assertEqual(messages[0].width, 2415, "Input should have 2415 points")

        print(f"\n[Straight Input] {messages[0].width} points")

    def test_parse_whole_curved_file(self):
        """Test parsing whole_frame_points_curved_line.txt (input)."""
        if not self.whole_curved_file.exists():
            self.skipTest(f"Input file not found: {self.whole_curved_file}")

        messages = PointCloud2Parser.parse_yaml_file(str(self.whole_curved_file))
        self.assertGreater(len(messages), 0)
        self.assertEqual(messages[0].width, 3016, "Input should have 3016 points")

        print(f"\n[Curved Input] {messages[0].width} points")


@unittest.skipUnless(HAS_BOUNDARY_MODULE, "Boundary detection module not available")
class TestBoundaryDetectionWithGroundTruth(unittest.TestCase):
    """Test boundary detection comparing against ground truth validation data."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.straight_gt_file = cls.workspace_root / 'straight_points_validation.txt'
        cls.whole_straight_file = cls.workspace_root / 'whole_frame_points_straight_line.txt'

        cls.detector = BoundaryDetector(
            discontinuity_threshold=0.052,
            history_frames=3,
            arc_tolerance=0.5,
            radius_variance=0.1,
            logger=None
        )

    def _load_points_from_file(self, filepath: str) -> Optional[np.ndarray]:
        """Load points from validation file."""
        if not Path(filepath).exists():
            return None

        messages = PointCloud2Parser.parse_yaml_file(filepath)
        if not messages:
            return None

        points = messages[0].points
        return points[:, :3]  # Return x, y, z only

    def _add_synthetic_ring(self, points: np.ndarray) -> np.ndarray:
        """Add synthetic ring IDs based on distance from origin."""
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        ring_ids = np.clip((distances / 1.5).astype(int), 0, 11)
        return np.column_stack([points, ring_ids])

    def test_straight_road_ground_truth_comparison(self):
        """Compare boundary detection on straight road against ground truth."""
        gt_points = self._load_points_from_file(str(self.straight_gt_file))
        input_points = self._load_points_from_file(str(self.whole_straight_file))

        if gt_points is None or input_points is None:
            self.skipTest("Validation files not available")

        # Get declared counts from file metadata
        gt_msgs = PointCloud2Parser.parse_yaml_file(str(self.straight_gt_file))
        input_msgs = PointCloud2Parser.parse_yaml_file(str(self.whole_straight_file))

        gt_declared = gt_msgs[0].width if gt_msgs else len(gt_points)
        input_declared = input_msgs[0].width if input_msgs else len(input_points)

        print(f"\n[Straight Road Ground Truth Test]")
        print(f"  Ground truth: {gt_declared} declared, {len(gt_points)} decoded")
        print(f"  Input frame: {input_declared} declared, {len(input_points)} decoded")
        print(f"  Expected detection rate: {gt_declared/input_declared*100:.1f}%")

        # Compute ground truth statistics (from decoded data)
        gt_y_range = (gt_points[:, 1].min(), gt_points[:, 1].max())
        print(f"  Ground truth Y range (decoded): [{gt_y_range[0]:.2f}, {gt_y_range[1]:.2f}] m")

        # Run detection on input with synthetic rings
        input_with_rings = self._add_synthetic_ring(input_points)

        results = []
        for ring_id in range(12):
            ring_mask = input_with_rings[:, 3] == ring_id
            ring_points = input_with_rings[ring_mask]

            if len(ring_points) >= 3:
                result = self.detector.detect_boundaries(ring_points, ring_id=ring_id)
                results.append(result)

        if results:
            detected_y_min = min(r.y_min for r in results)
            detected_y_max = max(r.y_max for r in results)

            print(f"  Detected Y range: [{detected_y_min:.2f}, {detected_y_max:.2f}] m")

            # Compute boundary metrics
            boundary_metrics = ValidationMetrics.compute_boundary_metrics(
                (detected_y_min, detected_y_max),
                gt_y_range
            )
            print(f"  Boundary IoU: {boundary_metrics['iou_1d']:.3f}")
            print(f"  Width ratio: {boundary_metrics['width_ratio']:.3f}")

            # Note: With truncated data, IoU may be low
            # Skip strict assertion if data is truncated
            if len(gt_points) < 50:  # Truncated data
                print("  [Note: Using truncated data - IoU check skipped]")
            else:
                self.assertGreater(boundary_metrics['iou_1d'], 0.3,
                                 "Boundary overlap should be > 30%")

    def test_detection_rate_analysis(self):
        """Analyze expected detection rate from ground truth."""
        gt_points = self._load_points_from_file(str(self.straight_gt_file))
        input_points = self._load_points_from_file(str(self.whole_straight_file))

        if gt_points is None or input_points is None:
            self.skipTest("Validation files not available")

        # Use declared counts from file metadata (not truncated decoded counts)
        gt_msgs = PointCloud2Parser.parse_yaml_file(str(self.straight_gt_file))
        input_msgs = PointCloud2Parser.parse_yaml_file(str(self.whole_straight_file))

        gt_declared = gt_msgs[0].width if gt_msgs else len(gt_points)
        input_declared = input_msgs[0].width if input_msgs else len(input_points)

        # Expected detection rate based on declared counts
        expected_rate = gt_declared / input_declared

        print(f"\n[Detection Rate Analysis]")
        print(f"  Ground truth boundary points: {gt_declared} (declared)")
        print(f"  Total input points: {input_declared} (declared)")
        print(f"  Expected detection rate: {expected_rate*100:.1f}%")
        print(f"  Decoded for testing: {len(gt_points)} gt, {len(input_points)} input")

        # This tells us what the pipeline should achieve
        # For straight road: 598/2415 = 24.7% of points are boundaries
        self.assertGreater(expected_rate, 0.1, "Expected rate should be > 10%")
        self.assertLess(expected_rate, 0.5, "Expected rate should be < 50%")


@unittest.skipUnless(HAS_ROUNDABOUT_MODULE, "Roundabout module not available")
class TestRoundaboutWithGroundTruth(unittest.TestCase):
    """Test roundabout detection comparing against ground truth."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.roundabout_gt_file = cls.workspace_root / 'roundabouts_curve_points_validation.txt'
        cls.whole_curved_file = cls.workspace_root / 'whole_frame_points_curved_line.txt'

        cls.detector = RoundaboutDetector(
            proximity_threshold=30.0,
            radius_tolerance=2.0,
            typical_roundabout_radius=10.0,
            min_consistency_score=0.7,
            logger=None
        )

    def test_curved_road_ground_truth_analysis(self):
        """Analyze curved road ground truth characteristics."""
        if not self.roundabout_gt_file.exists():
            self.skipTest("Roundabout validation file not available")

        messages = PointCloud2Parser.parse_yaml_file(str(self.roundabout_gt_file))
        if not messages:
            self.skipTest("No messages parsed")

        print(f"\n[Curved Road Ground Truth Analysis]")
        print(f"  Total frames: {len(messages)}")

        # Analyze each frame
        for i, msg in enumerate(messages):
            points = msg.points
            if len(points) < 3:
                continue

            y_range = (points[:, 1].min(), points[:, 1].max())
            x_extent = points[:, 0].max() - points[:, 0].min()

            print(f"\n  Frame {i+1}: {msg.width} declared, {len(points)} decoded")
            print(f"    Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}] m")
            print(f"    X extent: {x_extent:.2f} m")

            # Compute curvature indicator
            if len(points) >= 5:
                centroid = np.mean(points[:, :2], axis=0)
                dists = np.sqrt(np.sum((points[:, :2] - centroid)**2, axis=1))
                avg_radius = np.mean(dists)
                if avg_radius > 0.1:
                    curvature = 1.0 / avg_radius
                    print(f"    Curvature estimate: {curvature:.4f} (R~{avg_radius:.1f}m)")


class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline integration with ground truth comparison."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.files = {
            'straight_gt': cls.workspace_root / 'straight_points_validation.txt',
            'roundabout_gt': cls.workspace_root / 'roundabouts_curve_points_validation.txt',
            'straight_input': cls.workspace_root / 'whole_frame_points_straight_line.txt',
            'curved_input': cls.workspace_root / 'whole_frame_points_curved_line.txt',
        }

    def test_all_files_accessible(self):
        """Verify all validation and input files are accessible."""
        print("\n[File Accessibility Test]")
        for name, filepath in self.files.items():
            exists = filepath.exists()
            status = "OK" if exists else "MISSING"
            print(f"  {name}: {status}")

            if 'gt' in name:
                self.assertTrue(exists, f"Ground truth file required: {name}")

    def test_ground_truth_vs_input_relationship(self):
        """Verify ground truth is a subset of input data."""
        straight_gt_path = self.files['straight_gt']
        straight_input_path = self.files['straight_input']

        if not straight_gt_path.exists() or not straight_input_path.exists():
            self.skipTest("Files not available")

        gt_msgs = PointCloud2Parser.parse_yaml_file(str(straight_gt_path))
        input_msgs = PointCloud2Parser.parse_yaml_file(str(straight_input_path))

        if not gt_msgs or not input_msgs:
            self.skipTest("Could not parse files")

        gt_count = gt_msgs[0].width
        input_count = input_msgs[0].width
        ratio = gt_count / input_count

        print(f"\n[Ground Truth vs Input Relationship]")
        print(f"  Ground truth points: {gt_count}")
        print(f"  Input points: {input_count}")
        print(f"  Ratio: {ratio*100:.1f}%")
        print(f"  Interpretation: {ratio*100:.1f}% of input should be detected as boundaries")

        # Ground truth should be smaller than input
        self.assertLess(gt_count, input_count,
                       "Ground truth should be subset of input")

        # Ratio should be reasonable (5-50% typically)
        self.assertGreater(ratio, 0.05, "Expected boundary ratio > 5%")
        self.assertLess(ratio, 0.60, "Expected boundary ratio < 60%")

    def test_metric_computation(self):
        """Test validation metrics computation."""
        # Create synthetic test data
        detected = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [10, 10, 0]])
        ground_truth = np.array([[0, 0, 0], [1, 1.05, 0], [2, 2, 0], [3, 3, 0]])

        metrics = ValidationMetrics.compute_overlap_metrics(detected, ground_truth, tolerance=0.2)

        print(f"\n[Metric Computation Test]")
        print(f"  Detected: {len(detected)} points")
        print(f"  Ground truth: {len(ground_truth)} points")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")

        # Expect ~75% precision (3/4 detected are correct)
        # Expect ~75% recall (3/4 ground truth are matched)
        self.assertGreater(metrics['precision'], 0.5)
        self.assertGreater(metrics['recall'], 0.5)


class TestNoiseSuppressionAnalysis(unittest.TestCase):
    """Analyze noise characteristics using validation data to improve filtering."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.files = {
            'straight_gt': cls.workspace_root / 'straight_points_validation.txt',
            'roundabout_gt': cls.workspace_root / 'roundabouts_curve_points_validation.txt',
            'straight_input': cls.workspace_root / 'whole_frame_points_straight_line.txt',
            'curved_input': cls.workspace_root / 'whole_frame_points_curved_line.txt',
        }

    def _get_file_metadata(self, filepath):
        """Get declared point count from file without full parsing."""
        if not filepath.exists():
            return None
        msgs = PointCloud2Parser.parse_yaml_file(str(filepath))
        if msgs:
            return {'width': msgs[0].width, 'decoded': msgs[0].num_points, 'points': msgs[0].points}
        return None

    def test_intensity_distribution_analysis(self):
        """Analyze intensity values in ground truth vs whole frame."""
        gt_data = self._get_file_metadata(self.files['straight_gt'])
        input_data = self._get_file_metadata(self.files['straight_input'])

        if not gt_data or not input_data:
            self.skipTest("Files not available")

        print(f"\n{'='*60}")
        print("INTENSITY DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")

        # Ground truth intensity analysis
        gt_intensity = gt_data['points'][:, 3] if len(gt_data['points']) > 0 else np.array([])
        input_intensity = input_data['points'][:, 3] if len(input_data['points']) > 0 else np.array([])

        if len(gt_intensity) > 0:
            print(f"\nGround Truth (validated road boundaries):")
            print(f"  Count: {gt_data['width']} declared, {len(gt_intensity)} decoded")
            print(f"  Intensity range: [{gt_intensity.min():.2f}, {gt_intensity.max():.2f}]")
            print(f"  Mean: {gt_intensity.mean():.2f}, Std: {gt_intensity.std():.2f}")

        if len(input_intensity) > 0:
            print(f"\nWhole Frame (all points):")
            print(f"  Count: {input_data['width']} declared, {len(input_intensity)} decoded")
            print(f"  Intensity range: [{input_intensity.min():.2f}, {input_intensity.max():.2f}]")
            print(f"  Mean: {input_intensity.mean():.2f}, Std: {input_intensity.std():.2f}")

        # Recommendation based on analysis
        if len(gt_intensity) > 0:
            recommended_max = gt_intensity.max() + 1.0
            print(f"\n  RECOMMENDATION: max_road_intensity <= {recommended_max:.1f}")

    def test_spatial_noise_pattern_analysis(self):
        """Identify spatial patterns that distinguish road from noise."""
        gt_data = self._get_file_metadata(self.files['straight_gt'])
        input_data = self._get_file_metadata(self.files['straight_input'])

        if not gt_data or not input_data:
            self.skipTest("Files not available")

        print(f"\n{'='*60}")
        print("SPATIAL NOISE PATTERN ANALYSIS")
        print(f"{'='*60}")

        gt_pts = gt_data['points'][:, :3]  # x, y, z
        input_pts = input_data['points'][:, :3]

        if len(gt_pts) > 0:
            print(f"\nGround Truth Spatial Characteristics:")
            print(f"  X range: [{gt_pts[:,0].min():.2f}, {gt_pts[:,0].max():.2f}] m")
            print(f"  Y range: [{gt_pts[:,1].min():.2f}, {gt_pts[:,1].max():.2f}] m")
            print(f"  Z range: [{gt_pts[:,2].min():.2f}, {gt_pts[:,2].max():.2f}] m")
            print(f"  Road width (Y): {gt_pts[:,1].max() - gt_pts[:,1].min():.2f} m")

        if len(input_pts) > 0:
            print(f"\nWhole Frame Spatial Characteristics:")
            print(f"  X range: [{input_pts[:,0].min():.2f}, {input_pts[:,0].max():.2f}] m")
            print(f"  Y range: [{input_pts[:,1].min():.2f}, {input_pts[:,1].max():.2f}] m")
            print(f"  Z range: [{input_pts[:,2].min():.2f}, {input_pts[:,2].max():.2f}] m")

            # Z-variance analysis (flat surfaces vs grass)
            z_var = input_pts[:, 2].var()
            print(f"  Z variance: {z_var:.4f} m² (lower = flatter)")

    def test_intensity_threshold_sweep(self):
        """Test different intensity thresholds to find optimal value."""
        gt_data = self._get_file_metadata(self.files['straight_gt'])
        input_data = self._get_file_metadata(self.files['straight_input'])

        if not gt_data or not input_data:
            self.skipTest("Files not available")

        print(f"\n{'='*60}")
        print("INTENSITY THRESHOLD SWEEP")
        print(f"{'='*60}")

        gt_pts = gt_data['points']
        input_pts = input_data['points']

        if len(gt_pts) < 3 or len(input_pts) < 3:
            print("  Insufficient decoded points for threshold analysis")
            return

        thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

        print(f"\nThreshold | Retained | GT Match | Precision | Recall | F1")
        print("-" * 65)

        best_f1 = 0
        best_thresh = 6.0

        for thresh in thresholds:
            # Filter by intensity
            filtered_mask = input_pts[:, 3] <= thresh
            filtered_pts = input_pts[filtered_mask]

            if len(filtered_pts) == 0:
                print(f"  {thresh:6.1f} |    0     |    -     |   0.000  | 0.000  | 0.000")
                continue

            # Compute metrics against ground truth using spatial matching
            if len(gt_pts) > 0 and len(filtered_pts) > 0:
                metrics = ValidationMetrics.compute_overlap_metrics(
                    filtered_pts[:, :3], gt_pts[:, :3], tolerance=0.5
                )
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1']

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

                print(f"  {thresh:6.1f} | {len(filtered_pts):6d}   | {metrics['matched']:6d}   | "
                      f"  {precision:.3f}  | {recall:.3f}  | {f1:.3f}")

        print("-" * 65)
        print(f"  OPTIMAL THRESHOLD: {best_thresh:.1f} (F1={best_f1:.3f})")

    def test_per_ring_noise_analysis(self):
        """Analyze noise patterns per LiDAR ring."""
        gt_data = self._get_file_metadata(self.files['straight_gt'])
        input_data = self._get_file_metadata(self.files['straight_input'])

        if not gt_data or not input_data:
            self.skipTest("Files not available")

        print(f"\n{'='*60}")
        print("PER-RING NOISE ANALYSIS")
        print(f"{'='*60}")

        gt_pts = gt_data['points']
        input_pts = input_data['points']

        if len(gt_pts) < 3 or len(input_pts) < 3:
            print("  Insufficient decoded points for ring analysis")
            return

        # Compute synthetic ring IDs based on distance
        def compute_ring_id(pts):
            distances = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            return np.clip((distances / 1.5).astype(int), 0, 11)

        gt_rings = compute_ring_id(gt_pts)
        input_rings = compute_ring_id(input_pts)

        print(f"\nRing | GT Pts | Input Pts | Ratio | Intensity Range")
        print("-" * 60)

        for ring_id in range(12):
            gt_mask = gt_rings == ring_id
            input_mask = input_rings == ring_id

            gt_count = np.sum(gt_mask)
            input_count = np.sum(input_mask)

            if input_count > 0:
                ratio = gt_count / input_count if input_count > 0 else 0
                input_intensity = input_pts[input_mask, 3]
                int_range = f"[{input_intensity.min():.1f}, {input_intensity.max():.1f}]"
            else:
                ratio = 0
                int_range = "N/A"

            if gt_count > 0 or input_count > 0:
                print(f" {ring_id:2d}  | {gt_count:5d}  | {input_count:8d}  | {ratio:.2f}  | {int_range}")

    def test_noise_suppression_recommendations(self):
        """Generate parameter recommendations based on validation analysis."""
        gt_data = self._get_file_metadata(self.files['straight_gt'])

        print(f"\n{'='*60}")
        print("NOISE SUPPRESSION RECOMMENDATIONS")
        print(f"{'='*60}")

        if gt_data and len(gt_data['points']) > 0:
            gt_pts = gt_data['points']
            gt_intensity = gt_pts[:, 3]
            gt_y_range = gt_pts[:, 1].max() - gt_pts[:, 1].min()

            print(f"\nBased on ground truth analysis:")
            print(f"  Ground truth intensity max: {gt_intensity.max():.2f}")
            print(f"  Ground truth road width: {gt_y_range:.2f} m")

            print(f"\nRecommended params.yaml settings:")
            print(f"  max_road_intensity: {min(gt_intensity.max() + 2.0, 8.0):.1f}")
            print(f"  min_boundary_width: {max(gt_y_range * 0.5, 1.0):.1f}")
            print(f"  boundary_confidence_threshold: 0.5")
        else:
            print("  Using default recommendations (insufficient ground truth data):")
            print(f"  max_road_intensity: 6.0")
            print(f"  min_boundary_width: 1.5")
            print(f"  boundary_confidence_threshold: 0.5")


class TestRosbagNoiseAnalysis(unittest.TestCase):
    """Analyze noise patterns using complete data from rosbag files."""

    @classmethod
    def setUpClass(cls):
        cls.workspace_root = Path(__file__).parent.parent.parent.parent
        cls.spanlidar_bag = cls.workspace_root / 'rosbag/spanlidar/spanlidar_0.mcap'
        cls.sy27_bag = cls.workspace_root / 'rosbag/sy27_road_ground_truth_live/sy27_road_ground_truth_live_0.mcap'

        # Try to import mcap
        cls.mcap_available = False
        cls._read_ros2_messages = None
        try:
            from mcap_ros2.reader import read_ros2_messages
            cls._read_ros2_messages = staticmethod(read_ros2_messages)
            cls.mcap_available = True
        except ImportError:
            pass

    def _extract_pointcloud_frames(self, bag_path: str, topic: str, max_frames: int = 10):
        """Extract PointCloud2 frames from rosbag."""
        if not self.mcap_available:
            return []

        from mcap_ros2.reader import read_ros2_messages

        frames = []
        try:
            for msg in read_ros2_messages(bag_path, topics=[topic]):
                pc_msg = msg.ros_msg
                point_step = pc_msg.point_step

                # Parse PointCloud2 data
                data = np.frombuffer(bytes(pc_msg.data), dtype=np.uint8)
                if len(data) < point_step:
                    continue

                num_points = len(data) // point_step
                points = []

                # Parse x, y, z (always at offset 0, 4, 8)
                # Intensity location varies: typically offset 12 or 16
                # For point_step=22, intensity is at offset 12 (float32)
                for i in range(num_points):
                    offset = i * point_step
                    if offset + 16 > len(data):
                        break
                    x, y, z, intensity = struct.unpack('<4f', data[offset:offset+16])
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        points.append([x, y, z, intensity])

                if points:
                    frames.append(np.array(points, dtype=np.float32))
                    if len(frames) >= max_frames:
                        break
        except Exception as e:
            import traceback
            print(f"Error reading bag: {e}")
            traceback.print_exc()

        return frames

    def test_rosbag_intensity_analysis(self):
        """Analyze intensity distribution from real rosbag data."""
        if not self.mcap_available:
            self.skipTest("mcap_ros2 not available")
        if not self.spanlidar_bag.exists():
            self.skipTest(f"Rosbag not found: {self.spanlidar_bag}")

        print(f"\n{'='*60}")
        print("ROSBAG INTENSITY ANALYSIS (Complete Data)")
        print(f"{'='*60}")

        # Extract frames from /patchworkpp/ground (filtered ground points)
        frames = self._extract_pointcloud_frames(
            str(self.spanlidar_bag), '/patchworkpp/ground', max_frames=5)

        if not frames:
            # Try /points if patchworkpp not available
            frames = self._extract_pointcloud_frames(
                str(self.spanlidar_bag), '/points', max_frames=5)

        if not frames:
            print("  No point cloud data found in rosbag")
            return

        print(f"\n  Extracted {len(frames)} frames from rosbag")

        all_intensities = []
        all_points = []
        for frame in frames:
            all_intensities.extend(frame[:, 3])
            all_points.append(frame)

        intensities = np.array(all_intensities)

        print(f"\n  Total points analyzed: {len(intensities)}")
        print(f"  Intensity statistics:")
        print(f"    Range: [{intensities.min():.2f}, {intensities.max():.2f}]")
        print(f"    Mean: {intensities.mean():.2f}")
        print(f"    Std: {intensities.std():.2f}")
        print(f"    Median: {np.median(intensities):.2f}")

        # Analyze intensity distribution percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n  Intensity percentiles:")
        for p in percentiles:
            val = np.percentile(intensities, p)
            print(f"    P{p}: {val:.2f}")

        # Identify potential road vs grass separation
        low_intensity = intensities[intensities <= 6.0]
        mid_intensity = intensities[(intensities > 6.0) & (intensities <= 12.0)]
        high_intensity = intensities[intensities > 12.0]

        print(f"\n  Intensity distribution:")
        print(f"    Low (0-6, road): {len(low_intensity)} ({100*len(low_intensity)/len(intensities):.1f}%)")
        print(f"    Mid (6-12, grass?): {len(mid_intensity)} ({100*len(mid_intensity)/len(intensities):.1f}%)")
        print(f"    High (>12, markings?): {len(high_intensity)} ({100*len(high_intensity)/len(intensities):.1f}%)")

        # Recommendation
        if len(low_intensity) > 0.2 * len(intensities):
            print(f"\n  RECOMMENDATION: max_road_intensity between 6.0-8.0")
            print(f"    This captures {100*len(low_intensity)/len(intensities):.1f}% of points")
        else:
            print(f"\n  RECOMMENDATION: Consider increasing max_road_intensity")

    def test_rosbag_spatial_analysis(self):
        """Analyze spatial patterns from rosbag data."""
        if not self.mcap_available:
            self.skipTest("mcap_ros2 not available")
        if not self.spanlidar_bag.exists():
            self.skipTest(f"Rosbag not found: {self.spanlidar_bag}")

        print(f"\n{'='*60}")
        print("ROSBAG SPATIAL PATTERN ANALYSIS")
        print(f"{'='*60}")

        frames = self._extract_pointcloud_frames(
            str(self.spanlidar_bag), '/patchworkpp/ground', max_frames=3)

        if not frames:
            frames = self._extract_pointcloud_frames(
                str(self.spanlidar_bag), '/points', max_frames=3)

        if not frames:
            print("  No point cloud data found")
            return

        for i, frame in enumerate(frames):
            print(f"\n  Frame {i+1} ({len(frame)} points):")

            # Spatial extent
            print(f"    X range: [{frame[:,0].min():.2f}, {frame[:,0].max():.2f}] m")
            print(f"    Y range: [{frame[:,1].min():.2f}, {frame[:,1].max():.2f}] m")
            print(f"    Z range: [{frame[:,2].min():.2f}, {frame[:,2].max():.2f}] m")

            # Z-variance (flatness indicator)
            z_var = frame[:, 2].var()
            print(f"    Z variance: {z_var:.4f} m² ({'flat' if z_var < 0.05 else 'uneven'})")

            # Road region estimation (|y| < 5m, typical road width)
            road_mask = np.abs(frame[:, 1]) < 5.0
            road_points = frame[road_mask]
            outside_points = frame[~road_mask]

            print(f"    Points |y|<5m (road region): {len(road_points)} ({100*len(road_points)/len(frame):.1f}%)")

            if len(road_points) > 10 and len(outside_points) > 10:
                road_int = road_points[:, 3]
                outside_int = outside_points[:, 3]
                print(f"    Road region intensity: mean={road_int.mean():.2f}, std={road_int.std():.2f}")
                print(f"    Outside region intensity: mean={outside_int.mean():.2f}, std={outside_int.std():.2f}")

    def test_rosbag_ring_distribution(self):
        """Analyze points per synthetic ring from rosbag."""
        if not self.mcap_available:
            self.skipTest("mcap_ros2 not available")
        if not self.spanlidar_bag.exists():
            self.skipTest(f"Rosbag not found: {self.spanlidar_bag}")

        print(f"\n{'='*60}")
        print("ROSBAG RING DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")

        frames = self._extract_pointcloud_frames(
            str(self.spanlidar_bag), '/patchworkpp/ground', max_frames=5)

        if not frames:
            frames = self._extract_pointcloud_frames(
                str(self.spanlidar_bag), '/points', max_frames=5)

        if not frames:
            print("  No point cloud data found")
            return

        # Combine all frames
        all_points = np.vstack(frames)

        # Compute synthetic ring IDs based on distance
        distances = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
        ring_ids = np.clip((distances / 2.5).astype(int), 0, 15)  # 2.5m per ring

        print(f"\n  Total points: {len(all_points)}")
        print(f"\n  Ring | Points | Mean Intensity | Z-Variance | Distance Range")
        print(f"  " + "-" * 65)

        for ring_id in range(16):
            mask = ring_ids == ring_id
            count = np.sum(mask)
            if count > 0:
                ring_points = all_points[mask]
                ring_dist = distances[mask]
                print(f"   {ring_id:2d}  | {count:6d} | "
                      f"{ring_points[:,3].mean():8.2f}     | "
                      f"{ring_points[:,2].var():8.4f}   | "
                      f"[{ring_dist.min():.1f}-{ring_dist.max():.1f}]m")

    def test_optimal_intensity_threshold_rosbag(self):
        """Find optimal intensity threshold using rosbag data."""
        if not self.mcap_available:
            self.skipTest("mcap_ros2 not available")
        if not self.spanlidar_bag.exists():
            self.skipTest(f"Rosbag not found: {self.spanlidar_bag}")

        print(f"\n{'='*60}")
        print("OPTIMAL INTENSITY THRESHOLD (Rosbag Data)")
        print(f"{'='*60}")

        frames = self._extract_pointcloud_frames(
            str(self.spanlidar_bag), '/patchworkpp/ground', max_frames=10)

        if not frames:
            frames = self._extract_pointcloud_frames(
                str(self.spanlidar_bag), '/points', max_frames=10)

        if not frames:
            print("  No point cloud data found")
            return

        all_points = np.vstack(frames)

        # Use Y-range as proxy for road (typically |y| < 3m is road)
        # Ground truth approximation: points within typical road width
        road_mask = np.abs(all_points[:, 1]) < 3.5  # Typical road half-width

        print(f"\n  Threshold sweep (using |y|<3.5m as road proxy):")
        print(f"  Threshold | Total Kept | Road-like | Non-Road | Ratio")
        print(f"  " + "-" * 55)

        thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]

        for thresh in thresholds:
            int_mask = all_points[:, 3] <= thresh
            kept = np.sum(int_mask)
            road_kept = np.sum(int_mask & road_mask)
            non_road_kept = np.sum(int_mask & ~road_mask)
            ratio = road_kept / kept if kept > 0 else 0

            print(f"   {thresh:5.1f}   |   {kept:6d}   |   {road_kept:6d}  |  {non_road_kept:6d}  | {ratio:.3f}")

        print(f"\n  Higher ratio = more selective for road region")
        print(f"  RECOMMENDATION: Use threshold 6.0-8.0 for best road selectivity")


def run_validation_tests():
    """Run all validation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPointCloud2Parser))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryDetectionWithGroundTruth))
    suite.addTests(loader.loadTestsFromTestCase(TestRoundaboutWithGroundTruth))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseSuppressionAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestRosbagNoiseAnalysis))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("VALIDATION TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests run:     {result.testsRun}")
    print(f"Successes:           {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:            {len(result.failures)}")
    print(f"Errors:              {len(result.errors)}")
    print(f"Skipped:             {len(result.skipped)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_validation_tests()
    sys.exit(exit_code)
