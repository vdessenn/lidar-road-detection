#!/usr/bin/env python3
"""
Boundary Extraction Tuner
=========================
Tests different boundary extraction parameters on labeled validation data.

Usage:
    python3 boundary_extraction_tuner.py --latest
    python3 boundary_extraction_tuner.py tools/validation_output/session_XXXX/

Parameters tested:
    - max_boundary_width: Maximum width between left/right boundaries (meters)
    - min_points_per_ring: Minimum points required to process a ring
    - y_threshold: Minimum |y| to be considered a boundary point
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config, HISTORICAL_DEFAULTS


# ===========================
# Data Structures
# ===========================

@dataclass
class BoundaryParams:
    """Parameters for boundary extraction."""
    name: str
    max_boundary_width: float = 5.0
    min_points_per_ring: int = 50
    y_threshold: float = 0.2
    # New: Robust extraction
    robust_n_points: int = 3
    # New: Inter-ring consistency
    enable_ring_consistency: bool = True
    max_y_deviation: float = 0.5

    def __str__(self):
        return f"{self.name} (w≤{self.max_boundary_width}m, n≥{self.min_points_per_ring}, y>{self.y_threshold}, robust={self.robust_n_points})"


@dataclass
class FrameResult:
    """Per-frame test results."""
    frame_file: str
    params_name: str
    label: str
    n_left: int
    n_right: int
    left_y_mean: float
    right_y_mean: float
    width_mean: float
    symmetry_score: float  # How symmetric are left/right boundaries
    processing_time_ms: float


@dataclass
class ParamsResult:
    """Aggregate results per parameter set."""
    params: BoundaryParams
    frames_processed: int
    good_frames: int
    bad_frames: int
    # On good frames
    avg_left_count: float
    avg_right_count: float
    avg_width: float
    avg_symmetry: float
    # On bad frames
    bad_left_count: float
    bad_right_count: float
    # Overall
    avg_latency_ms: float
    frame_results: List[FrameResult] = field(default_factory=list)

    def to_dict(self):
        return {
            'params_name': self.params.name,
            'max_boundary_width': self.params.max_boundary_width,
            'min_points_per_ring': self.params.min_points_per_ring,
            'y_threshold': self.params.y_threshold,
            'robust_n_points': self.params.robust_n_points,
            'enable_ring_consistency': self.params.enable_ring_consistency,
            'max_y_deviation': self.params.max_y_deviation,
            'frames_processed': self.frames_processed,
            'good_frames': self.good_frames,
            'bad_frames': self.bad_frames,
            'avg_left_count': round(self.avg_left_count, 2),
            'avg_right_count': round(self.avg_right_count, 2),
            'avg_width': round(self.avg_width, 2),
            'avg_symmetry': round(self.avg_symmetry, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 2)
        }


# ===========================
# Enhanced Boundary Extractor (matches Node 02 v2)
# ===========================

class BoundaryExtractor:
    """Standalone boundary extraction for testing parameters (matches 02_ring_boundary_extraction.py v2)."""

    def __init__(self, params: BoundaryParams):
        self.params = params

    def extract(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                ring: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left/right boundaries from road points.
        
        Returns:
            (left_points, right_points) as Nx3 arrays
        """
        if ring is not None:
            return self._extract_from_rings(x, y, z, ring)
        else:
            return self._extract_from_x_bins(x, y, z)

    def _extract_from_rings(self, x: np.ndarray, y: np.ndarray,
                            z: np.ndarray, ring: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract boundaries with robust extraction and consistency check."""
        from typing import Dict
        
        # Step 1: Extract boundaries per ring
        boundaries_by_ring: Dict[int, Tuple[Tuple, Tuple]] = {}

        for ring_id in np.unique(ring):
            mask = ring == ring_id
            if np.sum(mask) < self.params.min_points_per_ring:
                continue

            x_r, y_r, z_r = x[mask], y[mask], z[mask]

            # Robust extraction: median of N extreme points
            left_pt = self._get_robust_extreme(x_r, y_r, z_r, side='left')
            right_pt = self._get_robust_extreme(x_r, y_r, z_r, side='right')

            if left_pt is None or right_pt is None:
                continue

            # Symmetry constraint
            width = left_pt[1] - right_pt[1]
            if width > self.params.max_boundary_width:
                if abs(right_pt[1]) < abs(left_pt[1]):
                    left_pt = (left_pt[0], -right_pt[1], left_pt[2])
                else:
                    right_pt = (right_pt[0], -left_pt[1], right_pt[2])

            boundaries_by_ring[int(ring_id)] = (left_pt, right_pt)

        # Step 2: Inter-ring consistency check
        if self.params.enable_ring_consistency:
            boundaries_by_ring = self._validate_ring_consistency(boundaries_by_ring)

        # Step 3: Build output arrays
        left_points = []
        right_points = []

        for ring_id, (left_pt, right_pt) in boundaries_by_ring.items():
            if left_pt[1] > self.params.y_threshold:
                left_points.append(list(left_pt))
            if right_pt[1] < -self.params.y_threshold:
                right_points.append(list(right_pt))

        left = np.array(left_points) if left_points else np.empty((0, 3))
        right = np.array(right_points) if right_points else np.empty((0, 3))
        return left, right

    def _get_robust_extreme(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            side: str) -> Optional[Tuple[float, float, float]]:
        """Get median of N extreme points instead of single extreme."""
        n = min(self.params.robust_n_points, len(y))
        if n == 0:
            return None

        if side == 'left':
            indices = np.argsort(y)[-n:]  # Highest y values
        else:
            indices = np.argsort(y)[:n]   # Lowest y values

        return (
            float(np.median(x[indices])),
            float(np.median(y[indices])),
            float(np.median(z[indices]))
        )

    def _validate_ring_consistency(self, boundaries: dict) -> dict:
        """Remove rings with boundaries too different from neighbors."""
        if len(boundaries) < 2:
            return boundaries

        validated = {}
        ring_ids = sorted(boundaries.keys())

        for i, ring_id in enumerate(ring_ids):
            left_pt, right_pt = boundaries[ring_id]
            is_valid = True

            # Check previous ring
            if i > 0:
                prev_id = ring_ids[i - 1]
                prev_left, prev_right = boundaries[prev_id]
                if abs(left_pt[1] - prev_left[1]) > self.params.max_y_deviation:
                    is_valid = False
                if abs(right_pt[1] - prev_right[1]) > self.params.max_y_deviation:
                    is_valid = False

            # Check next ring
            if i < len(ring_ids) - 1:
                next_id = ring_ids[i + 1]
                next_left, next_right = boundaries[next_id]
                if abs(left_pt[1] - next_left[1]) > self.params.max_y_deviation:
                    is_valid = False
                if abs(right_pt[1] - next_right[1]) > self.params.max_y_deviation:
                    is_valid = False

            if is_valid:
                validated[ring_id] = (left_pt, right_pt)

        # Fallback to all if none valid
        return validated if validated else boundaries

    def _extract_from_x_bins(self, x: np.ndarray, y: np.ndarray,
                             z: np.ndarray, bin_size: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: X-binning if no ring field (also uses robust extraction)."""
        left_points = []
        right_points = []

        if len(x) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        x_bins = np.arange(x.min(), x.max() + bin_size, bin_size)
        digitized = np.digitize(x, x_bins)

        for bin_idx in range(1, len(x_bins)):
            mask = digitized == bin_idx
            if np.sum(mask) < self.params.min_points_per_ring:
                continue

            x_bin, y_bin, z_bin = x[mask], y[mask], z[mask]

            # Use robust extraction
            left_pt = self._get_robust_extreme(x_bin, y_bin, z_bin, 'left')
            right_pt = self._get_robust_extreme(x_bin, y_bin, z_bin, 'right')

            if left_pt and left_pt[1] > self.params.y_threshold:
                left_points.append(list(left_pt))
            if right_pt and right_pt[1] < -self.params.y_threshold:
                right_points.append(list(right_pt))

        left = np.array(left_points) if left_points else np.empty((0, 3))
        right = np.array(right_points) if right_points else np.empty((0, 3))
        return left, right


# ===========================
# NPZ Data Loader
# ===========================

class ValidationDataLoader:
    """Loads labeled validation data from NPZ files."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.good_dir = self.session_dir / 'good'
        self.bad_dir = self.session_dir / 'bad'

    def load_frames(self) -> List[Tuple[Path, str, dict]]:
        """Load all labeled frames."""
        frames = []
        
        if self.good_dir.exists():
            for npz_file in sorted(self.good_dir.glob('*.npz')):
                data = dict(np.load(npz_file))
                frames.append((npz_file, 'good', data))
        
        if self.bad_dir.exists():
            for npz_file in sorted(self.bad_dir.glob('*.npz')):
                data = dict(np.load(npz_file))
                frames.append((npz_file, 'bad', data))
        
        return frames

    def get_statistics(self) -> dict:
        """Get statistics about the validation dataset."""
        good_files = list(self.good_dir.glob('*.npz')) if self.good_dir.exists() else []
        bad_files = list(self.bad_dir.glob('*.npz')) if self.bad_dir.exists() else []
        
        return {
            'good_frames': len(good_files),
            'bad_frames': len(bad_files),
            'total_frames': len(good_files) + len(bad_files)
        }


# ===========================
# Boundary Tester
# ===========================

class BoundaryTester:
    """Tests boundary parameters on validation data."""

    def __init__(self, session_dir: Path):
        self.loader = ValidationDataLoader(session_dir)
        self.frames = self.loader.load_frames()

    def test_params(self, params: BoundaryParams) -> ParamsResult:
        """Test a set of boundary parameters on all frames."""
        extractor = BoundaryExtractor(params)
        frame_results = []
        
        for filepath, label, data in self.frames:
            # Use road points as input (simulating what Node 02 receives)
            road_x = data.get('road_x', np.array([]))
            road_y = data.get('road_y', np.array([]))
            road_z = data.get('road_z', np.array([]))
            road_ring = data.get('road_ring', None)
            
            n_road = len(road_x)
            if n_road == 0:
                continue
            
            # Time the extraction
            start_time = time.perf_counter()
            left, right = extractor.extract(road_x, road_y, road_z, road_ring)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            n_left = len(left)
            n_right = len(right)
            
            # Calculate metrics
            left_y_mean = float(np.mean(left[:, 1])) if n_left > 0 else 0.0
            right_y_mean = float(np.mean(right[:, 1])) if n_right > 0 else 0.0
            
            # Average width (mean of left_y - right_y per matching x)
            if n_left > 0 and n_right > 0:
                width_mean = left_y_mean - right_y_mean
            else:
                width_mean = 0.0
            
            # Symmetry score: how close are |left_y| and |right_y|
            if n_left > 0 and n_right > 0:
                symmetry = 1.0 - abs(abs(left_y_mean) - abs(right_y_mean)) / max(abs(left_y_mean), abs(right_y_mean), 1.0)
            else:
                symmetry = 0.0
            
            frame_results.append(FrameResult(
                frame_file=str(filepath.name),
                params_name=params.name,
                label=label,
                n_left=n_left,
                n_right=n_right,
                left_y_mean=left_y_mean,
                right_y_mean=right_y_mean,
                width_mean=width_mean,
                symmetry_score=symmetry,
                processing_time_ms=elapsed_ms
            ))
        
        # Aggregate results
        good_results = [r for r in frame_results if r.label == 'good']
        bad_results = [r for r in frame_results if r.label == 'bad']
        
        return ParamsResult(
            params=params,
            frames_processed=len(frame_results),
            good_frames=len(good_results),
            bad_frames=len(bad_results),
            avg_left_count=np.mean([r.n_left for r in good_results]) if good_results else 0.0,
            avg_right_count=np.mean([r.n_right for r in good_results]) if good_results else 0.0,
            avg_width=np.mean([r.width_mean for r in good_results]) if good_results else 0.0,
            avg_symmetry=np.mean([r.symmetry_score for r in good_results]) if good_results else 0.0,
            bad_left_count=np.mean([r.n_left for r in bad_results]) if bad_results else 0.0,
            bad_right_count=np.mean([r.n_right for r in bad_results]) if bad_results else 0.0,
            avg_latency_ms=np.mean([r.processing_time_ms for r in frame_results]) if frame_results else 0.0,
            frame_results=frame_results
        )


# ===========================
# Parameter Grid
# ===========================

def generate_param_grid() -> List[BoundaryParams]:
    """Generate parameter combinations to test."""
    params_list = []

    # Load current baseline from params.yaml
    config = get_config()
    ros_params = config.get_node_params('ring_boundary_node')

    # Current params.yaml values
    cur_width = ros_params.get('max_boundary_width', 5.0)
    cur_min_pts = ros_params.get('min_points_per_ring', 50)
    cur_y_thresh = ros_params.get('y_threshold', 0.2)
    cur_robust_n = ros_params.get('robust_n_points', 3)
    cur_consistency = ros_params.get('enable_ring_consistency', True)
    cur_deviation = ros_params.get('max_y_deviation', 0.5)

    print(f"  Loaded baseline from params.yaml:")
    print(f"    max_boundary_width={cur_width}, y_threshold={cur_y_thresh}")
    print(f"    min_points_per_ring={cur_min_pts}, robust_n_points={cur_robust_n}")

    # Current baseline from params.yaml
    params_list.append(BoundaryParams(
        name="baseline_current",
        max_boundary_width=cur_width,
        min_points_per_ring=cur_min_pts,
        y_threshold=cur_y_thresh,
        robust_n_points=cur_robust_n,
        enable_ring_consistency=cur_consistency,
        max_y_deviation=cur_deviation
    ))

    # Historical baseline for comparison (original values before tuning)
    hist = HISTORICAL_DEFAULTS['ring_boundary_node']
    params_list.append(BoundaryParams(
        name="baseline_historical",
        max_boundary_width=hist['max_boundary_width'],
        min_points_per_ring=50,  # Historical default
        y_threshold=hist['y_threshold'],
        robust_n_points=3,
        enable_ring_consistency=True,
        max_y_deviation=0.5
    ))
    print(f"  Added historical baseline for comparison:")
    print(f"    max_boundary_width={hist['max_boundary_width']}, y_threshold={hist['y_threshold']}")

    # Test WITHOUT enhancements (old method)
    params_list.append(BoundaryParams(
        name="no_enhancements",
        max_boundary_width=cur_width,
        min_points_per_ring=cur_min_pts,
        y_threshold=cur_y_thresh,
        robust_n_points=1,  # Single point = old method
        enable_ring_consistency=False,
        max_y_deviation=999.0
    ))
    
    # Vary robust_n_points
    for n_pts in [1, 2, 5, 7]:
        params_list.append(BoundaryParams(
            name=f"robust_{n_pts}",
            max_boundary_width=cur_width,
            min_points_per_ring=cur_min_pts,
            y_threshold=cur_y_thresh,
            robust_n_points=n_pts,
            enable_ring_consistency=True,
            max_y_deviation=cur_deviation
        ))

    # Vary max_y_deviation (ring consistency strictness)
    for dev in [0.3, 0.7, 1.0, 1.5]:
        params_list.append(BoundaryParams(
            name=f"deviation_{dev}",
            max_boundary_width=cur_width,
            min_points_per_ring=cur_min_pts,
            y_threshold=cur_y_thresh,
            robust_n_points=cur_robust_n,
            enable_ring_consistency=True,
            max_y_deviation=dev
        ))

    # Vary max_boundary_width (around current baseline)
    for width in [3.0, 4.0, 6.0, 7.0]:
        params_list.append(BoundaryParams(
            name=f"width_{width}",
            max_boundary_width=width,
            min_points_per_ring=cur_min_pts,
            y_threshold=cur_y_thresh,
            robust_n_points=cur_robust_n,
            enable_ring_consistency=True,
            max_y_deviation=cur_deviation
        ))

    # Vary min_points_per_ring
    for min_pts in [20, 30, 70]:
        params_list.append(BoundaryParams(
            name=f"minpts_{min_pts}",
            max_boundary_width=cur_width,
            min_points_per_ring=min_pts,
            y_threshold=cur_y_thresh,
            robust_n_points=cur_robust_n,
            enable_ring_consistency=True,
            max_y_deviation=cur_deviation
        ))

    # Vary y_threshold (around current baseline)
    for y_thresh in [0.1, 0.3, 0.4, 0.5]:
        params_list.append(BoundaryParams(
            name=f"ythresh_{y_thresh}",
            max_boundary_width=cur_width,
            min_points_per_ring=cur_min_pts,
            y_threshold=y_thresh,
            robust_n_points=cur_robust_n,
            enable_ring_consistency=True,
            max_y_deviation=cur_deviation
        ))
    
    # Optimized combinations
    params_list.append(BoundaryParams(
        name="tight_robust",
        max_boundary_width=3.5,
        min_points_per_ring=30,
        y_threshold=0.3,
        robust_n_points=5,
        enable_ring_consistency=True,
        max_y_deviation=0.4
    ))
    
    params_list.append(BoundaryParams(
        name="relaxed",
        max_boundary_width=5.0,
        min_points_per_ring=20,
        y_threshold=0.2,
        robust_n_points=3,
        enable_ring_consistency=True,
        max_y_deviation=0.8
    ))
    
    return params_list


# ===========================
# Visualization
# ===========================

class VisualizationGenerator:
    """Generates comparative visualization plots."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_parameter_comparison(self, results: List[ParamsResult]):
        """Bar chart comparing parameter sets."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Boundary Extraction Parameters Comparison', fontsize=16, fontweight='bold')

        names = [r.params.name for r in results]
        x = np.arange(len(names))

        # Plot 1: Left/Right boundary counts
        ax = axes[0, 0]
        width = 0.35
        ax.bar(x - width/2, [r.avg_left_count for r in results], width, label='Left', color='blue', alpha=0.7)
        ax.bar(x + width/2, [r.avg_right_count for r in results], width, label='Right', color='red', alpha=0.7)
        ax.set_ylabel('Boundary Points')
        ax.set_title('Boundary Points per Frame (Good Frames)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Average road width
        ax = axes[0, 1]
        ax.bar(x, [r.avg_width for r in results], color='green', alpha=0.7)
        ax.set_ylabel('Width (m)')
        ax.set_title('Average Detected Road Width (Good Frames)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=3.75, color='orange', linestyle='--', label='Target 3.75m')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Symmetry score
        ax = axes[1, 0]
        ax.bar(x, [r.avg_symmetry for r in results], color='purple', alpha=0.7)
        ax.set_ylabel('Symmetry Score')
        ax.set_title('Boundary Symmetry (Higher = More Symmetric)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0.8, color='green', linestyle='--', label='Target 0.8')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 4: Processing latency
        ax = axes[1, 1]
        ax.bar(x, [r.avg_latency_ms for r in results], color='orange', alpha=0.7)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Processing Time (Lower = Faster)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=10, color='red', linestyle='--', label='Budget 10ms')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'boundary_parameter_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    def plot_boundary_distribution(self, frames: List[Tuple[Path, str, dict]]):
        """Compare boundary Y distributions between good and bad frames."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Road Points Y Distribution: Good vs Bad Frames', fontsize=16, fontweight='bold')
        
        # Good frames
        good_y = []
        for _, label, data in frames:
            if label == 'good':
                y = data.get('road_y', np.array([]))
                good_y.extend(y)
        
        if good_y:
            axes[0].hist(good_y, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0].axvline(0, color='red', linestyle='--', label='Center (y=0)')
            axes[0].set_title(f'GOOD Frames (n={len(good_y)} points)')
            axes[0].set_xlabel('Y (lateral position)')
            axes[0].set_ylabel('Count')
            axes[0].legend()
        
        # Bad frames
        bad_y = []
        for _, label, data in frames:
            if label == 'bad':
                y = data.get('road_y', np.array([]))
                bad_y.extend(y)
        
        if bad_y:
            axes[1].hist(bad_y, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1].axvline(0, color='blue', linestyle='--', label='Center (y=0)')
            axes[1].set_title(f'BAD Frames (n={len(bad_y)} points)')
            axes[1].set_xlabel('Y (lateral position)')
            axes[1].set_ylabel('Count')
            axes[1].legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'boundary_y_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")


# ===========================
# Recommendations Generator
# ===========================

def generate_recommendations(results: List[ParamsResult], output_dir: Path) -> str:
    """Generate recommendations based on test results."""
    
    # Score = symmetry * 0.4 + (boundary_count > 5) * 0.3 + (width in [3,5]) * 0.3
    scored = []
    for r in results:
        # Prefer params with:
        # - High symmetry
        # - Good number of boundary points (5-15 per side)
        # - Reasonable width (3-5m)
        
        sym_score = r.avg_symmetry
        count_score = 1.0 if 5 <= r.avg_left_count <= 15 and 5 <= r.avg_right_count <= 15 else 0.5
        width_score = 1.0 if 3.0 <= r.avg_width <= 5.0 else 0.5
        
        total_score = sym_score * 0.4 + count_score * 0.3 + width_score * 0.3
        scored.append((total_score, r))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    
    md = f"""# Boundary Extraction Parameter Recommendations

## Test Summary
- **Total frames tested**: {results[0].frames_processed if results else 0}
- **Good frames**: {results[0].good_frames if results else 0}
- **Bad frames**: {results[0].bad_frames if results else 0}

## Best Parameters: **{best.params.name}**

| Parameter | Value |
|-----------|-------|
| max_boundary_width | {best.params.max_boundary_width} m |
| min_points_per_ring | {best.params.min_points_per_ring} |
| y_threshold | {best.params.y_threshold} m |
| robust_n_points | {best.params.robust_n_points} |
| enable_ring_consistency | {best.params.enable_ring_consistency} |
| max_y_deviation | {best.params.max_y_deviation} m |

### Performance
- Left boundaries: **{best.avg_left_count:.1f}** points/frame
- Right boundaries: **{best.avg_right_count:.1f}** points/frame
- Average width: **{best.avg_width:.2f}** m
- Symmetry score: **{best.avg_symmetry:.3f}**
- Latency: **{best.avg_latency_ms:.2f}** ms

## All Results Ranked

| Rank | Name | Left | Right | Width | Symmetry | Latency |
|------|------|------|-------|-------|----------|---------|
"""
    
    for i, (score, r) in enumerate(scored[:10], 1):
        md += f"| {i} | {r.params.name} | {r.avg_left_count:.1f} | {r.avg_right_count:.1f} | {r.avg_width:.2f}m | {r.avg_symmetry:.3f} | {r.avg_latency_ms:.1f}ms |\n"
    
    md += f"""

## Recommended params.yaml Update

```yaml
ring_boundary_node:
  enable_extraction: true
  max_boundary_width: {best.params.max_boundary_width}
  min_points_per_ring: {best.params.min_points_per_ring}
  y_threshold: {best.params.y_threshold}
  robust_n_points: {best.params.robust_n_points}
  enable_ring_consistency: {str(best.params.enable_ring_consistency).lower()}
  max_y_deviation: {best.params.max_y_deviation}
  enable_temporal_smoothing: true
  temporal_buffer_frames: 5
```

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save recommendations
    output_path = output_dir / 'boundary_recommendations.md'
    with open(output_path, 'w') as f:
        f.write(md)
    print(f"  ✓ Saved: {output_path}")
    
    return md


# ===========================
# Session Finder
# ===========================

def find_latest_session(base_dir: Path = None) -> Optional[Path]:
    """Find the most recent validation session."""
    if base_dir is None:
        base_dir = Path('tools/validation_output')
    
    if not base_dir.exists():
        return None
    
    sessions = sorted(base_dir.glob('session_*'), reverse=True)
    return sessions[0] if sessions else None


# ===========================
# Main Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='Test boundary extraction parameters on labeled validation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 boundary_extraction_tuner.py --latest
    python3 boundary_extraction_tuner.py tools/validation_output/session_20260112_120000/
        """
    )
    parser.add_argument('session_dir', nargs='?', help='Path to validation session directory')
    parser.add_argument('--latest', action='store_true', help='Use the most recent session')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Find session directory
    if args.latest:
        session_dir = find_latest_session()
        if session_dir is None:
            print("❌ No validation sessions found in tools/validation_output/")
            print("   Run detection_validator.py first to create labeled data.")
            sys.exit(1)
    elif args.session_dir:
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"❌ Session directory not found: {session_dir}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("BOUNDARY EXTRACTION PARAMETER TUNER")
    print(f"{'='*60}")
    print(f"Session: {session_dir}")
    
    # Load and validate data
    loader = ValidationDataLoader(session_dir)
    stats = loader.get_statistics()
    print(f"\nDataset: {stats['good_frames']} good, {stats['bad_frames']} bad frames")
    
    if stats['total_frames'] == 0:
        print("❌ No labeled frames found!")
        print("   Label some frames with 'g' (good) or 'b' (bad) in detection_validator.py")
        sys.exit(1)
    
    # Create tester
    tester = BoundaryTester(session_dir)
    
    # Generate parameter grid
    params_grid = generate_param_grid()
    print(f"\nTesting {len(params_grid)} parameter combinations...")
    
    # Test all parameters
    results = []
    for i, params in enumerate(params_grid, 1):
        print(f"  [{i}/{len(params_grid)}] {params.name}...", end='', flush=True)
        result = tester.test_params(params)
        results.append(result)
        print(f" ✓ (sym={result.avg_symmetry:.3f}, w={result.avg_width:.2f}m)")
    
    # Output directory
    output_dir = session_dir / 'boundary_tuning_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_results = [r.to_dict() for r in results]
    json_path = output_dir / 'boundary_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Saved: {json_path}")
    
    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating plots...")
        viz = VisualizationGenerator(output_dir)
        viz.plot_parameter_comparison(results)
        viz.plot_boundary_distribution(tester.frames)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    generate_recommendations(results, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Find best
    best = max(results, key=lambda r: r.avg_symmetry * 0.4 + 
               (1.0 if 5 <= r.avg_left_count <= 15 else 0.5) * 0.3 +
               (1.0 if 3.0 <= r.avg_width <= 5.0 else 0.5) * 0.3)
    
    print(f"\n🏆 Best Parameters: {best.params.name}")
    print(f"   max_boundary_width:  {best.params.max_boundary_width} m")
    print(f"   min_points_per_ring: {best.params.min_points_per_ring}")
    print(f"   y_threshold:         {best.params.y_threshold} m")
    print(f"   robust_n_points:     {best.params.robust_n_points}")
    print(f"   ring_consistency:    {best.params.enable_ring_consistency}")
    print(f"   max_y_deviation:     {best.params.max_y_deviation} m")
    print(f"\n   Left:  {best.avg_left_count:.1f} pts/frame")
    print(f"   Right: {best.avg_right_count:.1f} pts/frame")
    print(f"   Width: {best.avg_width:.2f} m")
    print(f"   Symmetry: {best.avg_symmetry:.3f}")
    print(f"   Latency: {best.avg_latency_ms:.2f} ms")
    
    print(f"\n📁 Output: {output_dir}")
    print(f"   - boundary_results.json")
    print(f"   - boundary_recommendations.md")
    if not args.no_plots:
        print(f"   - boundary_parameter_comparison.png")
        print(f"   - boundary_y_distribution.png")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()