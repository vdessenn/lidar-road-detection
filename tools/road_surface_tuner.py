#!/usr/bin/env python3
"""
Enhanced Filter Test Tool - Updated for Simplified Filter
==========================================================
Tests different filter parameters on labeled validation data (NPZ files).

Usage:
    python3 enhanced_filter_test_tool.py <session_dir>
    python3 enhanced_filter_test_tool.py tools/validation_output/session_XXXX/
    python3 enhanced_filter_test_tool.py --latest

Outputs:
    - parameter_comparison.png - Bar chart of metrics per parameter set
    - intensity_distribution.png - Intensity histograms good vs bad
    - spatial_distribution.png - XY scatter of points
    - results_summary.json - Machine-readable results
    - recommendations.md - Optimal parameters recommendation

The tool tests the simplified filter pipeline:
    1. Intensity filter: [min_intensity, max_intensity]
    2. Range filter: distance < max_range
    3. Density filter: neighbors >= min_neighbors within neighbor_radius
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config, HISTORICAL_DEFAULTS


# ===========================
# Data Structures
# ===========================

@dataclass
class FilterParams:
    """Parameters for the DBSCAN-based road filter (matches Node 01 pipeline)."""
    name: str
    max_intensity: float = 8.0           # Updated from 6.0 to match params.yaml
    max_range: float = 20.0
    min_neighbors: int = 2               # Updated from 5 to match params.yaml
    neighbor_radius: float = 1.2         # Updated from 0.8 to match params.yaml
    # DBSCAN clustering parameters
    dbscan_eps: float = 0.052
    dbscan_min_samples: int = 30
    dbscan_min_cluster_size: int = 50

    def __str__(self):
        return (f"{self.name} (int≤{self.max_intensity}, r≤{self.max_range}m, "
                f"n≥{self.min_neighbors}@{self.neighbor_radius}m, "
                f"dbscan={self.dbscan_eps}/{self.dbscan_min_samples})")


@dataclass
class FrameResult:
    """Per-frame test results."""
    frame_file: str
    params_name: str
    label: str  # "good" or "bad" from ground truth
    input_points: int
    output_points: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    processing_time_ms: float


@dataclass
class ParamsResult:
    """Aggregate results per parameter set."""
    params: FilterParams
    frames_processed: int
    good_frames: int
    bad_frames: int
    # On good frames (we want high recall - keep road points)
    avg_precision_good: float
    avg_recall_good: float
    avg_f1_good: float
    # On bad frames (we want high precision - reject noise)
    avg_precision_bad: float
    avg_recall_bad: float
    avg_f1_bad: float
    # Overall
    avg_latency_ms: float
    frame_results: List[FrameResult] = field(default_factory=list)

    def to_dict(self):
        return {
            'params_name': self.params.name,
            'max_intensity': self.params.max_intensity,
            'max_range': self.params.max_range,
            'min_neighbors': self.params.min_neighbors,
            'neighbor_radius': self.params.neighbor_radius,
            'dbscan_eps': self.params.dbscan_eps,
            'dbscan_min_samples': self.params.dbscan_min_samples,
            'dbscan_min_cluster_size': self.params.dbscan_min_cluster_size,
            'frames_processed': self.frames_processed,
            'good_frames': self.good_frames,
            'bad_frames': self.bad_frames,
            'avg_precision_good': round(self.avg_precision_good, 4),
            'avg_recall_good': round(self.avg_recall_good, 4),
            'avg_f1_good': round(self.avg_f1_good, 4),
            'avg_precision_bad': round(self.avg_precision_bad, 4),
            'avg_recall_bad': round(self.avg_recall_bad, 4),
            'avg_f1_bad': round(self.avg_f1_bad, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 2)
        }


# ===========================
# Simplified Filter Implementation
# ===========================

class SimplifiedRoadFilter:
    """Implements the DBSCAN-based road filter logic (matches Node 01 pipeline)."""

    def __init__(self, params: FilterParams):
        self.params = params

    def filter(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
               intensity: np.ndarray) -> np.ndarray:
        """
        Apply the DBSCAN-based filter pipeline (matches Node 01).

        Pipeline stages:
        1. Intensity filter: [0.0, max_intensity]
        2. Range filter: distance < max_range
        3. Density filter: neighbors >= min_neighbors within neighbor_radius
        4. DBSCAN clustering: remove noise and keep only large clusters

        Returns:
            Boolean mask of points that pass all filters
        """
        n_points = len(x)
        if n_points == 0:
            return np.array([], dtype=bool)

        # 1. Intensity filter
        intensity_mask = intensity <= self.params.max_intensity

        # 2. Range filter
        distances = np.sqrt(x**2 + y**2)
        range_mask = distances <= self.params.max_range

        # Combined mask after intensity + range
        combined_mask = intensity_mask & range_mask

        if not np.any(combined_mask):
            return np.zeros(n_points, dtype=bool)

        # 3. Density filter (only on points that passed intensity + range)
        candidate_indices = np.where(combined_mask)[0]
        candidate_x = x[combined_mask]
        candidate_y = y[combined_mask]

        if len(candidate_x) < self.params.min_neighbors:
            return np.zeros(n_points, dtype=bool)

        # Build KD-tree for density filtering
        coords = np.column_stack((candidate_x, candidate_y))
        tree = cKDTree(coords)

        # Count neighbors for each point (includes self)
        neighbor_counts = tree.query_ball_point(
            coords, r=self.params.neighbor_radius, return_length=True
        )

        # Keep points with enough neighbors (subtract 1 for self)
        density_mask = (neighbor_counts - 1) >= self.params.min_neighbors

        # Apply density filter
        density_indices = candidate_indices[density_mask]

        if len(density_indices) == 0:
            return np.zeros(n_points, dtype=bool)

        # 4. DBSCAN clustering for noise removal
        dbscan_x = x[density_indices]
        dbscan_y = y[density_indices]
        dbscan_coords = np.column_stack((dbscan_x, dbscan_y))

        clustering = DBSCAN(
            eps=self.params.dbscan_eps,
            min_samples=self.params.dbscan_min_samples
        )
        labels = clustering.fit_predict(dbscan_coords)

        # Keep only points in large clusters (exclude noise label -1)
        cluster_mask = np.zeros(len(labels), dtype=bool)
        for label in set(labels) - {-1}:
            label_count = np.sum(labels == label)
            if label_count >= self.params.dbscan_min_cluster_size:
                cluster_mask[labels == label] = True

        # Map back to original indices
        final_mask = np.zeros(n_points, dtype=bool)
        final_mask[density_indices[cluster_mask]] = True

        return final_mask


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
        """
        Load all labeled frames.
        
        Returns:
            List of (filepath, label, data_dict) tuples
        """
        frames = []
        
        # Load good frames
        if self.good_dir.exists():
            for npz_file in sorted(self.good_dir.glob('*.npz')):
                data = dict(np.load(npz_file))
                frames.append((npz_file, 'good', data))
        
        # Load bad frames
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
# Filter Tester
# ===========================

class FilterTester:
    """Tests filter parameters on validation data."""

    def __init__(self, session_dir: Path):
        self.loader = ValidationDataLoader(session_dir)
        self.frames = self.loader.load_frames()

    def test_params(self, params: FilterParams) -> ParamsResult:
        """
        Test a set of filter parameters on all frames.
        
        For GOOD frames: We use ground points as input, road points as ground truth.
                         High recall means we correctly extract road from ground.
        
        For BAD frames: The road_points in these frames are noise (false detections).
                        High precision means we reject the noise.
        """
        road_filter = SimplifiedRoadFilter(params)
        frame_results = []
        
        for filepath, label, data in self.frames:
            # Use ground points as input (simulate what node 01 receives)
            ground_x = data.get('ground_x', np.array([]))
            ground_y = data.get('ground_y', np.array([]))
            ground_z = data.get('ground_z', np.array([]))
            ground_intensity = data.get('ground_intensity', np.array([]))
            
            # Road points are the "labeled" output (what the current filter produced)
            road_x = data.get('road_x', np.array([]))
            road_y = data.get('road_y', np.array([]))
            road_z = data.get('road_z', np.array([]))
            road_intensity = data.get('road_intensity', np.array([]))
            
            n_ground = len(ground_x)
            n_road_gt = len(road_x)
            
            if n_ground == 0:
                continue
            
            # Time the filter
            start_time = time.perf_counter()
            output_mask = road_filter.filter(ground_x, ground_y, ground_z, ground_intensity)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            n_output = np.sum(output_mask)
            
            # Calculate metrics based on frame type
            if label == 'good':
                # For GOOD frames: Compare filter output to validated road points
                # Build ground truth mask by matching road points to ground points
            
                if n_road_gt == 0:
                    continue  # Skip if no ground truth
                
                # Create ground truth mask: which ground points are road points?
                # Match by spatial proximity (within 0.1m tolerance)
                
                ground_coords = np.column_stack((ground_x, ground_y, ground_z))
                road_coords = np.column_stack((road_x, road_y, road_z))
                
                tree = cKDTree(ground_coords)
                distances, indices = tree.query(road_coords, k=1)
                
                # Create ground truth mask: ground points within 0.1m of any road point
                gt_mask = np.zeros(n_ground, dtype=bool)
                valid_matches = distances < 0.1  # 10cm tolerance
                gt_mask[indices[valid_matches]] = True
                
                # Calculate metrics
                tp = np.sum(output_mask & gt_mask)    # Correctly found road points
                fp = np.sum(output_mask & ~gt_mask)   # False detections (noise)
                fn = np.sum(~output_mask & gt_mask)   # Missed road points
                
            else:
                # For BAD frames: All road_points are noise, reject them
                # Goal: minimize output (high precision = good rejection)
                
                tp = 0  # No true road points exist
                fp = n_output  # Everything we output is false positive
                fn = 0  # Can't miss what doesn't exist
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            frame_results.append(FrameResult(
                frame_file=str(filepath.name),
                params_name=params.name,
                label=label,
                input_points=n_ground,
                output_points=int(n_output),
                true_positives=int(tp),
                false_positives=int(fp),
                false_negatives=int(fn),
                precision=precision,
                recall=recall,
                f1_score=f1,
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
            avg_precision_good=np.mean([r.precision for r in good_results]) if good_results else 0.0,
            avg_recall_good=np.mean([r.recall for r in good_results]) if good_results else 0.0,
            avg_f1_good=np.mean([r.f1_score for r in good_results]) if good_results else 0.0,
            avg_precision_bad=np.mean([r.precision for r in bad_results]) if bad_results else 0.0,
            avg_recall_bad=np.mean([r.recall for r in bad_results]) if bad_results else 0.0,
            avg_f1_bad=np.mean([r.f1_score for r in bad_results]) if bad_results else 0.0,
            avg_latency_ms=np.mean([r.processing_time_ms for r in frame_results]) if frame_results else 0.0,
            frame_results=frame_results
        )


# ===========================
# Parameter Grid
# ===========================

def generate_param_grid() -> List[FilterParams]:
    """Generate parameter combinations to test (DBSCAN-based pipeline)."""
    params_list = []

    # Load current baseline from params.yaml using ConfigLoader
    config = get_config()
    road_params = config.get_road_extraction_params()

    print(f"  Loaded baseline from params.yaml:")
    print(f"    max_intensity={road_params.max_road_intensity}, max_range={road_params.max_range}")
    print(f"    min_neighbors={road_params.min_neighbors}, neighbor_radius={road_params.neighbor_radius}")
    print(f"    dbscan_eps={road_params.dbscan_eps}, dbscan_min_samples={road_params.dbscan_min_samples}")
    print(f"    dbscan_min_cluster_size={road_params.dbscan_min_cluster_size}")

    # Current baseline from params.yaml
    params_list.append(FilterParams(
        name="baseline_current",
        max_intensity=road_params.max_road_intensity,
        max_range=road_params.max_range,
        min_neighbors=road_params.min_neighbors,
        neighbor_radius=road_params.neighbor_radius,
        dbscan_eps=road_params.dbscan_eps,
        dbscan_min_samples=road_params.dbscan_min_samples,
        dbscan_min_cluster_size=road_params.dbscan_min_cluster_size
    ))

    # Historical baseline for comparison (original values before tuning)
    hist = HISTORICAL_DEFAULTS['road_points_extraction_node']
    params_list.append(FilterParams(
        name="baseline_historical",
        max_intensity=hist['max_road_intensity'],
        max_range=hist['max_range'],
        min_neighbors=hist['min_neighbors'],
        neighbor_radius=hist['neighbor_radius'],
        dbscan_eps=hist['dbscan_eps'],
        dbscan_min_samples=hist['dbscan_min_samples'],
        dbscan_min_cluster_size=hist['dbscan_min_cluster_size']
    ))
    print(f"  Added historical baseline for comparison:")
    print(f"    max_intensity={hist['max_road_intensity']}, max_range={hist['max_range']}")
    print(f"    min_neighbors={hist['min_neighbors']}, neighbor_radius={hist['neighbor_radius']}")
    print(f"    dbscan_eps={hist['dbscan_eps']}, dbscan_min_samples={hist['dbscan_min_samples']}")

    # ==================================================================
    # ADAPTIVE VARIATIONS: Test around current baseline
    # ==================================================================
    
    # Vary intensity threshold (±50% and ±100% from baseline, avoid duplicates)
    base_intensity = road_params.max_road_intensity
    intensity_variations = [
        base_intensity * 0.5,   # -50%
        base_intensity * 0.75,  # -25%
        base_intensity * 1.25,  # +25%
        base_intensity * 1.5,   # +50%
    ]
    intensity_variations = [v for v in intensity_variations if abs(v - base_intensity) > 0.5]  # Skip if too close to baseline
    
    for max_int in intensity_variations:
        params_list.append(FilterParams(
            name=f"int_{max_int:.1f}",
            max_intensity=max_int,
            max_range=road_params.max_range,
            min_neighbors=road_params.min_neighbors,
            neighbor_radius=road_params.neighbor_radius
        ))
    
    # Vary range (±33% from baseline)
    base_range = road_params.max_range
    range_variations = [
        base_range * 0.67,  # -33%
        base_range * 1.33,  # +33%
    ]
    range_variations = [v for v in range_variations if abs(v - base_range) > 1.0]
    
    for max_r in range_variations:
        params_list.append(FilterParams(
            name=f"range_{max_r:.1f}",
            max_intensity=road_params.max_road_intensity,
            max_range=max_r,
            min_neighbors=road_params.min_neighbors,
            neighbor_radius=road_params.neighbor_radius
        ))
    
    # Vary density: min_neighbors (±2 from baseline)
    base_neighbors = road_params.min_neighbors
    for delta in [-2, +2]:
        min_n = base_neighbors + delta
        if min_n > 0:  # Must be positive
            params_list.append(FilterParams(
                name=f"density_n{min_n}",
                max_intensity=road_params.max_road_intensity,
                max_range=road_params.max_range,
                min_neighbors=min_n,
                neighbor_radius=road_params.neighbor_radius
            ))
    
    # Vary density: neighbor_radius (±50% from baseline)
    base_radius = road_params.neighbor_radius
    radius_variations = [
        base_radius * 0.5,   # -50%
        base_radius * 1.5,   # +50%
    ]
    
    for radius in radius_variations:
        params_list.append(FilterParams(
            name=f"density_r{radius:.1f}",
            max_intensity=road_params.max_road_intensity,
            max_range=road_params.max_range,
            min_neighbors=road_params.min_neighbors,
            neighbor_radius=radius
        ))
    
    # ==================================================================
    # FIXED REFERENCE POINTS: For cross-run comparison
    # ==================================================================
    
    # Conservative: Strict filtering
    params_list.append(FilterParams(
        name="reference_conservative",
        max_intensity=4.0,
        max_range=15.0,
        min_neighbors=7,
        neighbor_radius=0.5
    ))
    
    # Permissive: Relaxed filtering
    params_list.append(FilterParams(
        name="reference_permissive",
        max_intensity=10.0,
        max_range=25.0,
        min_neighbors=3,
        neighbor_radius=1.0
    ))
    
    # Aggressive noise rejection
    params_list.append(FilterParams(
        name="reference_noise_reject",
        max_intensity=5.0,
        max_range=20.0,
        min_neighbors=8,
        neighbor_radius=0.6
    ))
    
    # Maximum coverage
    params_list.append(FilterParams(
        name="reference_max_coverage",
        max_intensity=12.0,
        max_range=30.0,
        min_neighbors=2,
        neighbor_radius=1.2
    ))

    # ==================================================================
    # DBSCAN VARIATIONS: Test clustering parameters
    # ==================================================================

    base_eps = road_params.dbscan_eps
    base_min_samples = road_params.dbscan_min_samples
    base_min_cluster = road_params.dbscan_min_cluster_size

    # Vary DBSCAN eps (neighborhood radius for clustering)
    for eps_mult in [0.5, 2.0, 4.0]:
        new_eps = base_eps * eps_mult
        params_list.append(FilterParams(
            name=f"dbscan_eps_{new_eps:.3f}",
            max_intensity=road_params.max_road_intensity,
            max_range=road_params.max_range,
            min_neighbors=road_params.min_neighbors,
            neighbor_radius=road_params.neighbor_radius,
            dbscan_eps=new_eps,
            dbscan_min_samples=base_min_samples,
            dbscan_min_cluster_size=base_min_cluster
        ))

    # Vary DBSCAN min_samples (core point threshold)
    for min_samp in [10, 20, 50]:
        if min_samp != base_min_samples:
            params_list.append(FilterParams(
                name=f"dbscan_samples_{min_samp}",
                max_intensity=road_params.max_road_intensity,
                max_range=road_params.max_range,
                min_neighbors=road_params.min_neighbors,
                neighbor_radius=road_params.neighbor_radius,
                dbscan_eps=base_eps,
                dbscan_min_samples=min_samp,
                dbscan_min_cluster_size=base_min_cluster
            ))

    # Vary DBSCAN min_cluster_size (minimum cluster to keep)
    for min_clust in [20, 100, 200]:
        if min_clust != base_min_cluster:
            params_list.append(FilterParams(
                name=f"dbscan_cluster_{min_clust}",
                max_intensity=road_params.max_road_intensity,
                max_range=road_params.max_range,
                min_neighbors=road_params.min_neighbors,
                neighbor_radius=road_params.neighbor_radius,
                dbscan_eps=base_eps,
                dbscan_min_samples=base_min_samples,
                dbscan_min_cluster_size=min_clust
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
        if not results:
            print("  ⚠ No results to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Filter Parameters Comparison', fontsize=16, fontweight='bold')

        names = [r.params.name for r in results]
        x = np.arange(len(names))
        width = 0.35

        # Plot 1: F1 on Good vs Bad frames (FIXED - now showing both)
        ax = axes[0, 0]
        good_f1 = [r.avg_f1_good for r in results]
        bad_f1 = [r.avg_f1_bad for r in results]
        
        bars1 = ax.bar(x - width/2, good_f1, width, label='Good Frames', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, bad_f1, width, label='Bad Frames', color='red', alpha=0.7)
        
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title('F1 Score by Frame Type', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Only show if significant
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=6)

        # Plot 2: Precision on Good frames (FIXED - now showing values)
        ax = axes[0, 1]
        prec_good = [r.avg_precision_good for r in results]
        bars = ax.bar(x, prec_good, color='blue', alpha=0.7)
        
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision on Good Frames (Higher = Less Noise)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, label='Target 90%')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)

        # Plot 3: Recall on Good frames
        ax = axes[1, 0]
        recall_good = [r.avg_recall_good for r in results]
        bars = ax.bar(x, recall_good, color='orange', alpha=0.7)
        
        ax.set_ylabel('Recall', fontsize=11)
        ax.set_title('Recall on Good Frames (Higher = More Road Kept)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, label='Target 80%')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)

        # Plot 4: Processing latency
        ax = axes[1, 1]
        latencies = [r.avg_latency_ms for r in results]
        bars = ax.bar(x, latencies, color='purple', alpha=0.7)
        
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('Processing Time (Lower = Faster)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Set budget based on max latency
        if latencies:
            max_latency = max(latencies)
            budget = max(100, max_latency * 1.2)
            ax.axhline(y=budget, color='red', linestyle='--', linewidth=1.5, label=f'Budget {budget:.0f}ms')
            ax.legend()
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)
        
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'parameter_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    def plot_intensity_distribution(self, frames: List[Tuple[Path, str, dict]]):
        """Compare intensity distributions between good and bad frames."""
        good_road_int = []
        good_ground_int = []
        bad_road_int = []
        bad_ground_int = []
        
        for _, label, data in frames:
            road_int = data.get('road_intensity', np.array([]))
            ground_int = data.get('ground_intensity', np.array([]))
            
            if label == 'good':
                if len(road_int) > 0:
                    good_road_int.extend(road_int)
                if len(ground_int) > 0:
                    good_ground_int.extend(ground_int)
            else:
                if len(road_int) > 0:
                    bad_road_int.extend(road_int)
                if len(ground_int) > 0:
                    bad_ground_int.extend(ground_int)
        
        # Convert to numpy arrays and remove NaN/inf values
        good_road_int = np.array(good_road_int)
        good_ground_int = np.array(good_ground_int)
        bad_road_int = np.array(bad_road_int)
        bad_ground_int = np.array(bad_ground_int)
        
        # Filter out NaN and inf values
        good_road_int = good_road_int[np.isfinite(good_road_int)]
        good_ground_int = good_ground_int[np.isfinite(good_ground_int)]
        bad_road_int = bad_road_int[np.isfinite(bad_road_int)]
        bad_ground_int = bad_ground_int[np.isfinite(bad_ground_int)]
        
        print(f"  Intensity data loaded:")
        print(f"    Good road: {len(good_road_int)} points")
        print(f"    Good ground: {len(good_ground_int)} points")
        print(f"    Bad road: {len(bad_road_int)} points")
        print(f"    Bad ground: {len(bad_ground_int)} points")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Intensity Distribution Analysis', fontsize=16, fontweight='bold')
        
        # FIXED: Bins from 0 to 20 instead of 50
        bins = np.linspace(0, 20, 41)  # 0.5 intensity per bin
        
        # Good frames - Road points
        if len(good_road_int) > 0:
            good_road_mean = np.mean(good_road_int)
            good_road_std = np.std(good_road_int)
            good_road_median = np.median(good_road_int)
            
            axes[0, 0].hist(good_road_int, bins=bins, alpha=0.7, color='green', edgecolor='black')
            axes[0, 0].axvline(good_road_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {good_road_mean:.2f}')
            axes[0, 0].axvline(good_road_median, color='blue', linestyle=':', linewidth=2,
                            label=f'Median: {good_road_median:.2f}')
            axes[0, 0].set_title(f'GOOD Frames - Road Points (n={len(good_road_int)})', fontweight='bold')
            axes[0, 0].set_xlabel('Intensity')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)
            axes[0, 0].set_xlim(0, 20)
            
            # Add text box with statistics
            stats_text = f'μ={good_road_mean:.2f}\nσ={good_road_std:.2f}'
            axes[0, 0].text(0.95, 0.95, stats_text, transform=axes[0, 0].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[0, 0].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                        transform=axes[0, 0].transAxes, fontsize=14, color='gray')
            axes[0, 0].set_title('GOOD Frames - Road Points (n=0)', fontweight='bold')
            axes[0, 0].set_xlim(0, 20)
        
        # Good frames - Ground points
        if len(good_ground_int) > 0:
            good_ground_mean = np.mean(good_ground_int)
            good_ground_std = np.std(good_ground_int)
            good_ground_median = np.median(good_ground_int)
            
            axes[0, 1].hist(good_ground_int, bins=bins, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].axvline(good_ground_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {good_ground_mean:.2f}')
            axes[0, 1].axvline(good_ground_median, color='orange', linestyle=':', linewidth=2,
                            label=f'Median: {good_ground_median:.2f}')
            axes[0, 1].set_title(f'GOOD Frames - Ground Points (n={len(good_ground_int)})', fontweight='bold')
            axes[0, 1].set_xlabel('Intensity')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
            axes[0, 1].set_xlim(0, 20)
            
            # Add text box with statistics
            stats_text = f'μ={good_ground_mean:.2f}\nσ={good_ground_std:.2f}'
            axes[0, 1].text(0.95, 0.95, stats_text, transform=axes[0, 1].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                        transform=axes[0, 1].transAxes, fontsize=14, color='gray')
            axes[0, 1].set_title('GOOD Frames - Ground Points (n=0)', fontweight='bold')
            axes[0, 1].set_xlim(0, 20)
        
        # Bad frames - Road points
        if len(bad_road_int) > 0:
            bad_road_mean = np.mean(bad_road_int)
            bad_road_std = np.std(bad_road_int)
            bad_road_median = np.median(bad_road_int)
            
            axes[1, 0].hist(bad_road_int, bins=bins, alpha=0.7, color='red', edgecolor='black')
            axes[1, 0].axvline(bad_road_mean, color='black', linestyle='--', linewidth=2,
                            label=f'Mean: {bad_road_mean:.2f}')
            axes[1, 0].axvline(bad_road_median, color='yellow', linestyle=':', linewidth=2,
                            label=f'Median: {bad_road_median:.2f}')
            axes[1, 0].set_title(f'BAD Frames - Road Points = NOISE (n={len(bad_road_int)})', fontweight='bold')
            axes[1, 0].set_xlabel('Intensity')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
            axes[1, 0].set_xlim(0, 20)
            
            # Add text box with statistics
            stats_text = f'μ={bad_road_mean:.2f}\nσ={bad_road_std:.2f}'
            axes[1, 0].text(0.95, 0.95, stats_text, transform=axes[1, 0].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=14, color='gray')
            axes[1, 0].set_title('BAD Frames - Road Points = NOISE (n=0)', fontweight='bold')
            axes[1, 0].set_xlim(0, 20)
        
        # Bad frames - Ground points
        if len(bad_ground_int) > 0:
            bad_ground_mean = np.mean(bad_ground_int)
            bad_ground_std = np.std(bad_ground_int)
            bad_ground_median = np.median(bad_ground_int)
            
            axes[1, 1].hist(bad_ground_int, bins=bins, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].axvline(bad_ground_mean, color='black', linestyle='--', linewidth=2,
                            label=f'Mean: {bad_ground_mean:.2f}')
            axes[1, 1].axvline(bad_ground_median, color='purple', linestyle=':', linewidth=2,
                            label=f'Median: {bad_ground_median:.2f}')
            axes[1, 1].set_title(f'BAD Frames - Ground Points (n={len(bad_ground_int)})', fontweight='bold')
            axes[1, 1].set_xlabel('Intensity')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)
            axes[1, 1].set_xlim(0, 20)
            
            # Add text box with statistics
            stats_text = f'μ={bad_ground_mean:.2f}\nσ={bad_ground_std:.2f}'
            axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=14, color='gray')
            axes[1, 1].set_title('BAD Frames - Ground Points (n=0)', fontweight='bold')
            axes[1, 1].set_xlim(0, 20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'intensity_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
        
        # FIXED: Return correct statistics, handling empty arrays properly
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else 0.0
        
        def safe_std(arr):
            return float(np.std(arr)) if len(arr) > 0 else 0.0
        
        def safe_median(arr):
            return float(np.median(arr)) if len(arr) > 0 else 0.0
        
        return {
            'good_road_mean': safe_mean(good_road_int),
            'good_road_std': safe_std(good_road_int),
            'good_road_median': safe_median(good_road_int),
            'bad_road_mean': safe_mean(bad_road_int),
            'bad_road_std': safe_std(bad_road_int),
            'bad_road_median': safe_median(bad_road_int),
            'good_road_count': len(good_road_int),
            'bad_road_count': len(bad_road_int),
        }

    def plot_spatial_distribution(self, frames: List[Tuple[Path, str, dict]]):
        """Compare spatial distribution between good and bad frames."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Spatial Distribution: Good vs Bad Frames', fontsize=16, fontweight='bold')
        
        # Good frames
        good_points_plotted = False
        good_point_count = 0
        for _, label, data in frames:
            if label == 'good':
                x = data.get('road_x', np.array([]))
                y = data.get('road_y', np.array([]))
                if len(x) > 0:
                    axes[0].scatter(y, x, s=1, alpha=0.3, c='green')
                    good_points_plotted = True
                    good_point_count += len(x)
        
        axes[0].set_title(f'GOOD Frames - Road Points (n={good_point_count})', fontweight='bold')
        axes[0].set_xlabel('Y (lateral, m)')
        axes[0].set_ylabel('X (forward, m)')
        axes[0].set_xlim(-15, 15)
        axes[0].set_ylim(0, 30)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linewidth=0.5)
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        
        if not good_points_plotted:
            axes[0].text(0, 15, 'No data', ha='center', va='center', fontsize=14, color='gray')
        
        # Bad frames
        bad_points_plotted = False
        bad_point_count = 0
        for _, label, data in frames:
            if label == 'bad':
                x = data.get('road_x', np.array([]))
                y = data.get('road_y', np.array([]))
                if len(x) > 0:
                    axes[1].scatter(y, x, s=1, alpha=0.3, c='red')
                    bad_points_plotted = True
                    bad_point_count += len(x)
        
        axes[1].set_title(f'BAD Frames - Road Points = NOISE (n={bad_point_count})', fontweight='bold')
        axes[1].set_xlabel('Y (lateral, m)')
        axes[1].set_ylabel('X (forward, m)')
        axes[1].set_xlim(-15, 15)
        axes[1].set_ylim(0, 30)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        axes[1].axvline(x=0, color='black', linewidth=0.5)
        
        if not bad_points_plotted:
            axes[1].text(0, 15, 'No data', ha='center', va='center', fontsize=14, color='gray')
        
        plt.tight_layout()
        output_path = self.output_dir / 'spatial_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
        

# ===========================
# Recommendations Generator
# ===========================

def generate_recommendations(results: List[ParamsResult], intensity_stats: dict, 
                            output_dir: Path) -> str:
    """Generate recommendations based on test results."""
    
    # Find best parameters
    # Score = F1_good * 0.6 + precision_bad * 0.4 (prioritize good detection, penalize noise)
    scored = []
    for r in results:
        score = r.avg_f1_good * 0.6 + r.avg_precision_bad * 0.4
        scored.append((score, r))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    
    md = f"""# Filter Parameter Recommendations

## Test Summary
- **Total frames tested**: {results[0].frames_processed if results else 0}
- **Good frames**: {results[0].good_frames if results else 0}
- **Bad frames**: {results[0].bad_frames if results else 0}

## Intensity Analysis
- Good frames road intensity: **{intensity_stats['good_road_mean']:.1f}** ± {intensity_stats['good_road_std']:.1f}
- Bad frames road intensity (noise): **{intensity_stats['bad_road_mean']:.1f}** ± {intensity_stats['bad_road_std']:.1f}

## Best Parameters: **{best.params.name}**

| Parameter | Value |
|-----------|-------|
| max_intensity | {best.params.max_intensity} |
| max_range | {best.params.max_range} m |
| min_neighbors | {best.params.min_neighbors} |
| neighbor_radius | {best.params.neighbor_radius} m |

### Performance
- F1 on Good frames: **{best.avg_f1_good:.3f}**
- Precision on Good frames: **{best.avg_precision_good:.3f}**
- Recall on Good frames: **{best.avg_recall_good:.3f}**
- Latency: **{best.avg_latency_ms:.2f} ms**

## All Results Ranked

| Rank | Name | F1 Good | Prec Good | Recall Good | Latency |
|------|------|---------|-----------|-------------|---------|
"""
    
    for i, (score, r) in enumerate(scored[:10], 1):
        md += f"| {i} | {r.params.name} | {r.avg_f1_good:.3f} | {r.avg_precision_good:.3f} | {r.avg_recall_good:.3f} | {r.avg_latency_ms:.1f}ms |\n"
    
    md += f"""

## Recommended params.yaml Update

```yaml
road_points_extraction_node:
  enable_extraction: true
  min_road_intensity: 0.0
  max_road_intensity: {best.params.max_intensity}
  max_range: {best.params.max_range}
  min_neighbors: {best.params.min_neighbors}
  neighbor_radius: {best.params.neighbor_radius}
  # DBSCAN clustering parameters
  dbscan_eps: {best.params.dbscan_eps}
  dbscan_min_samples: {best.params.dbscan_min_samples}
  dbscan_min_cluster_size: {best.params.dbscan_min_cluster_size}
```

## Next Steps
1. Update `config/params.yaml` with recommended values
2. Rebuild: `colcon build --packages-select drivable_area_detector`
3. Test: `make run-bag-spanlidar`
4. Re-validate with `detection_validator.py`
"""
    
    output_path = output_dir / 'recommendations.md'
    output_path.write_text(md)
    print(f"  ✓ Saved: {output_path}")
    
    return best.params.name


# ===========================
# Main CLI
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Filter Test Tool - Test filter parameters on labeled validation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on latest validation session
  python3 enhanced_filter_test_tool.py tools/validation_output/session_20260112_143022/
  
  # Find latest session automatically
  python3 enhanced_filter_test_tool.py --latest

Outputs saved to session_dir/analysis/:
  - parameter_comparison.png
  - intensity_distribution.png
  - spatial_distribution.png
  - results_summary.json
  - recommendations.md
        """
    )

    parser.add_argument('session_dir', type=str, nargs='?', 
                       help='Path to validation session directory')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest validation session')

    args = parser.parse_args()

    # Find session directory
    if args.latest or not args.session_dir:
        validation_dir = Path('/home/ws/tools/validation_output')
        sessions = sorted(validation_dir.glob('session_*'))
        if not sessions:
            print("ERROR: No validation sessions found")
            sys.exit(1)
        session_dir = sessions[-1]
        print(f"Using latest session: {session_dir}")
    else:
        session_dir = Path(args.session_dir)
    
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        sys.exit(1)

    # Create output directory
    output_dir = session_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("ENHANCED FILTER TEST TOOL")
    print("="*60)
    
    loader = ValidationDataLoader(session_dir)
    stats = loader.get_statistics()
    frames = loader.load_frames()
    
    print(f"\nSession: {session_dir}")
    print(f"  Good frames: {stats['good_frames']}")
    print(f"  Bad frames:  {stats['bad_frames']}")
    print(f"  Total:       {stats['total_frames']}")

    if stats['total_frames'] == 0:
        print("\nERROR: No labeled frames found")
        sys.exit(1)

    # Generate parameter grid
    print("\nStep 1: Generating parameter grid...")
    params_grid = generate_param_grid()
    print(f"  Testing {len(params_grid)} parameter combinations")

    # Test each parameter set
    print("\nStep 2: Testing parameters...")
    tester = FilterTester(session_dir)
    results = []
    
    for params in params_grid:
        result = tester.test_params(params)
        results.append(result)
        print(f"  ✓ {params.name}: F1_good={result.avg_f1_good:.3f}, Prec={result.avg_precision_good:.3f}")

    # Generate visualizations
    print("\nStep 3: Generating visualizations...")
    viz = VisualizationGenerator(output_dir)
    viz.plot_parameter_comparison(results)
    intensity_stats = viz.plot_intensity_distribution(frames)
    viz.plot_spatial_distribution(frames)

    # Save JSON results
    print("\nStep 4: Saving results...")
    results_json = {
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'session_dir': str(session_dir),
        'stats': stats,
        'intensity_stats': intensity_stats,
        'results': [r.to_dict() for r in results]
    }
    
    json_path = output_dir / 'results_summary.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    # Generate recommendations
    print("\nStep 5: Generating recommendations...")
    best_name = generate_recommendations(results, intensity_stats, output_dir)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Best parameters: {best_name}")
    print("\nNext steps:")
    print("  1. Review recommendations.md")
    print("  2. Check visualization plots")
    print("  3. Update params.yaml with recommended values")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
