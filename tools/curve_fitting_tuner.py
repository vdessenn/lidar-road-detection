#!/usr/bin/env python3
"""
Curve Fitting Tuner (Node 02)
=============================
Tests different curve fitting parameters for ring_boundary_extraction (Node 02).

Usage:
    python3 curve_fitting_tuner.py --latest
    python3 curve_fitting_tuner.py tools/validation_output/session_XXXX/

Parameters tested (from params.yaml ring_boundary_node section):
    - fitting_method: 'ransac', 'spline', or 'both'
    - polynomial_degree: Degree of polynomial fit (1=linear, 2=quadratic, 3=cubic)
    - ransac_residual_threshold: RANSAC inlier threshold (meters)
    - min_points_for_fitting: Minimum points required for curve fitting
    - curve_resampling_spacing: Uniform spacing for resampled points (meters)
    - enforce_constant_width: Whether to enforce constant road width
    - width_tolerance: Width variation tolerance (fraction of median)

Note: This tool was repurposed from marker_array_tuner.py after the "Less is More"
refactoring moved curve fitting from Node 03 (rviz2_publisher) to Node 02.

Makefile:
    make tune-curves              # Tune with latest session
    make tune-curves SESSION=XX   # Tune with specific session
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# ===========================
# Data Structures
# ===========================

@dataclass
class CurveFittingParams:
    """Parameters for Node 02 curve fitting (ring_boundary_node)."""
    name: str
    fitting_method: str = 'ransac'          # 'ransac', 'spline', or 'both'
    polynomial_degree: int = 2              # Degree for polynomial fitting
    ransac_residual_threshold: float = 0.15 # RANSAC outlier threshold (m)
    max_iterations: int = 300              # RANSAC max iterations
    min_inlier_ratio: float = 0.15         # RANSAC min inlier ratio
    min_points_for_fitting: int = 10        # Minimum points required
    curve_resampling_spacing: float = 0.5   # Uniform spacing (m)
    enforce_constant_width: bool = True     # Enforce constant road width
    width_tolerance: float = 0.1            # Width variation tolerance (10%)

    def __str__(self):
        return (f"{self.name} (method={self.fitting_method}, deg={self.polynomial_degree}, "
                f"thresh={self.ransac_residual_threshold}m, const_width={self.enforce_constant_width})")


@dataclass
class FitResult:
    """Per-frame fitting results."""
    frame_file: str
    params_name: str
    label: str
    # Fit quality
    left_inliers: int
    right_inliers: int
    left_rmse: float
    right_rmse: float
    # Boundary metrics
    width_at_origin: float
    width_at_max_x: float
    width_std: float  # Width consistency
    # Smoothness
    max_curvature: float
    smoothness_score: float  # Higher = smoother
    # Performance
    processing_time_ms: float


@dataclass
class ParamsResult:
    """Aggregate results per parameter set."""
    params: CurveFittingParams
    frames_processed: int
    good_frames: int
    bad_frames: int
    # On good frames
    avg_inlier_ratio: float
    avg_rmse: float
    avg_width: float
    avg_width_std: float
    avg_smoothness: float
    # On bad frames
    bad_inlier_ratio: float
    bad_rmse: float
    # Performance
    avg_latency_ms: float
    success_rate: float  # Percentage of frames with valid fit
    fit_results: List[FitResult] = field(default_factory=list)

    def to_dict(self):
        return {
            'params_name': self.params.name,
            'fitting_method': self.params.fitting_method,
            'polynomial_degree': self.params.polynomial_degree,
            'ransac_residual_threshold': self.params.ransac_residual_threshold,
            'min_points_for_fitting': self.params.min_points_for_fitting,
            'curve_resampling_spacing': self.params.curve_resampling_spacing,
            'enforce_constant_width': self.params.enforce_constant_width,
            'width_tolerance': self.params.width_tolerance,
            'frames_processed': self.frames_processed,
            'good_frames': self.good_frames,
            'bad_frames': self.bad_frames,
            'avg_inlier_ratio': round(self.avg_inlier_ratio, 3),
            'avg_rmse': round(self.avg_rmse, 4),
            'avg_width': round(self.avg_width, 2),
            'avg_width_std': round(self.avg_width_std, 3),
            'avg_smoothness': round(self.avg_smoothness, 3),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'success_rate': round(self.success_rate, 3)
        }


# ===========================
# RANSAC Polynomial Fitter (standalone, matches rviz2_publisher.py)
# ===========================

class PolynomialRANSAC:
    """RANSAC polynomial fitter using scikit-learn."""
    def __init__(self, degree: int = 1, residual_threshold: float = 0.3, 
                 max_iterations: int = 300, min_inlier_ratio: float = 0.15):
        self.degree = degree
        self.residual_threshold = residual_threshold
        self.max_iterations = max_iterations
        self.min_inlier_ratio = min_inlier_ratio
        
        # Create polynomial regression pipeline
        base_model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        
        # Wrap with RANSAC
        self.ransac = RANSACRegressor(
            estimator=base_model,
            min_samples=degree + 1,
            residual_threshold=residual_threshold,
            max_trials=max_iterations,
            random_state=42,  # ← Pour reproductibilité
            stop_probability=0.99  # ← Stopper tôt si bon fit trouvé
        )
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        """Fit polynomial using RANSAC."""
        n = len(x)
        if n < self.degree + 1:
            return None, 0.0, 0
        
        # Reshape for sklearn (needs 2D array)
        X = x.reshape(-1, 1)
        
        try:
            # Fit RANSAC
            self.ransac.fit(X, y)
            
            # Get inlier mask
            inlier_mask = self.ransac.inlier_mask_
            n_inliers = np.sum(inlier_mask)
            
            # Check minimum inlier ratio
            if n_inliers < n * self.min_inlier_ratio:
                return None, 0.0, 0
            
            # Calculate RMSE on inliers
            y_pred = self.ransac.predict(X[inlier_mask])
            rmse = np.sqrt(np.mean((y[inlier_mask] - y_pred) ** 2))
            
            # Extract polynomial coefficients
            # Note: sklearn stores them differently than np.polyfit
            linear_model = self.ransac.estimator_.named_steps['linearregression']
            poly_features = self.ransac.estimator_.named_steps['polynomialfeatures']
            
            # Convert sklearn coefficients to numpy polyfit format
            coef = self._extract_coefficients(linear_model, self.degree)
            
            return coef, rmse, n_inliers
            
        except Exception as e:
            return None, 0.0, 0
    
    def _extract_coefficients(self, model, degree):
        """Extract polynomial coefficients in numpy format (highest degree first)."""
        # sklearn stores: intercept + coef[0]*x + coef[1]*x^2 + ...
        # numpy expects: [coef[n]*x^n, ..., coef[1]*x, intercept]
        
        # Get the linear regression from the pipeline
        poly_model = model.named_steps['polynomialfeatures']
        lin_model = model.named_steps['linearregression']
        
        # Build coefficient array in numpy format
        coef = np.zeros(degree + 1)
        
        # Intercept goes last (constant term)
        coef[-1] = lin_model.intercept_
        
        # Coefficients in reverse order (highest degree first)
        sklearn_coefs = lin_model.coef_[1:]  # Skip constant (always 1.0)
        for i in range(min(len(sklearn_coefs), degree)):
            coef[degree - 1 - i] = sklearn_coefs[i]
        
        return coef

class MarkerArrayFitter:
    """Tests RANSAC fitting on boundary points (matches rviz2_publisher.py logic)."""
    
    def __init__(self, params: CurveFittingParams):
        self.params = params
        self.fitter = PolynomialRANSAC(
            degree=params.polynomial_degree,
            residual_threshold=params.ransac_residual_threshold,
            max_iterations=params.max_iterations,
            min_inlier_ratio=params.min_inlier_ratio
        )
    
    def fit_boundaries(self, left_x: np.ndarray, left_y: np.ndarray,
                       right_x: np.ndarray, right_y: np.ndarray) -> Dict:
        """
        Fit polynomials to left and right boundaries.
        
        Returns:
            Dictionary with fit results
        """
        result = {
            'left_coef': None, 'right_coef': None,
            'left_rmse': 0.0, 'right_rmse': 0.0,
            'left_inliers': 0, 'right_inliers': 0,
            'width_at_origin': 0.0, 'width_at_max_x': 0.0,
            'width_std': 0.0, 'max_curvature': 0.0,
            'smoothness_score': 0.0, 'success': False
        }
        
        # Fit left boundary
        if len(left_x) >= self.params.polynomial_degree + 1:
            coef, rmse, inliers = self.fitter.fit(left_x, left_y)
            if coef is not None:
                result['left_coef'] = coef
                result['left_rmse'] = rmse
                result['left_inliers'] = inliers
        
        # Fit right boundary
        if len(right_x) >= self.params.polynomial_degree + 1:
            coef, rmse, inliers = self.fitter.fit(right_x, right_y)
            if coef is not None:
                result['right_coef'] = coef
                result['right_rmse'] = rmse
                result['right_inliers'] = inliers
        
        # Calculate metrics if both fits succeeded
        if result['left_coef'] is not None and result['right_coef'] is not None:
            result['success'] = True
            
            # Sample points along x
            x_min = max(np.min(left_x), np.min(right_x))
            x_max = min(np.max(left_x), np.max(right_x))
            x_samples = np.linspace(x_min, x_max, 20)
            
            left_fit = np.polyval(result['left_coef'], x_samples)
            right_fit = np.polyval(result['right_coef'], x_samples)
            
            # Width metrics
            widths = left_fit - right_fit
            result['width_at_origin'] = np.polyval(result['left_coef'], 0) - np.polyval(result['right_coef'], 0)
            result['width_at_max_x'] = widths[-1] if len(widths) > 0 else 0.0
            result['width_std'] = np.std(widths)
            
            # Curvature (from second derivative)
            if self.params.polynomial_degree >= 2:
                # Second derivative coefficients
                left_d2 = np.polyder(result['left_coef'], 2)
                right_d2 = np.polyder(result['right_coef'], 2)
                
                left_curv = np.abs(np.polyval(left_d2, x_samples))
                right_curv = np.abs(np.polyval(right_d2, x_samples))
                
                result['max_curvature'] = max(np.max(left_curv), np.max(right_curv))
            
            # Smoothness score (inverse of curvature variation)
            if self.params.polynomial_degree >= 2:
                left_d1 = np.polyder(result['left_coef'], 1)
                right_d1 = np.polyder(result['right_coef'], 1)
                
                left_slope_var = np.var(np.polyval(left_d1, x_samples))
                right_slope_var = np.var(np.polyval(right_d1, x_samples))
                
                result['smoothness_score'] = 1.0 / (1.0 + left_slope_var + right_slope_var)
            else:
                # Linear is always smooth
                result['smoothness_score'] = 1.0
        
        return result


# ===========================
# Data Loading
# ===========================

class ValidationDataLoader:
    """Loads labeled validation data from NPZ files."""
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.good_dir = self.session_dir / 'good'
        self.bad_dir = self.session_dir / 'bad'
    
    def load_frames(self) -> List[Tuple[Path, str, dict]]:
        """Load all NPZ frames with labels."""
        frames = []
        
        # Good frames
        if self.good_dir.exists():
            for f in sorted(self.good_dir.glob('*.npz')):
                try:
                    data = dict(np.load(f, allow_pickle=True))
                    frames.append((f, 'good', data))
                except Exception as e:
                    print(f"Warning: Could not load {f}: {e}")
        
        # Bad frames
        if self.bad_dir.exists():
            for f in sorted(self.bad_dir.glob('*.npz')):
                try:
                    data = dict(np.load(f, allow_pickle=True))
                    frames.append((f, 'bad', data))
                except Exception as e:
                    print(f"Warning: Could not load {f}: {e}")
        
        return frames
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        good_count = len(list(self.good_dir.glob('*.npz'))) if self.good_dir.exists() else 0
        bad_count = len(list(self.bad_dir.glob('*.npz'))) if self.bad_dir.exists() else 0
        return {
            'good_frames': good_count,
            'bad_frames': bad_count,
            'total_frames': good_count + bad_count
        }


# ===========================
# MarkerArray Tester
# ===========================

class MarkerArrayTester:
    """Tests RANSAC parameters on validation data."""
    
    def __init__(self, session_dir: Path):
        self.loader = ValidationDataLoader(session_dir)
        self.frames = self.loader.load_frames()
    
    def _extract_boundaries(self, data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract left and right boundary points from road data.
        
        Simulates what Node 02 outputs: ring_boundaries with left/right points.
        """
        road_x = data.get('road_x', np.array([]))
        road_y = data.get('road_y', np.array([]))
        
        if len(road_x) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Simple extraction: left = positive y, right = negative y
        left_mask = road_y > 0.2
        right_mask = road_y < -0.2
        
        left_x = road_x[left_mask]
        left_y = road_y[left_mask]
        right_x = road_x[right_mask]
        right_y = road_y[right_mask]
        
        return left_x, left_y, right_x, right_y
    
    def test_params(self, params: CurveFittingParams) -> ParamsResult:
        """Test a set of RANSAC parameters on all frames."""
        fitter = MarkerArrayFitter(params)
        fit_results = []
        
        good_metrics = {'inlier_ratio': [], 'rmse': [], 'width': [], 'width_std': [], 'smoothness': []}
        bad_metrics = {'inlier_ratio': [], 'rmse': []}
        latencies = []
        success_count = 0
        
        for filepath, label, data in self.frames:
            left_x, left_y, right_x, right_y = self._extract_boundaries(data)
            
            if len(left_x) < 3 or len(right_x) < 3:
                continue
            
            # Time the fitting
            start_time = time.perf_counter()
            result = fitter.fit_boundaries(left_x, left_y, right_x, right_y)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(elapsed_ms)
            
            if result['success']:
                success_count += 1
                
                inlier_ratio = (result['left_inliers'] + result['right_inliers']) / (len(left_x) + len(right_x))
                avg_rmse = (result['left_rmse'] + result['right_rmse']) / 2
                
                fr = FitResult(
                    frame_file=str(filepath.name),
                    params_name=params.name,
                    label=label,
                    left_inliers=result['left_inliers'],
                    right_inliers=result['right_inliers'],
                    left_rmse=result['left_rmse'],
                    right_rmse=result['right_rmse'],
                    width_at_origin=result['width_at_origin'],
                    width_at_max_x=result['width_at_max_x'],
                    width_std=result['width_std'],
                    max_curvature=result['max_curvature'],
                    smoothness_score=result['smoothness_score'],
                    processing_time_ms=elapsed_ms
                )
                fit_results.append(fr)
                
                if label == 'good':
                    good_metrics['inlier_ratio'].append(inlier_ratio)
                    good_metrics['rmse'].append(avg_rmse)
                    good_metrics['width'].append(result['width_at_origin'])
                    good_metrics['width_std'].append(result['width_std'])
                    good_metrics['smoothness'].append(result['smoothness_score'])
                else:
                    bad_metrics['inlier_ratio'].append(inlier_ratio)
                    bad_metrics['rmse'].append(avg_rmse)
        
        # Aggregate results
        n_frames = len(self.frames)
        n_good = len([f for f in self.frames if f[1] == 'good'])
        n_bad = len([f for f in self.frames if f[1] == 'bad'])
        
        return ParamsResult(
            params=params,
            frames_processed=n_frames,
            good_frames=n_good,
            bad_frames=n_bad,
            avg_inlier_ratio=np.mean(good_metrics['inlier_ratio']) if good_metrics['inlier_ratio'] else 0.0,
            avg_rmse=np.mean(good_metrics['rmse']) if good_metrics['rmse'] else 0.0,
            avg_width=np.mean(good_metrics['width']) if good_metrics['width'] else 0.0,
            avg_width_std=np.mean(good_metrics['width_std']) if good_metrics['width_std'] else 0.0,
            avg_smoothness=np.mean(good_metrics['smoothness']) if good_metrics['smoothness'] else 0.0,
            bad_inlier_ratio=np.mean(bad_metrics['inlier_ratio']) if bad_metrics['inlier_ratio'] else 0.0,
            bad_rmse=np.mean(bad_metrics['rmse']) if bad_metrics['rmse'] else 0.0,
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
            success_rate=success_count / n_frames if n_frames > 0 else 0.0,
            fit_results=fit_results
        )


# ===========================
# Parameter Grid
# ===========================

def generate_param_grid() -> List[CurveFittingParams]:
    """Generate parameter combinations to test for Node 02 curve fitting."""
    params_list = []

    # ==================================================================
    # BASELINE: Current params.yaml values for ring_boundary_node
    # ==================================================================
    params_list.append(CurveFittingParams(
        name='baseline',
        fitting_method='ransac',
        polynomial_degree=2,
        ransac_residual_threshold=0.15,
        max_iterations=300,
        min_inlier_ratio=0.15,
        min_points_for_fitting=10,
        curve_resampling_spacing=0.5,
        enforce_constant_width=True,
        width_tolerance=0.1
    ))

    # ==================================================================
    # FITTING METHOD VARIATIONS
    # ==================================================================
    for method in ['ransac', 'spline', 'both']:
        if method != 'ransac':  # Skip baseline
            params_list.append(CurveFittingParams(
                name=f'method_{method}',
                fitting_method=method,
                polynomial_degree=2,
                ransac_residual_threshold=0.15,
                min_points_for_fitting=10,
                curve_resampling_spacing=0.5,
                enforce_constant_width=True,
                width_tolerance=0.1
            ))

    # ==================================================================
    # POLYNOMIAL DEGREE VARIATIONS
    # ==================================================================
    for deg in [1, 3]:  # Skip 2 (baseline)
        params_list.append(CurveFittingParams(
            name=f'degree_{deg}',
            fitting_method='ransac',
            polynomial_degree=deg,
            ransac_residual_threshold=0.15,
            min_points_for_fitting=10,
            curve_resampling_spacing=0.5,
            enforce_constant_width=True,
            width_tolerance=0.1
        ))

    # ==================================================================
    # RANSAC RESIDUAL THRESHOLD VARIATIONS
    # ==================================================================
    for thresh in [0.1, 0.2, 0.3]:  # Skip 0.15 (baseline)
        params_list.append(CurveFittingParams(
            name=f'thresh_{thresh}m',
            fitting_method='ransac',
            polynomial_degree=2,
            ransac_residual_threshold=thresh,
            min_points_for_fitting=10,
            curve_resampling_spacing=0.5,
            enforce_constant_width=True,
            width_tolerance=0.1
        ))

    # ==================================================================
    # MIN POINTS FOR FITTING VARIATIONS
    # ==================================================================
    for min_pts in [5, 15, 20]:  # Skip 10 (baseline)
        params_list.append(CurveFittingParams(
            name=f'minpts_{min_pts}',
            fitting_method='ransac',
            polynomial_degree=2,
            ransac_residual_threshold=0.15,
            min_points_for_fitting=min_pts,
            curve_resampling_spacing=0.5,
            enforce_constant_width=True,
            width_tolerance=0.1
        ))

    # ==================================================================
    # CURVE RESAMPLING SPACING VARIATIONS
    # ==================================================================
    for spacing in [0.3, 0.7, 1.0]:  # Skip 0.5 (baseline)
        params_list.append(CurveFittingParams(
            name=f'spacing_{spacing}m',
            fitting_method='ransac',
            polynomial_degree=2,
            ransac_residual_threshold=0.15,
            min_points_for_fitting=10,
            curve_resampling_spacing=spacing,
            enforce_constant_width=True,
            width_tolerance=0.1
        ))

    # ==================================================================
    # CONSTANT WIDTH CONSTRAINT VARIATIONS
    # ==================================================================
    params_list.append(CurveFittingParams(
        name='no_width_constraint',
        fitting_method='ransac',
        polynomial_degree=2,
        ransac_residual_threshold=0.15,
        min_points_for_fitting=10,
        curve_resampling_spacing=0.5,
        enforce_constant_width=False,
        width_tolerance=0.1
    ))

    # ==================================================================
    # WIDTH TOLERANCE VARIATIONS
    # ==================================================================
    for tol in [0.05, 0.15, 0.2]:  # Skip 0.1 (baseline)
        params_list.append(CurveFittingParams(
            name=f'width_tol_{int(tol*100)}pct',
            fitting_method='ransac',
            polynomial_degree=2,
            ransac_residual_threshold=0.15,
            min_points_for_fitting=10,
            curve_resampling_spacing=0.5,
            enforce_constant_width=True,
            width_tolerance=tol
        ))

    # ==================================================================
    # OPTIMIZED COMBINATIONS
    # ==================================================================
    # Tight: More strict for cleaner outputs
    params_list.append(CurveFittingParams(
        name='tight_fit',
        fitting_method='ransac',
        polynomial_degree=2,
        ransac_residual_threshold=0.1,
        min_points_for_fitting=15,
        curve_resampling_spacing=0.3,
        enforce_constant_width=True,
        width_tolerance=0.05
    ))

    # Relaxed: More permissive for noisy data
    params_list.append(CurveFittingParams(
        name='relaxed_fit',
        fitting_method='ransac',
        polynomial_degree=2,
        ransac_residual_threshold=0.25,
        min_points_for_fitting=5,
        curve_resampling_spacing=0.7,
        enforce_constant_width=False,
        width_tolerance=0.2
    ))

    return params_list


# ===========================
# Visualization
# ===========================

class VisualizationGenerator:
    """Generates comparison plots for parameter results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def plot_parameter_comparison(self, results: List[ParamsResult]):
        """Create comparison plots for all parameter sets."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RANSAC Parameter Comparison', fontsize=16, fontweight='bold')
        
        names = [r.params.name for r in results]
        x = np.arange(len(names))
        
        # Plot 1: Success rate
        ax = axes[0, 0]
        ax.bar(x, [r.success_rate for r in results], color='green', alpha=0.7)
        ax.set_ylabel('Success Rate')
        ax.set_title('Fit Success Rate (Higher = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.axhline(y=0.9, color='red', linestyle='--', label='Target 90%')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Plot 2: Inlier ratio
        ax = axes[0, 1]
        ax.bar(x, [r.avg_inlier_ratio for r in results], color='blue', alpha=0.7)
        ax.set_ylabel('Inlier Ratio')
        ax.set_title('Average Inlier Ratio (Higher = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylim([0, 1])
        
        # Plot 3: RMSE
        ax = axes[0, 2]
        ax.bar(x, [r.avg_rmse for r in results], color='orange', alpha=0.7)
        ax.set_ylabel('RMSE (m)')
        ax.set_title('Average RMSE (Lower = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.axhline(y=0.1, color='red', linestyle='--', label='Target 0.1m')
        ax.legend()
        
        # Plot 4: Width consistency
        ax = axes[1, 0]
        ax.bar(x, [r.avg_width_std for r in results], color='purple', alpha=0.7)
        ax.set_ylabel('Width Std (m)')
        ax.set_title('Width Consistency (Lower = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        
        # Plot 5: Smoothness
        ax = axes[1, 1]
        ax.bar(x, [r.avg_smoothness for r in results], color='cyan', alpha=0.7)
        ax.set_ylabel('Smoothness Score')
        ax.set_title('Smoothness Score (Higher = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylim([0, 1])
        
        # Plot 6: Latency
        ax = axes[1, 2]
        ax.bar(x, [r.avg_latency_ms for r in results], color='red', alpha=0.7)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Processing Time (Lower = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.axhline(y=10, color='green', linestyle='--', label='Budget 10ms')
        ax.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'ransac_parameter_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
    
    def plot_degree_comparison(self, results: List[ParamsResult]):
        """Compare different polynomial degrees."""
        import matplotlib.pyplot as plt
        
        # Filter by degree
        deg_results = {1: [], 2: [], 3: []}
        for r in results:
            if r.params.polynomial_degree in deg_results:
                deg_results[r.params.polynomial_degree].append(r)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Polynomial Degree Comparison', fontsize=14, fontweight='bold')
        
        degrees = [1, 2, 3]
        colors = ['green', 'blue', 'orange']
        
        for ax, (deg, color) in zip(axes, zip(degrees, colors)):
            if deg_results[deg]:
                r = deg_results[deg][0]  # First result for this degree
                metrics = ['Success\nRate', 'Inlier\nRatio', 'Smoothness']
                values = [r.success_rate, r.avg_inlier_ratio, r.avg_smoothness]
                ax.bar(metrics, values, color=color, alpha=0.7)
                ax.set_title(f'Degree {deg} ({"Linear" if deg==1 else "Quadratic" if deg==2 else "Cubic"})')
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = self.output_dir / 'polynomial_degree_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")


# ===========================
# Recommendations
# ===========================

def generate_recommendations(results: List[ParamsResult], output_dir: Path) -> str:
    """Generate recommendations based on test results."""
    
    # Score = success_rate * 0.3 + inlier_ratio * 0.2 + smoothness * 0.2 + (1-rmse/0.5) * 0.2 + (1-latency/20) * 0.1
    scored = []
    for r in results:
        if r.success_rate == 0:
            continue
        
        success_score = r.success_rate
        inlier_score = r.avg_inlier_ratio
        smooth_score = r.avg_smoothness
        rmse_score = max(0, 1.0 - r.avg_rmse / 0.5)
        latency_score = max(0, 1.0 - r.avg_latency_ms / 20)
        
        total = success_score * 0.3 + inlier_score * 0.2 + smooth_score * 0.2 + rmse_score * 0.2 + latency_score * 0.1
        scored.append((total, r))
    
    if not scored:
        return "No valid results to analyze."
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    
    md = f"""# MarkerArray RANSAC Parameter Recommendations

## Test Summary
- **Total frames tested**: {results[0].frames_processed if results else 0}
- **Good frames**: {results[0].good_frames if results else 0}
- **Bad frames**: {results[0].bad_frames if results else 0}

## Best Parameters: **{best.params.name}**

| Parameter | Value |
|-----------|-------|
| polynomial_degree | {best.params.polynomial_degree} |
| residual_threshold | {best.params.residual_threshold} m |
| max_iterations | {best.params.max_iterations} |
| min_inlier_ratio | {best.params.min_inlier_ratio} |
| target_road_width | {best.params.target_road_width} m |
| max_curvature | {best.params.max_curvature} |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Success Rate | {best.success_rate:.1%} |
| Inlier Ratio | {best.avg_inlier_ratio:.1%} |
| RMSE | {best.avg_rmse:.4f} m |
| Width | {best.avg_width:.2f} m |
| Width Consistency | {best.avg_width_std:.3f} m |
| Smoothness | {best.avg_smoothness:.3f} |
| Latency | {best.avg_latency_ms:.2f} ms |

## params.yaml Update (ring_boundary_node section)

```yaml
ring_boundary_node:
  # Curve fitting parameters
  enable_curve_fitting: true
  fitting_method: '{best.params.fitting_method}'
  polynomial_degree: {best.params.polynomial_degree}
  ransac_residual_threshold: {best.params.ransac_residual_threshold}
  min_points_for_fitting: {best.params.min_points_for_fitting}
  curve_resampling_spacing: {best.params.curve_resampling_spacing}
  # Geometric constraints
  enforce_constant_width: {str(best.params.enforce_constant_width).lower()}
  width_tolerance: {best.params.width_tolerance}
```

## Top 5 Parameter Sets

| Rank | Name | Success | Inlier | RMSE | Latency |
|------|------|---------|--------|------|---------|
"""
    
    for i, (score, r) in enumerate(scored[:5], 1):
        md += f"| {i} | {r.params.name} | {r.success_rate:.1%} | {r.avg_inlier_ratio:.1%} | {r.avg_rmse:.4f}m | {r.avg_latency_ms:.1f}ms |\n"
    
    md += f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    output_path = output_dir / 'ransac_recommendations.md'
    with open(output_path, 'w') as f:
        f.write(md)
    print(f"  ✓ Saved: {output_path}")
    
    return md


# ===========================
# Find Latest Session
# ===========================

def find_latest_session() -> Path:
    """Find the most recent validation session."""
    output_dir = Path('/home/ws/tools/validation_output')
    if not output_dir.exists():
        return None
    
    sessions = sorted(output_dir.glob('session_*'), reverse=True)
    for session in sessions:
        # Prefer sessions with labeled frames
        good_dir = session / 'good'
        bad_dir = session / 'bad'
        if (good_dir.exists() and list(good_dir.glob('*.npz'))) or \
           (bad_dir.exists() and list(bad_dir.glob('*.npz'))):
            return session
    
    return sessions[0] if sessions else None


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='MarkerArray RANSAC Parameter Tuner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 marker_array_tuner.py --latest
    python3 marker_array_tuner.py tools/validation_output/session_XXXX/
"""
    )
    parser.add_argument('session_dir', nargs='?', help='Path to validation session directory')
    parser.add_argument('--latest', '-l', action='store_true', help='Use most recent session')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Find session
    if args.latest:
        session_dir = find_latest_session()
        if not session_dir:
            print("❌ No validation sessions found!")
            print("   Run 'make validate-detection' first to label frames.")
            sys.exit(1)
    elif args.session_dir:
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"❌ Session not found: {session_dir}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("MARKERARRAY RANSAC PARAMETER TUNER")
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
    tester = MarkerArrayTester(session_dir)
    
    # Generate parameter grid
    params_grid = generate_param_grid()
    print(f"\nTesting {len(params_grid)} parameter combinations...")
    
    # Test all parameters
    results = []
    for i, params in enumerate(params_grid, 1):
        print(f"  [{i}/{len(params_grid)}] {params.name}...", end='', flush=True)
        result = tester.test_params(params)
        results.append(result)
        print(f" ✓ (success={result.success_rate:.1%}, rmse={result.avg_rmse:.4f}m)")
    
    # Output directory
    output_dir = session_dir / 'ransac_tuning_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_results = [r.to_dict() for r in results]
    json_path = output_dir / 'ransac_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Saved: {json_path}")
    
    # Generate visualizations
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            print("\nGenerating plots...")
            viz = VisualizationGenerator(output_dir)
            viz.plot_parameter_comparison(results)
            viz.plot_degree_comparison(results)
        except ImportError:
            print("\n⚠ matplotlib not available, skipping plots")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    generate_recommendations(results, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Find best
    valid_results = [r for r in results if r.success_rate > 0]
    if valid_results:
        best = max(valid_results, key=lambda r: 
                   r.success_rate * 0.3 + r.avg_inlier_ratio * 0.2 + 
                   r.avg_smoothness * 0.2 + max(0, 1.0 - r.avg_rmse / 0.5) * 0.2)
        
        print(f"\n🏆 Best Parameters: {best.params.name}")
        print(f"   polynomial_degree:  {best.params.polynomial_degree}")
        print(f"   residual_threshold: {best.params.residual_threshold} m")
        print(f"   max_iterations:     {best.params.max_iterations}")
        print(f"   min_inlier_ratio:   {best.params.min_inlier_ratio}")
        print(f"   target_road_width:  {best.params.target_road_width} m")
        print(f"\n   Success Rate: {best.success_rate:.1%}")
        print(f"   Inlier Ratio: {best.avg_inlier_ratio:.1%}")
        print(f"   RMSE: {best.avg_rmse:.4f} m")
        print(f"   Width: {best.avg_width:.2f} m")
        print(f"   Smoothness: {best.avg_smoothness:.3f}")
        print(f"   Latency: {best.avg_latency_ms:.2f} ms")
    
    print(f"\n📁 Output: {output_dir}")
    print(f"   - ransac_results.json")
    print(f"   - ransac_recommendations.md")
    if not args.no_plots:
        print(f"   - ransac_parameter_comparison.png")
        print(f"   - polynomial_degree_comparison.png")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
