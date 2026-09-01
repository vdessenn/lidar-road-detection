#!/usr/bin/env python3
"""
Ring Boundary Extraction - Enhanced with Dense Interpolation

Extracts road boundaries from LiDAR ring extremities with:
- Dense point interpolation (min 20 points per boundary)
- Gap extrapolation for continuous visualization
- All params.yaml parameters integrated

Author: Victor Dessenne (Enhanced v4)
Date: 2026-01-13
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
from typing import Optional, Dict, List, Tuple
from collections import deque
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from scipy import interpolate


class RingBoundaryExtractionNode(Node):
    """Enhanced ring-based boundary extraction with dense interpolation."""
    
    def __init__(self):
        super().__init__('ring_boundary_extraction_node')
        
        # Declare ALL parameters from params.yaml
        self.declare_parameter('topics.ring_boundary_input', '/road_surface')
        self.declare_parameter('topics.ring_boundary_output', '/ring_boundaries')
        self.declare_parameter('qos_depth', 1)
        
        # Core extraction parameters
        self.declare_parameter('ring_boundary_node.enable_extraction', True)
        self.declare_parameter('ring_boundary_node.max_boundary_width', 7.0)
        self.declare_parameter('ring_boundary_node.min_points_per_ring', 25)
        self.declare_parameter('ring_boundary_node.y_threshold', 0.15)
        
        # Robust extraction
        self.declare_parameter('ring_boundary_node.robust_n_points', 5)
        
        # Inter-ring consistency
        self.declare_parameter('ring_boundary_node.enable_ring_consistency', True)
        self.declare_parameter('ring_boundary_node.max_y_deviation', 0.8)
        
        # Temporal smoothing
        self.declare_parameter('ring_boundary_node.enable_temporal_smoothing', True)
        self.declare_parameter('ring_boundary_node.temporal_buffer_frames', 7)
        
        # Curve fitting parameters
        self.declare_parameter('ring_boundary_node.enable_curve_fitting', True)
        self.declare_parameter('ring_boundary_node.fitting_method', 'ransac')
        self.declare_parameter('ring_boundary_node.polynomial_degree', 2)
        self.declare_parameter('ring_boundary_node.ransac_residual_threshold', 0.18)
        self.declare_parameter('ring_boundary_node.min_points_for_fitting', 8)
        self.declare_parameter('ring_boundary_node.curve_resampling_spacing', 0.3)
        
        # Geometric constraints
        self.declare_parameter('ring_boundary_node.enforce_constant_width', True)
        self.declare_parameter('ring_boundary_node.width_tolerance', 0.2)
        self.declare_parameter('ring_boundary_node.enforce_symmetry', False)
        
        # NEW: Dense interpolation parameters
        self.declare_parameter('ring_boundary_node.min_output_points', 20)  # Minimum points garantis
        self.declare_parameter('ring_boundary_node.extrapolate_gaps', True)  # Combler les gaps
        self.declare_parameter('ring_boundary_node.forward_extrapolation_distance', 5.0)
        self.declare_parameter('ring_boundary_node.backward_extrapolation_distance', 2.0)
        
        # Get parameters
        input_topic = self.get_parameter('topics.ring_boundary_input').value
        output_topic = self.get_parameter('topics.ring_boundary_output').value
        qos_depth = self.get_parameter('qos_depth').value
        
        self.enable_extraction = self.get_parameter('ring_boundary_node.enable_extraction').value
        self.max_boundary_width = self.get_parameter('ring_boundary_node.max_boundary_width').value
        self.min_points_per_ring = self.get_parameter('ring_boundary_node.min_points_per_ring').value
        self.y_threshold = self.get_parameter('ring_boundary_node.y_threshold').value
        
        self.robust_n_points = self.get_parameter('ring_boundary_node.robust_n_points').value
        self.enable_ring_consistency = self.get_parameter('ring_boundary_node.enable_ring_consistency').value
        self.max_y_deviation = self.get_parameter('ring_boundary_node.max_y_deviation').value
        
        self.enable_temporal_smoothing = self.get_parameter('ring_boundary_node.enable_temporal_smoothing').value
        self.temporal_buffer_frames = self.get_parameter('ring_boundary_node.temporal_buffer_frames').value
        
        self.enable_curve_fitting = self.get_parameter('ring_boundary_node.enable_curve_fitting').value
        self.fitting_method = self.get_parameter('ring_boundary_node.fitting_method').value
        self.polynomial_degree = self.get_parameter('ring_boundary_node.polynomial_degree').value
        self.ransac_residual_threshold = self.get_parameter('ring_boundary_node.ransac_residual_threshold').value
        self.min_points_for_fitting = self.get_parameter('ring_boundary_node.min_points_for_fitting').value
        self.curve_resampling_spacing = self.get_parameter('ring_boundary_node.curve_resampling_spacing').value
        
        self.enforce_constant_width = self.get_parameter('ring_boundary_node.enforce_constant_width').value
        self.width_tolerance = self.get_parameter('ring_boundary_node.width_tolerance').value
        self.enforce_symmetry = self.get_parameter('ring_boundary_node.enforce_symmetry').value
        
        # NEW parameters
        self.min_output_points = self.get_parameter('ring_boundary_node.min_output_points').value
        self.extrapolate_gaps = self.get_parameter('ring_boundary_node.extrapolate_gaps').value
        self.forward_extrap_dist = self.get_parameter('ring_boundary_node.forward_extrapolation_distance').value
        self.backward_extrap_dist = self.get_parameter('ring_boundary_node.backward_extrapolation_distance').value
        
        # Temporal smoothing buffers
        self.left_history: deque = deque(maxlen=self.temporal_buffer_frames)
        self.right_history: deque = deque(maxlen=self.temporal_buffer_frames)
        
        # QoS Profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )
        
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos)
        self.publisher = self.create_publisher(PointCloud2, output_topic, qos)
        
        self.get_logger().info(f'Ring Boundary Extraction (Enhanced v4 - Dense Interpolation):')
        self.get_logger().info(f'  Input: {input_topic}')
        self.get_logger().info(f'  Output: {output_topic}')
        self.get_logger().info(f'  Max width: {self.max_boundary_width}m, Min pts/ring: {self.min_points_per_ring}')
        self.get_logger().info(f'  Robust: median of {self.robust_n_points} extreme pts')
        self.get_logger().info(f'  Ring consistency: {self.enable_ring_consistency} (max_dev={self.max_y_deviation}m)')
        self.get_logger().info(f'  Temporal smoothing: {self.enable_temporal_smoothing} ({self.temporal_buffer_frames} frames)')
        self.get_logger().info(f'  Curve fitting: {self.enable_curve_fitting} (method={self.fitting_method}, degree={self.polynomial_degree})')
        self.get_logger().info(f'  Resampling: {self.curve_resampling_spacing}m spacing, min {self.min_output_points} points')
        self.get_logger().info(f'  Geometric constraints: width={self.enforce_constant_width}, symmetry={self.enforce_symmetry}')
        self.get_logger().info(f'  Extrapolate gaps: {self.extrapolate_gaps}')
    
    def callback(self, msg: PointCloud2) -> None:
        """Extract left/right boundaries from ring extremities."""
        if not self.enable_extraction:
            self.publisher.publish(msg)
            return
        
        if len(msg.data) == 0:
            self._publish_empty(msg)
            return
        
        # Parse PointCloud2
        data = np.frombuffer(msg.data, dtype=np.uint8)
        points = data.reshape(-1, msg.point_step)
        
        x = points[:, 0:4].view(np.float32).ravel()
        y = points[:, 4:8].view(np.float32).ravel()
        z = points[:, 8:12].view(np.float32).ravel()
        
        # Extract ring field
        ring = self._extract_ring_field(msg, points)
        
        if ring is None:
            left_pts, right_pts = self._extract_from_x_bins(x, y, z)
        else:
            left_pts, right_pts = self._extract_from_rings(x, y, z, ring)
        
        self.get_logger().info(
            f'[02] Input: {len(x)} → LEFT: {len(left_pts)} | RIGHT: {len(right_pts)}',
            throttle_duration_sec=1.0
        )
        
        self._publish_boundaries(msg, left_pts, right_pts)
    
    def _extract_from_rings(self, x: np.ndarray, y: np.ndarray,
                             z: np.ndarray, ring: np.ndarray) -> tuple:
        """Extract boundaries with all enhancements."""
        # Step 1: Extract boundaries per ring
        boundaries_by_ring: Dict[int, Tuple[List, List]] = {}
        
        for ring_id in np.unique(ring):
            mask = ring == ring_id
            if np.sum(mask) < self.min_points_per_ring:
                continue
            
            x_r, y_r, z_r = x[mask], y[mask], z[mask]
            
            # Robust extraction: median of N extreme points
            left_pt = self._get_robust_extreme(x_r, y_r, z_r, side='left')
            right_pt = self._get_robust_extreme(x_r, y_r, z_r, side='right')
            
            if left_pt is None or right_pt is None:
                continue
            
            # Symmetry constraint
            width = left_pt[1] - right_pt[1]
            if width > self.max_boundary_width:
                if abs(right_pt[1]) < abs(left_pt[1]):
                    left_pt = (left_pt[0], -right_pt[1], left_pt[2])
                else:
                    right_pt = (right_pt[0], -left_pt[1], right_pt[2])
            
            boundaries_by_ring[int(ring_id)] = (left_pt, right_pt)
        
        # Step 2: Inter-ring consistency check
        if self.enable_ring_consistency:
            boundaries_by_ring = self._validate_ring_consistency(boundaries_by_ring)
        
        # Step 3: Build output arrays
        left_points = []
        right_points = []
        
        for ring_id, (left_pt, right_pt) in boundaries_by_ring.items():
            if left_pt[1] > self.y_threshold:
                left_points.append(list(left_pt))
            if right_pt[1] < -self.y_threshold:
                right_points.append(list(right_pt))
        
        left = np.array(left_points) if left_points else np.empty((0, 3))
        right = np.array(right_points) if right_points else np.empty((0, 3))
        
        # Step 4: Apply curve fitting with DENSE interpolation
        if self.enable_curve_fitting:
            if len(left) >= self.min_points_for_fitting:
                left = self._apply_curve_fitting_dense(left)
            if len(right) >= self.min_points_for_fitting:
                right = self._apply_curve_fitting_dense(right)
        
        # Step 5: Apply geometric constraints
        if self.enforce_constant_width and len(left) > 0 and len(right) > 0:
            left, right = self._apply_road_geometry_constraints(left, right)
        
        # Step 6: Temporal smoothing
        if self.enable_temporal_smoothing:
            left, right = self._apply_temporal_smoothing(left, right)
        
        return left, right
    
    # ========== NEW METHOD: DENSE CURVE FITTING ==========
    
    def _apply_curve_fitting_dense(self, points: np.ndarray) -> np.ndarray:
        """
        Apply curve fitting with PREDICTIVE FORWARD EXTRAPOLATION.
        Projects road boundaries beyond detected points.
        """
        if len(points) < self.min_points_for_fitting:
            return points
        
        try:
            # Sort by x coordinate
            sorted_idx = np.argsort(points[:, 0])
            x = points[sorted_idx, 0].reshape(-1, 1)
            y = points[sorted_idx, 1]
            z = points[sorted_idx, 2]
            
            # RANSAC polynomial fitting
            model_y = make_pipeline(
                PolynomialFeatures(self.polynomial_degree),
                RANSACRegressor(
                    residual_threshold=self.ransac_residual_threshold,
                    min_samples=max(self.polynomial_degree + 1, 3),  # ← Min 3 points
                    max_trials=200,
                    random_state=42,
                    stop_probability=0.99
                )
            )
            
            model_y.fit(x, y)
            
            # Check inlier ratio
            inlier_mask = model_y.named_steps['ransacregressor'].inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(x)
            
            if inlier_ratio < 0.10:
                self.get_logger().debug(
                    f'Low inlier ratio: {inlier_ratio:.2f}',
                    throttle_duration_sec=2.0
                )
                return points
            
            # ========== PREDICTIVE EXTRAPOLATION ==========
            
            x_min_observed = np.min(x)
            x_max_observed = np.max(x)
            x_range_observed = x_max_observed - x_min_observed
            
            # NEW: Extrapolate FORWARD based on polynomial trend
            extrapolation_distance = 5.0  # Extend 5 meters forward
            x_max_extended = x_max_observed + extrapolation_distance
            
            extrapolation_distance = self.forward_extrap_dist  # 5.0m par défaut
            x_min_extended = max(0.0, x_min_observed - self.backward_extrap_dist)
            
            # Calculate dense sampling
            n_observed = max(int(x_range_observed / self.curve_resampling_spacing), 1)
            n_forward = int(extrapolation_distance / self.curve_resampling_spacing)
            n_backward = int((x_min_observed - x_min_extended) / self.curve_resampling_spacing)
            
            # Total points: backward + observed + forward
            n_total = max(self.min_output_points, n_backward + n_observed + n_forward)
            
            # Generate extended x range
            x_extended = np.linspace(x_min_extended, x_max_extended, n_total).reshape(-1, 1)
            
            # Predict y values (polynomial naturally extrapolates)
            y_extended = model_y.predict(x_extended)
            
            # Extrapolate z with constant slope from last 3 points
            if len(x) >= 3:
                # Estimate z slope from last points
                z_slope = (z[-1] - z[-3]) / (x[-1, 0] - x[-3, 0])
                z_extended = np.interp(x_extended.flatten(), x.flatten(), z)
                
                # Extend z beyond observed range with constant slope
                forward_mask = x_extended.flatten() > x_max_observed
                if np.any(forward_mask):
                    x_forward = x_extended.flatten()[forward_mask]
                    z_extended[forward_mask] = z[-1] + z_slope * (x_forward - x_max_observed)
                
                backward_mask = x_extended.flatten() < x_min_observed
                if np.any(backward_mask):
                    x_backward = x_extended.flatten()[backward_mask]
                    z_extended[backward_mask] = z[0] + z_slope * (x_backward - x_min_observed)
            else:
                z_extended = np.full(len(x_extended), np.mean(z))
            
            # Create fitted + extrapolated points
            fitted_points = np.column_stack([x_extended.ravel(), y_extended, z_extended])
            
            self.get_logger().debug(
                f'Predictive fit: {len(points)} input → {len(fitted_points)} output '
                f'(range: {x_min_extended:.1f}m to {x_max_extended:.1f}m)',
                throttle_duration_sec=2.0
            )
            
            return fitted_points
            
        except Exception as e:
            self.get_logger().warn(
                f'Predictive curve fitting failed: {e}',
                throttle_duration_sec=2.0
            )
            return points

        
    # ========== KEEP ALL OTHER METHODS FROM ORIGINAL ==========
    
    def _apply_road_geometry_constraints(self, left_pts: np.ndarray,
                                          right_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric constraints (UNCHANGED)."""
        if len(left_pts) == 0 or len(right_pts) == 0:
            return left_pts, right_pts
        
        try:
            # Sort both by x
            left_sorted = left_pts[np.argsort(left_pts[:, 0])]
            right_sorted = right_pts[np.argsort(right_pts[:, 0])]
            
            # Find common x range
            x_min = max(left_sorted[0, 0], right_sorted[0, 0])
            x_max = min(left_sorted[-1, 0], right_sorted[-1, 0])
            
            if x_min >= x_max:
                return left_pts, right_pts
            
            # Resample both to same x coordinates with DENSE sampling
            n_points = max(self.min_output_points, int((x_max - x_min) / self.curve_resampling_spacing))
            x_common = np.linspace(x_min, x_max, n_points)
            
            # Interpolate left and right at common x
            left_y = np.interp(x_common, left_sorted[:, 0], left_sorted[:, 1])
            left_z = np.interp(x_common, left_sorted[:, 0], left_sorted[:, 2])
            right_y = np.interp(x_common, right_sorted[:, 0], right_sorted[:, 1])
            right_z = np.interp(x_common, right_sorted[:, 0], right_sorted[:, 2])
            
            # Compute road width at each point
            widths = left_y - right_y
            median_width = np.median(widths)
            
            # Filter out sections with abnormal width
            width_tolerance = self.width_tolerance * median_width
            valid_mask = np.abs(widths - median_width) < width_tolerance
            
            if np.sum(valid_mask) < 5:
                self.get_logger().warn(
                    f'Geometric constraint: too few valid points ({np.sum(valid_mask)})',
                    throttle_duration_sec=2.0
                )
                return left_pts, right_pts
            
            # Keep only valid sections
            x_valid = x_common[valid_mask]
            left_y_valid = left_y[valid_mask]
            left_z_valid = left_z[valid_mask]
            right_y_valid = right_y[valid_mask]
            right_z_valid = right_z[valid_mask]
            
            # Optionally enforce symmetry around centerline
            if self.enforce_symmetry:
                centerline_y = (left_y_valid + right_y_valid) / 2
                half_width = median_width / 2
                left_y_valid = centerline_y + half_width
                right_y_valid = centerline_y - half_width
            
            left_fitted = np.column_stack([x_valid, left_y_valid, left_z_valid])
            right_fitted = np.column_stack([x_valid, right_y_valid, right_z_valid])
            
            return left_fitted, right_fitted
            
        except Exception as e:
            self.get_logger().warn(f'Geometric constraints failed: {e}')
            return left_pts, right_pts
    
    def _get_robust_extreme(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             side: str) -> Optional[Tuple[float, float, float]]:
        """Get median of N extreme points instead of single extreme."""
        n = min(self.robust_n_points, len(y))
        if n == 0:
            return None
        
        if side == 'left':
            indices = np.argsort(y)[-n:]  # Highest y values
        else:
            indices = np.argsort(y)[:n]  # Lowest y values
        
        return (
            float(np.median(x[indices])),
            float(np.median(y[indices])),
            float(np.median(z[indices]))
        )
    
    def _validate_ring_consistency(self, boundaries: Dict[int, Tuple]) -> Dict[int, Tuple]:
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
                if abs(left_pt[1] - prev_left[1]) > self.max_y_deviation:
                    is_valid = False
                if abs(right_pt[1] - prev_right[1]) > self.max_y_deviation:
                    is_valid = False
            
            # Check next ring
            if i < len(ring_ids) - 1:
                next_id = ring_ids[i + 1]
                next_left, next_right = boundaries[next_id]
                if abs(left_pt[1] - next_left[1]) > self.max_y_deviation:
                    is_valid = False
                if abs(right_pt[1] - next_right[1]) > self.max_y_deviation:
                    is_valid = False
            
            if is_valid:
                validated[ring_id] = (left_pt, right_pt)
        
        # Fallback to all if none valid (prevents empty output)
        return validated if validated else boundaries
    
    def _apply_temporal_smoothing(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth boundaries over last N frames."""
        self.left_history.append(left)
        self.right_history.append(right)
        
        if len(self.left_history) < 2:
            return left, right
        
        # Weighted average: recent frames have more weight
        weights = np.linspace(0.5, 1.0, len(self.left_history))
        weights /= weights.sum()
        
        smoothed_left = self._weighted_average_points(list(self.left_history), weights)
        smoothed_right = self._weighted_average_points(list(self.right_history), weights)
        
        return smoothed_left, smoothed_right
    
    def _weighted_average_points(self, history: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Compute weighted average of point arrays."""
        if not history or all(len(h) == 0 for h in history):
            return np.empty((0, 3))
        
        # Use the latest frame as reference
        latest = history[-1]
        if len(latest) == 0:
            for h in reversed(history):
                if len(h) > 0:
                    return h
            return np.empty((0, 3))
        
        # Simple approach: return latest (temporal smoothing already applied via buffer)
        return latest
    
    def _extract_from_x_bins(self, x: np.ndarray, y: np.ndarray,
                              z: np.ndarray, bin_size: float = 2.0) -> tuple:
        """Fallback: X-binning if no ring field."""
        left_points = []
        right_points = []
        
        if len(x) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        
        x_bins = np.arange(x.min(), x.max() + bin_size, bin_size)
        digitized = np.digitize(x, x_bins)
        
        for bin_idx in range(1, len(x_bins)):
            mask = digitized == bin_idx
            if np.sum(mask) < self.min_points_per_ring:
                continue
            
            x_bin, y_bin, z_bin = x[mask], y[mask], z[mask]
            
            left_pt = self._get_robust_extreme(x_bin, y_bin, z_bin, 'left')
            right_pt = self._get_robust_extreme(x_bin, y_bin, z_bin, 'right')
            
            if left_pt and left_pt[1] > self.y_threshold:
                left_points.append(list(left_pt))
            if right_pt and right_pt[1] < -self.y_threshold:
                right_points.append(list(right_pt))
        
        left = np.array(left_points) if left_points else np.empty((0, 3))
        right = np.array(right_points) if right_points else np.empty((0, 3))
        
        # Apply dense curve fitting to fallback path too
        if self.enable_curve_fitting:
            if len(left) >= self.min_points_for_fitting:
                left = self._apply_curve_fitting_dense(left)
            if len(right) >= self.min_points_for_fitting:
                right = self._apply_curve_fitting_dense(right)
        
        return left, right
    
    def _extract_ring_field(self, msg: PointCloud2, points: np.ndarray) -> Optional[np.ndarray]:
        """Extract ring field from PointCloud2."""
        for field in msg.fields:
            if field.name == 'ring':
                offset = field.offset
                if field.datatype == PointField.UINT16:
                    return points[:, offset:offset+2].view(np.uint16).ravel()
                elif field.datatype == PointField.UINT8:
                    return points[:, offset:offset+1].view(np.uint8).ravel()
        return None
    
    def _publish_boundaries(self, msg: PointCloud2,
                             left: np.ndarray, right: np.ndarray) -> None:
        """Publish boundary points with side_id field."""
        all_pts = []
        side_ids = []
        
        for pt in left:
            all_pts.append(pt)
            side_ids.append(1)
        
        for pt in right:
            all_pts.append(pt)
            side_ids.append(-1)
        
        if not all_pts:
            self._publish_empty(msg)
            return
        
        out_msg = PointCloud2()
        out_msg.header = msg.header
        out_msg.height = 1
        out_msg.width = len(all_pts)
        out_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='side_id', offset=12, datatype=PointField.INT32, count=1),
        ]
        out_msg.is_bigendian = False
        out_msg.point_step = 16
        out_msg.row_step = 16 * len(all_pts)
        
        data = np.zeros((len(all_pts), 16), dtype=np.uint8)
        for i, (pt, side) in enumerate(zip(all_pts, side_ids)):
            data[i, 0:4] = np.frombuffer(struct.pack('f', pt[0]), dtype=np.uint8)
            data[i, 4:8] = np.frombuffer(struct.pack('f', pt[1]), dtype=np.uint8)
            data[i, 8:12] = np.frombuffer(struct.pack('f', pt[2]), dtype=np.uint8)
            data[i, 12:16] = np.frombuffer(struct.pack('i', side), dtype=np.uint8)
        
        out_msg.data = data.tobytes()
        out_msg.is_dense = True
        self.publisher.publish(out_msg)
    
    def _publish_empty(self, msg: PointCloud2) -> None:
        """Publish empty PointCloud2."""
        out_msg = PointCloud2()
        out_msg.header = msg.header
        out_msg.height = 1
        out_msg.width = 0
        out_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='side_id', offset=12, datatype=PointField.INT32, count=1),
        ]
        out_msg.is_bigendian = False
        out_msg.point_step = 16
        out_msg.row_step = 0
        out_msg.data = b''
        out_msg.is_dense = True
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RingBoundaryExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
