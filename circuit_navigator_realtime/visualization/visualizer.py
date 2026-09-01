#!/usr/bin/env python3
"""
Real-Time Visualizer - Shows lidar edges, computed centerline, and comparison.

Displays:
- Track background (navigable area)
- Sensor range circle
- Detected left/right edge points
- Computed centerline (green)
- Reference centerline (blue dashed)
- Vehicle position and heading
- Quality metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
from typing import Tuple, Optional
from dataclasses import dataclass

from ..sensor.lidar_simulator import LidarScan
from ..evaluation.centerline_compare import EvaluationMetrics


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    vehicle_length: float = 4.0
    vehicle_width: float = 2.0
    trail_length: int = 200
    sensor_range: float = 40.0
    figsize: Tuple[int, int] = (16, 8)
    update_interval_ms: int = 50


class RealtimeVisualizer:
    """
    Real-time visualization for edge-based centerline navigation.

    Shows lidar edges, computed path, reference path, and quality metrics.
    """

    def __init__(
        self,
        navigable_grid: np.ndarray,
        reference_centerline: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        config: Optional[VisualizerConfig] = None
    ):
        """
        Initialize visualizer.

        Args:
            navigable_grid: Binary grid (True = navigable)
            reference_centerline: Pre-computed reference centerline
            origin: World origin of grid
            resolution: Grid resolution (m/cell)
            config: Visualization configuration
        """
        self.grid = navigable_grid
        self.reference = reference_centerline
        self.origin = origin
        self.resolution = resolution
        self.config = config or VisualizerConfig()

        # Trail history
        self.trail_x = []
        self.trail_y = []

        # Setup matplotlib
        plt.ion()
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure."""
        rows, cols = self.grid.shape

        self.extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        self.fig, (self.ax_map, self.ax_info) = plt.subplots(
            1, 2,
            figsize=self.config.figsize,
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # Track background
        track_image = np.zeros((*self.grid.shape, 3))
        track_image[self.grid] = [0.85, 0.85, 0.85]
        track_image[~self.grid] = [0.3, 0.6, 0.3]

        self.ax_map.imshow(track_image, origin='lower', extent=self.extent, aspect='equal')

        # Reference centerline (blue dashed)
        self.ax_map.plot(
            self.reference[:, 0], self.reference[:, 1],
            'b--', linewidth=1.5, alpha=0.6, label='Reference'
        )

        # Dynamic elements
        # Sensor range circle
        self.sensor_circle = patches.Circle(
            (0, 0), self.config.sensor_range,
            fill=False, linestyle=':', color='cyan', linewidth=1, alpha=0.5
        )
        self.ax_map.add_patch(self.sensor_circle)

        # Left edge points (orange)
        self.left_scatter = self.ax_map.scatter(
            [], [], c='orange', s=10, alpha=0.8, label='Left edge'
        )

        # Right edge points (yellow)
        self.right_scatter = self.ax_map.scatter(
            [], [], c='yellow', s=10, alpha=0.8, label='Right edge'
        )

        # Computed centerline (green solid)
        self.computed_line, = self.ax_map.plot(
            [], [], 'g-', linewidth=2.5, label='Computed'
        )

        # Target point (red star)
        self.target_point, = self.ax_map.plot(
            [], [], 'r*', markersize=15, label='Target'
        )

        # Trail
        self.trail_line, = self.ax_map.plot(
            [], [], 'b-', linewidth=1, alpha=0.4
        )

        # Vehicle
        self.vehicle_patch = patches.Rectangle(
            (0, 0),
            self.config.vehicle_length,
            self.config.vehicle_width,
            linewidth=2,
            edgecolor='blue',
            facecolor='lightblue',
            alpha=0.8
        )
        self.ax_map.add_patch(self.vehicle_patch)

        # Heading arrow
        self.heading_arrow = self.ax_map.annotate(
            '',
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )

        self.ax_map.set_xlabel('X (m)')
        self.ax_map.set_ylabel('Y (m)')
        self.ax_map.set_title('Real-Time Centerline Navigation')
        self.ax_map.legend(loc='upper right', fontsize=8)
        self.ax_map.set_aspect('equal')

        # Info panel
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        self.ax_info.set_title('Status & Metrics')

        self.info_text = self.ax_info.text(
            0.05, 0.95, '', fontsize=10, family='monospace',
            verticalalignment='top', transform=self.ax_info.transAxes
        )

        plt.tight_layout()

    def update(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        car_speed: float,
        steering: float,
        lidar_scan: LidarScan,
        computed_path: np.ndarray,
        target_point: Tuple[float, float],
        metrics: Optional[EvaluationMetrics] = None,
        distance_traveled: float = 0.0
    ):
        """
        Update visualization.

        Args:
            car_x, car_y: Vehicle position
            car_yaw: Vehicle heading
            car_speed: Vehicle speed
            steering: Steering angle
            lidar_scan: Current lidar scan
            computed_path: Computed centerline
            target_point: Pure Pursuit target
            metrics: Quality metrics (optional)
            distance_traveled: Total distance traveled
        """
        # Update trail
        self.trail_x.append(car_x)
        self.trail_y.append(car_y)
        if len(self.trail_x) > self.config.trail_length:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        self.trail_line.set_data(self.trail_x, self.trail_y)

        # Update sensor circle
        self.sensor_circle.center = (car_x, car_y)

        # Update edge points
        if len(lidar_scan.left_edge) > 0:
            self.left_scatter.set_offsets(lidar_scan.left_edge)
        else:
            self.left_scatter.set_offsets(np.empty((0, 2)))

        if len(lidar_scan.right_edge) > 0:
            self.right_scatter.set_offsets(lidar_scan.right_edge)
        else:
            self.right_scatter.set_offsets(np.empty((0, 2)))

        # Update computed centerline
        if len(computed_path) > 0:
            self.computed_line.set_data(computed_path[:, 0], computed_path[:, 1])
        else:
            self.computed_line.set_data([], [])

        # Update target point
        self.target_point.set_data([target_point[0]], [target_point[1]])

        # Update vehicle
        transform = Affine2D()
        transform.translate(-self.config.vehicle_length / 4, -self.config.vehicle_width / 2)
        transform.rotate(car_yaw)
        transform.translate(car_x, car_y)
        transform += self.ax_map.transData
        self.vehicle_patch.set_transform(transform)

        # Update heading arrow
        arrow_length = 3.0
        arrow_end_x = car_x + arrow_length * np.cos(car_yaw)
        arrow_end_y = car_y + arrow_length * np.sin(car_yaw)
        self.heading_arrow.set_position((car_x, car_y))
        self.heading_arrow.xy = (arrow_end_x, arrow_end_y)

        # Update info panel
        info_str = self._format_info(
            car_x, car_y, car_yaw, car_speed, steering,
            lidar_scan, computed_path, metrics, distance_traveled
        )
        self.info_text.set_text(info_str)

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _format_info(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        car_speed: float,
        steering: float,
        lidar_scan: LidarScan,
        computed_path: np.ndarray,
        metrics: Optional[EvaluationMetrics],
        distance_traveled: float
    ) -> str:
        """Format info panel text."""
        lines = [
            "VEHICLE STATE",
            f"  Position: ({car_x:.1f}, {car_y:.1f})",
            f"  Heading:  {np.degrees(car_yaw):.1f}°",
            f"  Speed:    {car_speed:.1f} m/s ({car_speed*3.6:.1f} km/h)",
            f"  Steering: {np.degrees(steering):.1f}°",
            "",
            "SENSOR DATA",
            f"  Left edges:  {len(lidar_scan.left_edge)} pts",
            f"  Right edges: {len(lidar_scan.right_edge)} pts",
            f"  Total edges: {len(lidar_scan.all_points)} pts",
            "",
            "COMPUTED PATH",
            f"  Points: {len(computed_path)}",
        ]

        if metrics is not None:
            lines.extend([
                "",
                "QUALITY METRICS",
                f"  Mean error: {metrics.mean_lateral_error:.3f} m",
                f"  Max error:  {metrics.max_lateral_error:.3f} m",
                f"  RMS error:  {metrics.rms_error:.3f} m",
                f"  Coverage:   {metrics.coverage*100:.1f}%",
                f"  Smoothness: {metrics.smoothness:.6f}",
            ])

        lines.extend([
            "",
            "PROGRESS",
            f"  Distance: {distance_traveled:.1f} m",
        ])

        return "\n".join(lines)

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)

    def save_frame(self, filename: str):
        """Save current frame."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    # Test visualizer
    print("Testing RealtimeVisualizer...")
    print("(This test requires a display)")

    # Create simple track
    grid = np.zeros((100, 200), dtype=bool)
    grid[30:70, 20:180] = True

    # Simple reference centerline
    t = np.linspace(20, 180, 100)
    reference = np.column_stack([t, np.ones(100) * 50])

    try:
        viz = RealtimeVisualizer(
            navigable_grid=grid,
            reference_centerline=reference,
            origin=(0, 0),
            resolution=1.0
        )

        # Create mock lidar scan
        scan = LidarScan(
            left_edge=np.column_stack([np.linspace(55, 100, 20), np.ones(20) * 70]),
            right_edge=np.column_stack([np.linspace(55, 100, 20), np.ones(20) * 30]),
            all_points=np.zeros((40, 2)),
            ranges=np.zeros(40),
            angles=np.zeros(40),
            car_position=(50, 50),
            car_yaw=0.0
        )

        # Mock computed path
        computed = np.column_stack([np.linspace(55, 95, 20), np.ones(20) * 50])

        # Mock metrics
        metrics = EvaluationMetrics(
            mean_lateral_error=0.15,
            max_lateral_error=0.3,
            min_lateral_error=0.05,
            std_lateral_error=0.08,
            rms_error=0.17,
            coverage=0.95,
            smoothness=0.001,
            num_computed_points=20,
            num_matched_points=19
        )

        import time
        for i in range(50):
            viz.update(
                car_x=50 + i,
                car_y=50,
                car_yaw=0.0,
                car_speed=5.0,
                steering=0.05,
                lidar_scan=scan,
                computed_path=computed,
                target_point=(60 + i, 50),
                metrics=metrics,
                distance_traveled=i * 5.0
            )
            time.sleep(0.1)

        viz.close()
        print("Test complete!")

    except Exception as e:
        print(f"Test skipped (no display?): {e}")
