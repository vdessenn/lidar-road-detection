#!/usr/bin/env python3
"""
Circuit Visualizer - 2D matplotlib visualization for circuit navigation.

Displays:
- Navigable area (track)
- Centerline path
- Vehicle position and heading
- Target lookahead point
- Visible path segment
- Status information (speed, steering, lap progress)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    vehicle_length: float = 4.0      # Vehicle length (m)
    vehicle_width: float = 2.0       # Vehicle width (m)
    update_interval_ms: int = 50     # Display update interval
    trail_length: int = 100          # Number of past positions to show
    show_info_panel: bool = True     # Show status info
    figsize: Tuple[int, int] = (14, 8)


class CircuitVisualizer:
    """Real-time visualization of vehicle navigating circuit."""

    def __init__(
        self,
        navigable_grid: np.ndarray,
        centerline_world: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        config: Optional[VisualizerConfig] = None
    ):
        """
        Initialize visualizer.

        Args:
            navigable_grid: Binary grid (True = navigable)
            centerline_world: (N, 2) centerline points in world coordinates [x, y]
            origin: World origin (x, y) of grid
            resolution: Grid resolution (m/cell)
            config: Visualization configuration
        """
        self.navigable_grid = navigable_grid
        self.centerline = centerline_world
        self.origin = origin
        self.resolution = resolution
        self.config = config or VisualizerConfig()

        # Trail history
        self.trail_x = []
        self.trail_y = []

        # Setup matplotlib
        plt.ion()  # Interactive mode
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure and axes."""
        rows, cols = self.navigable_grid.shape

        # Calculate extent in world coordinates
        self.extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        if self.config.show_info_panel:
            self.fig, (self.ax_map, self.ax_info) = plt.subplots(
                1, 2,
                figsize=self.config.figsize,
                gridspec_kw={'width_ratios': [3, 1]}
            )
        else:
            self.fig, self.ax_map = plt.subplots(figsize=self.config.figsize)
            self.ax_info = None

        # Plot track background
        track_image = np.zeros((*self.navigable_grid.shape, 3))
        track_image[self.navigable_grid] = [0.85, 0.85, 0.85]  # Light gray for track
        track_image[~self.navigable_grid] = [0.3, 0.6, 0.3]    # Green for grass

        self.ax_map.imshow(track_image, origin='lower', extent=self.extent, aspect='equal')

        # Plot full centerline (thin, semi-transparent)
        self.ax_map.plot(
            self.centerline[:, 0], self.centerline[:, 1],
            'b-', linewidth=1, alpha=0.3, label='Centerline'
        )

        # Initialize dynamic elements
        # Visible path (will be updated)
        self.visible_path_line, = self.ax_map.plot(
            [], [], 'g-', linewidth=2, label='Visible path'
        )

        # Target point
        self.target_point, = self.ax_map.plot(
            [], [], 'r*', markersize=15, label='Target'
        )

        # Trail
        self.trail_line, = self.ax_map.plot(
            [], [], 'b-', linewidth=1, alpha=0.5
        )

        # Vehicle (rectangle)
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

        # Vehicle heading arrow
        self.heading_arrow = self.ax_map.annotate(
            '',
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )

        # Configure map axis
        self.ax_map.set_xlabel('X (m)')
        self.ax_map.set_ylabel('Y (m)')
        self.ax_map.set_title('Circuit Navigation')
        self.ax_map.legend(loc='upper right')
        self.ax_map.set_aspect('equal')

        # Setup info panel
        if self.ax_info is not None:
            self.ax_info.set_xlim(0, 1)
            self.ax_info.set_ylim(0, 1)
            self.ax_info.axis('off')
            self.ax_info.set_title('Status')

            # Info text elements
            self.info_text = self.ax_info.text(
                0.1, 0.9, '', fontsize=12, family='monospace',
                verticalalignment='top'
            )

        plt.tight_layout()

    def update(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        car_speed: float,
        steering: float,
        target_point: Tuple[float, float],
        visible_path: Optional[np.ndarray] = None,
        path_idx: int = 0,
        lap_progress: float = 0.0
    ):
        """
        Update visualization with current state.

        Args:
            car_x, car_y: Vehicle position (m)
            car_yaw: Vehicle heading (rad)
            car_speed: Vehicle speed (m/s)
            steering: Steering angle (rad)
            target_point: Lookahead target (x, y)
            visible_path: (N, 2) visible path segment
            path_idx: Current index on centerline
            lap_progress: Progress through lap (0-1)
        """
        # Update trail
        self.trail_x.append(car_x)
        self.trail_y.append(car_y)
        if len(self.trail_x) > self.config.trail_length:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        self.trail_line.set_data(self.trail_x, self.trail_y)

        # Update vehicle rectangle
        # Vehicle is positioned with rear axle at (car_x, car_y)
        # Rectangle needs to be centered and rotated
        veh_length = self.config.vehicle_length
        veh_width = self.config.vehicle_width

        # Transform: translate to origin, rotate, translate to position
        transform = Affine2D()
        transform.translate(-veh_length / 4, -veh_width / 2)  # Rear axle at 1/4 from back
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

        # Update target point
        self.target_point.set_data([target_point[0]], [target_point[1]])

        # Update visible path
        if visible_path is not None and len(visible_path) > 0:
            self.visible_path_line.set_data(visible_path[:, 0], visible_path[:, 1])

        # Update info panel
        if self.ax_info is not None:
            info_str = (
                f"Position:\n"
                f"  X: {car_x:7.1f} m\n"
                f"  Y: {car_y:7.1f} m\n"
                f"\n"
                f"Motion:\n"
                f"  Speed: {car_speed:5.1f} m/s ({car_speed * 3.6:5.1f} km/h)\n"
                f"  Heading: {np.degrees(car_yaw):6.1f} deg\n"
                f"\n"
                f"Control:\n"
                f"  Steering: {np.degrees(steering):6.1f} deg\n"
                f"\n"
                f"Progress:\n"
                f"  Path idx: {path_idx:5d} / {len(self.centerline)}\n"
                f"  Lap: {lap_progress * 100:5.1f}%\n"
            )
            self.info_text.set_text(info_str)

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)

    def save_frame(self, filename: str):
        """Save current frame to file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    # Test visualizer with simple track
    print("Testing CircuitVisualizer...")

    # Create simple oval track
    grid = np.zeros((100, 150), dtype=bool)
    center = (50, 75)

    for i in range(100):
        for j in range(150):
            dist = np.sqrt(((i - center[0]) / 35) ** 2 + ((j - center[1]) / 55) ** 2)
            if 0.6 < dist < 1.0:
                grid[i, j] = True

    # Simple centerline
    t = np.linspace(0, 2 * np.pi, 200)
    centerline = np.column_stack([
        75 + 45 * np.cos(t),
        50 + 28 * np.sin(t)
    ])

    # Create visualizer
    viz = CircuitVisualizer(
        navigable_grid=grid,
        centerline_world=centerline,
        origin=(0, 0),
        resolution=1.0
    )

    # Simulate vehicle moving around track
    import time
    for i in range(200):
        idx = i % len(centerline)
        x, y = centerline[idx]
        # Calculate heading from path
        next_idx = (idx + 1) % len(centerline)
        dx = centerline[next_idx, 0] - x
        dy = centerline[next_idx, 1] - y
        yaw = np.arctan2(dy, dx)

        # Visible path segment
        vis_start = idx
        vis_end = min(idx + 30, len(centerline))
        visible = centerline[vis_start:vis_end]

        # Target point
        target_idx = min(idx + 10, len(centerline) - 1)
        target = (centerline[target_idx, 0], centerline[target_idx, 1])

        viz.update(
            car_x=x,
            car_y=y,
            car_yaw=yaw,
            car_speed=5.0,
            steering=0.1,
            target_point=target,
            visible_path=visible,
            path_idx=idx,
            lap_progress=idx / len(centerline)
        )

        time.sleep(0.05)

    print("Test complete")
    viz.close()
