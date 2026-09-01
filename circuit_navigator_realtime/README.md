# Circuit Navigator Realtime - LiDAR-Based Navigation Module

Real-time autonomous circuit navigation using simulated LiDAR edge detection for on-the-fly centerline computation.

---

## Overview

This module implements a **real-time approach** to circuit navigation:

1. **Each frame**: Detect track edges using simulated LiDAR
2. **Filter**: Apply multi-stage smoothing to noisy edge data
3. **Plan**: Compute centerline from edge midpoints
4. **Follow**: Use Pure Pursuit to track the computed path

This approach is suited for dynamic environments where the track cannot be known in advance.

---

## Key Difference from Offline Approach

| Aspect | `circuit_navigator/` | `circuit_navigator_realtime/` |
|--------|----------------------|-------------------------------|
| Centerline | Pre-computed at startup | Computed every frame |
| Sensor | None (uses map) | Simulated LiDAR |
| Adaptability | Fixed path | Adapts to sensor view |
| Computation | Low (follow only) | Higher (sense + filter + plan + follow) |
| Use case | Known circuits | Unknown/dynamic environments |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME LOOP (10 Hz)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. SENSE                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    LidarEdgeSimulator                                │   │
│   │                                                                      │   │
│   │   • Cast 180 rays across 180° FOV                                   │   │
│   │   • Detect track boundary intersections                             │   │
│   │   • Classify edges as LEFT or RIGHT (track-relative)                │   │
│   │   • Detect islands (roundabout centers)                             │   │
│   │                                                                      │   │
│   │   Output: LidarScan (left_edge, right_edge, island_points)          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   2. FILTER                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      EdgeSmoother                                    │   │
│   │                                                                      │   │
│   │   Stage 1: RANSAC Outlier Rejection                                 │   │
│   │   ├── Fit polynomial to edge points                                 │   │
│   │   └── Reject points > 0.5m from fitted curve                        │   │
│   │                                                                      │   │
│   │   Stage 2: Savitzky-Golay Spatial Smoothing                         │   │
│   │   ├── Window size: 11 points                                        │   │
│   │   └── Polynomial order: 3 (preserves shape)                         │   │
│   │                                                                      │   │
│   │   Stage 3: Kalman Temporal Smoothing                                │   │
│   │   ├── Reduces frame-to-frame jitter                                 │   │
│   │   └── Process noise: 0.1, Measurement noise: 0.5                    │   │
│   │                                                                      │   │
│   │   Output: Smoothed left_edge, right_edge                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   3. PLAN                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  LocalCenterlinePlanner                              │   │
│   │                                                                      │   │
│   │   • Interpolate left/right edges at uniform forward distances       │   │
│   │   • Compute midpoints with adaptive right-bias:                     │   │
│   │     - Normal: alpha = 0.45 (near center)                            │   │
│   │     - Roundabout approach: alpha = 0.15 (strong right bias)         │   │
│   │   • Island avoidance: Stay right of detected islands                │   │
│   │   • Savitzky-Golay smoothing on final path                          │   │
│   │                                                                      │   │
│   │   Output: computed_path (N x 2 array)                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   4. EVALUATE (optional)                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   CenterlineEvaluator                                │   │
│   │                                                                      │   │
│   │   Compare computed path against reference centerline:               │   │
│   │   • Mean lateral error                                              │   │
│   │   • Max lateral error                                               │   │
│   │   • RMS error                                                       │   │
│   │   • Coverage percentage                                             │   │
│   │   • Path smoothness                                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   5. CONTROL                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │               PurePursuitController (shared/)                        │   │
│   │                                                                      │   │
│   │   • Compute lookahead distance based on speed                       │   │
│   │   • Find target point on computed path                              │   │
│   │   • Compute steering angle: delta = atan2(2*L*sin(alpha), Ld)       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   6. ACT                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  BicycleModel (shared/)                              │   │
│   │                                                                      │   │
│   │   Update vehicle state using kinematic bicycle model                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   7. VISUALIZE                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  RealtimeVisualizer                                  │   │
│   │                                                                      │   │
│   │   Display: Track, reference centerline, LiDAR rays, detected edges, │   │
│   │            computed path, vehicle, target point, metrics            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### `main.py` - Entry Point

Main simulation loop with sense-filter-plan-control architecture.

**Configuration Constants:**

```python
# LiDAR sensor
SENSOR_RANGE_M = 20.0           # Default detection range (configurable)
NUM_LIDAR_RAYS = 180            # Angular resolution
LIDAR_FOV = np.pi               # 180 degrees forward-facing

# Edge smoothing
RANSAC_ENABLED = True
SAVGOL_ENABLED = True
KALMAN_ENABLED = True

# Centerline planning
MIN_FORWARD_DIST = 2.0          # Minimum forward distance
MAX_FORWARD_DIST = 40.0         # Maximum forward distance
CENTERLINE_SAFETY_MARGIN = 0.5  # Safety margin from edges

# Roundabout navigation (counterclockwise - European style)
ROUNDABOUT_RIGHT_BIAS = 0.45            # Normal driving: near center
ROUNDABOUT_WIDENING_THRESHOLD = 1.1     # Width ratio to trigger roundabout mode
BASELINE_TRACK_WIDTH = 6.0              # Normal track width (m)
ADAPTIVE_BIAS_ENABLED = True            # Enable adaptive bias

# Speed control
SPEED_DEFAULT_MS = 5.0          # Default cruise speed (m/s)
SPEED_CURVE_MS = 3.5            # Speed in curves (m/s)
CURVATURE_THRESHOLD = 0.05      # Curvature to trigger slow down
```

---

### `sensor/lidar_simulator.py` - LiDAR Simulation

Simulates a forward-facing LiDAR sensor detecting track boundaries.

**Classes:**

#### `LidarScan`
Dataclass containing scan results:

| Field | Type | Description |
|-------|------|-------------|
| `left_edge` | `np.ndarray` | Left boundary points (N x 2) |
| `right_edge` | `np.ndarray` | Right boundary points (M x 2) |
| `all_points` | `np.ndarray` | All detected edge points |
| `ranges` | `np.ndarray` | Distance to each point |
| `angles` | `np.ndarray` | Angle of each ray (relative to heading) |
| `car_position` | `Tuple[float, float]` | Car position when scan taken |
| `car_yaw` | `float` | Car heading when scan taken |
| `island_points` | `np.ndarray` | Island boundary points |
| `has_island_ahead` | `bool` | True if island detected in forward arc |
| `island_center_angle` | `float` | Angle to island center |

#### `LidarEdgeSimulator`
Ray-casting LiDAR with track-relative edge classification:

```python
lidar = LidarEdgeSimulator(
    navigable_grid=grid,
    origin=origin,
    resolution=resolution,
    sensor_range=40.0,          # Detection range
    num_rays=180,               # Angular resolution
    fov=np.pi,                  # 180° FOV
    reference_path=centerline   # For track-relative classification
)

scan = lidar.scan(car_x, car_y, car_yaw)
```

**Key Features:**

1. **Ray Casting** (`cast_ray`)
   - Step-based ray marching (0.2m steps)
   - Detects navigable → non-navigable transitions
   - Island detection: checks if ray re-enters navigable area

2. **Track-Relative Edge Classification**
   - Uses reference path direction instead of car heading
   - Prevents left/right confusion when car heading deviates
   - Sequential progress tracking prevents jumping at roundabouts

3. **Island Detection**
   - Rays that hit non-navigable then re-enter navigable = island
   - Reports island points and center angle

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensor_range` | 40.0 | Maximum detection range (m) |
| `num_rays` | 180 | Number of rays to cast |
| `fov` | π | Field of view (radians) |
| `ray_step` | 0.2 | Ray marching step size (m) |

---

### `filtering/edge_smoother.py` - Multi-Stage Filtering

Research-based filtering pipeline for noisy LiDAR edge data.

**Classes:**

#### `SmootherConfig`
Configuration for the filtering pipeline:

| Field | Default | Description |
|-------|---------|-------------|
| `ransac_enabled` | True | Enable RANSAC outlier rejection |
| `ransac_threshold` | 0.5 | Inlier threshold (m) |
| `ransac_min_samples` | 5 | Minimum samples for model |
| `ransac_max_trials` | 100 | Maximum RANSAC iterations |
| `savgol_enabled` | True | Enable Savitzky-Golay smoothing |
| `savgol_window` | 11 | Window size (must be odd) |
| `savgol_polyorder` | 3 | Polynomial order |
| `kalman_enabled` | True | Enable Kalman temporal smoothing |
| `kalman_process_noise` | 0.1 | Process noise Q |
| `kalman_measurement_noise` | 0.5 | Measurement noise R |

#### `KalmanFilter1D`
Simple 1D Kalman filter for position smoothing:

```python
kf = KalmanFilter1D(Q=0.1, R=0.5)
smoothed_value = kf.update(measurement)
```

#### `EdgeSmoother`
Main filtering class applying three-stage pipeline:

```python
smoother = EdgeSmoother(SmootherConfig())
left_smooth, right_smooth = smoother.smooth_edges(
    left_edge, right_edge, car_x, car_y, car_yaw
)
```

**Pipeline Stages:**

1. **RANSAC Outlier Rejection**
   - Fits 2nd-degree polynomial to edge
   - Rejects points > threshold from curve
   - Robust to sporadic sensor errors

2. **Savitzky-Golay Spatial Smoothing**
   - Applies to x and y coordinates separately
   - Preserves edge shape while reducing noise
   - Window size adapts to available points

3. **Kalman Temporal Smoothing**
   - One filter per tracked point
   - Reduces frame-to-frame jitter
   - Resets when point count changes significantly

---

### `planning/local_centerline.py` - Centerline Computation

Computes centerline from edge midpoints with adaptive bias.

**Classes:**

#### `CenterlinePlannerConfig`
Configuration for centerline computation:

| Field | Default | Description |
|-------|---------|-------------|
| `min_forward_dist` | 2.0 | Minimum forward distance (m) |
| `max_forward_dist` | 40.0 | Maximum forward distance (m) |
| `num_slices` | 40 | Number of distance slices for pairing |
| `smoothing_window` | 7 | Final path smoothing window |
| `min_path_points` | 5 | Minimum points for valid path |
| `safety_margin` | 0.5 | Safety margin from edges (m) |
| `right_bias_alpha` | 0.45 | Normal driving bias (0.5 = center) |
| `track_widening_threshold` | 1.1 | Width ratio for roundabout detection |
| `baseline_track_width` | 6.0 | Normal track width (m) |
| `adaptive_bias_enabled` | True | Enable adaptive bias |

#### `LocalCenterlinePlanner`
Main planning class:

```python
planner = LocalCenterlinePlanner(config)
centerline = planner.compute(
    left_edge, right_edge,
    car_x, car_y, car_yaw,
    island_points=scan.island_points,
    has_island_ahead=scan.has_island_ahead
)
```

**Algorithm:**

1. **Edge Interpolation**
   - Transform edges to car frame
   - Create interpolators for left/right Y position vs forward X
   - Sample at uniform forward distances

2. **Centerline Computation**
   - Normal: `y_center = y_right + 0.45 * (y_left - y_right)`
   - Roundabout approach (width > 1.1x baseline AND left asymmetric):
     `y_center = y_right + 0.15 * (y_left - y_right)`

3. **Island Avoidance**
   - Validates island is genuine (not full-width obstacle)
   - Computes path staying right of island
   - Uses 5% alpha (maximum right bias) near islands

4. **Gap Handling**
   - When edges don't overlap in forward distance
   - Uses single-edge offset with estimated track width

5. **Smoothing**
   - Savitzky-Golay filter on final path
   - Window size 7, polynomial order 2

---

### `evaluation/centerline_compare.py` - Quality Metrics

Evaluates computed centerline against pre-computed reference.

**Classes:**

#### `EvaluationMetrics`
Dataclass for quality measurements:

| Field | Type | Description |
|-------|------|-------------|
| `mean_lateral_error` | `float` | Average distance from reference (m) |
| `max_lateral_error` | `float` | Maximum deviation (m) |
| `min_lateral_error` | `float` | Minimum deviation (m) |
| `std_lateral_error` | `float` | Standard deviation |
| `rms_error` | `float` | Root mean square error |
| `coverage` | `float` | Fraction of points matched (0-1) |
| `smoothness` | `float` | Curvature variance (lower = smoother) |
| `num_computed_points` | `int` | Points in computed path |
| `num_matched_points` | `int` | Points matched to reference |

#### `CenterlineEvaluator`
Comparison against reference:

```python
evaluator = CenterlineEvaluator(reference_path, match_radius=5.0)
metrics = evaluator.evaluate(computed_path, car_x, car_y)
print(format_metrics(metrics))
```

**Metrics Computation:**

- **Lateral Error**: Perpendicular distance to reference path tangent
- **Coverage**: Points within `match_radius` of reference
- **Smoothness**: Variance of path curvature

---

### `visualization/visualizer.py` - Enhanced Display

Real-time visualization with LiDAR rays and quality metrics.

**Classes:**

#### `VisualizerConfig`
Extended configuration:

| Field | Default | Description |
|-------|---------|-------------|
| `vehicle_length` | 4.087 | Vehicle length (Renault Zoe) |
| `vehicle_width` | 1.73 | Vehicle width |
| `trail_length` | 200 | Position history length |
| `sensor_range` | 40.0 | LiDAR range for display |

#### `RealtimeVisualizer`
Enhanced matplotlib renderer:

```python
visualizer = RealtimeVisualizer(
    navigable_grid=grid,
    reference_centerline=reference_path,
    origin=origin,
    resolution=resolution,
    config=viz_config
)

visualizer.update(
    car_x=state.x,
    car_y=state.y,
    car_yaw=state.yaw,
    car_speed=state.speed,
    steering=state.steering,
    lidar_scan=scan,
    computed_path=path,
    target_point=target,
    metrics=metrics,
    distance_traveled=distance
)
```

**Display Elements:**
- Track/navigable area
- Reference centerline (dashed)
- LiDAR rays
- Detected left/right edges (colored points)
- Computed centerline
- Vehicle position and orientation
- Target point
- Quality metrics overlay
- Distance traveled

---

## Usage

### Basic Usage

```bash
python -m circuit_navigator_realtime.main --bag sy27_road_ground_truth2 --range 40.0 --laps 2
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bag` | string | `sy27_road_ground_truth2/` | Path to ROS2 MCAP bag |
| `--range` | float | 40.0 | LiDAR sensor range (m) |
| `--laps` | int | 2 | Number of laps (0 = infinite) |
| `--headless` | flag | False | Run without visualization |
| `--no-metrics` | flag | False | Disable console metrics |

### Examples

```bash
# Short range sensor (20m)
python -m circuit_navigator_realtime.main --range 20.0

# Long range sensor (60m)
python -m circuit_navigator_realtime.main --range 60.0

# Run single lap without visualization
python -m circuit_navigator_realtime.main --headless --laps 1
```

### Programmatic Usage

```python
from circuit_navigator_realtime.sensor.lidar_simulator import LidarEdgeSimulator
from circuit_navigator_realtime.filtering.edge_smoother import EdgeSmoother
from circuit_navigator_realtime.planning.local_centerline import LocalCenterlinePlanner

# Initialize components
lidar = LidarEdgeSimulator(grid, origin, resolution, sensor_range=40.0)
smoother = EdgeSmoother()
planner = LocalCenterlinePlanner()

# Per-frame processing
scan = lidar.scan(car_x, car_y, car_yaw)
left_smooth, right_smooth = smoother.smooth_edges(
    scan.left_edge, scan.right_edge, car_x, car_y, car_yaw
)
centerline = planner.compute(left_smooth, right_smooth, car_x, car_y, car_yaw)
```

---

## Roundabout Navigation

The system includes specialized logic for counterclockwise (European-style) roundabout navigation.

### Detection

Roundabout approach is detected when **both** conditions are met:
1. Track width > 1.1x baseline (widening)
2. Left edge farther than right edge (asymmetry > 1.2)

### Strategy

| Condition | Alpha | Behavior |
|-----------|-------|----------|
| Normal track | 0.45 | Near center (slight right) |
| Roundabout approach | 0.15 | Strong right bias |
| Island detected | 0.05 | Maximum right (stay right of island) |

### Island Avoidance

1. Validate island is genuine (not full-width obstacle)
2. Track island's rightmost boundary per forward distance
3. Compute path staying right of island with safety margin

---

## Output

### Console Output

```
======================================================================
Real-Time Centerline Navigator - Starting Simulation
======================================================================
  Sensor range: 40.0m
  Lidar rays: 180
  Lidar FOV: 180 degrees
======================================================================

[1/5] Extracting map from MCAP bag...
  Grid size: (850, 1200)
  Resolution: 0.5 m/cell

[2/5] Processing navigable area...
  Navigable cells: 125000 (12.3%)

[3/5] Extracting REFERENCE centerline (for comparison)...
  Reference centerline: 2847 points
  Reference length: 1423.5 m

[4/5] Initializing real-time components...
  Lidar simulator: 40.0m range, 180 rays
  Using track-relative edge classification (reference path)
  Edge smoother: RANSAC=True, Savgol=True, Kalman=True
  Centerline planner: forward range [2.0, 40.0]m
  Roundabout mode: right_bias=0.45, widening_threshold=1.1

[5/5] Initializing visualization...

======================================================================
Starting REAL-TIME simulation loop (Ctrl+C to stop)
======================================================================

Pos: (  45.2,  123.4) | Speed:  5.0 m/s | Steer:   0.0° | Lat Err: 0.123m | Progress:   0.0%
Pos: (  50.1,  127.8) | Speed:  5.0 m/s | Steer:  -2.3° | Lat Err: 0.089m | Progress:   5.2%
...

*** Path traversal 1 complete! ***
  Mean lateral error: 0.145 m
  Max lateral error: 0.892 m
  RMS error: 0.198 m
  Coverage: 98.5%

======================================================================
Simulation Complete - Final Statistics
======================================================================
  Total steps: 2856
  Laps completed: 2
  Distance traveled: 2847.0 m

  QUALITY METRICS vs REFERENCE:
    Mean lateral error: 0.142 m
    Max lateral error: 0.912 m
    RMS error: 0.195 m
    Coverage: 98.7%
======================================================================
```

---

## File Structure

```
circuit_navigator_realtime/
├── __init__.py
├── main.py                      # Entry point with sense-filter-plan-control loop
├── README.md                    # This file
├── diagnostic_data.npz          # Saved diagnostic data
│
├── sensor/
│   ├── __init__.py
│   └── lidar_simulator.py       # Ray-casting LiDAR with edge classification
│
├── filtering/
│   ├── __init__.py
│   └── edge_smoother.py         # RANSAC + Savgol + Kalman pipeline
│
├── planning/
│   ├── __init__.py
│   └── local_centerline.py      # Edge-midpoint centerline with adaptive bias
│
├── evaluation/
│   ├── __init__.py
│   └── centerline_compare.py    # Quality metrics vs reference
│
├── visualization/
│   ├── __init__.py
│   └── visualizer.py            # Enhanced display with LiDAR rays
│
└── archived_analysis/           # Development and debug scripts
    ├── diagnostic_analysis.py
    ├── test_centerline.py
    └── test_edge_cases.py
```

---

## Dependencies

Uses components from `shared/`:
- `shared.simulation.BicycleModel`
- `shared.simulation.VehicleState`
- `shared.navigation.PurePursuitController`
- `shared.navigation.PurePursuitConfig`

Uses components from `circuit_navigator/`:
- `circuit_navigator.map_processing.GridMapExtractor`
- `circuit_navigator.map_processing.CenterlineExtractor`

External packages:
- `numpy` - Array operations
- `scipy` - Signal processing, interpolation
- `scikit-image` - Morphological operations
- `matplotlib` - Visualization

---

## Performance Tuning

### Sensor Range

| Range | Pros | Cons |
|-------|------|------|
| Short (20m) | Fast, responsive | Limited planning horizon |
| Medium (40m) | Balanced | Good for most tracks |
| Long (60m+) | Long-range planning | More computational cost |

### Filtering

Disable filters for faster but noisier operation:
```python
smoother_config = SmootherConfig(
    ransac_enabled=False,   # Faster, but sensitive to outliers
    savgol_enabled=True,    # Keep for spatial smoothing
    kalman_enabled=False    # Faster, but more jittery
)
```

### Roundabout Bias

Adjust for different track widths:
```python
planner_config = CenterlinePlannerConfig(
    baseline_track_width=8.0,          # For wider tracks
    track_widening_threshold=1.2,      # Less sensitive
    right_bias_alpha=0.40              # More centered normal driving
)
```

---

