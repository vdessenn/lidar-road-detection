# Circuit Navigator - Offline Navigation Module

Autonomous circuit navigation using pre-computed centerline extraction from occupancy grids.

---

## Overview

This module implements an **offline approach** to circuit navigation:

1. **At startup**: Extract the complete track centerline from a pre-recorded map
2. **During runtime**: Follow the centerline using Pure Pursuit control

This approach is ideal for known circuits where an optimal reference trajectory can be pre-computed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STARTUP PHASE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ROS2 MCAP Bag                                                             │
│   (/road_whole PointCloud2)                                                 │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────┐                                                   │
│   │  GridMapExtractor   │  Parse MCAP, extract PointCloud2                  │
│   │                     │  Convert to occupancy grid                        │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │  process_raw_grid() │  Morphological closing + safety erosion           │
│   │                     │  Creates binary navigable grid                    │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │ CenterlineExtractor │  Skeletonization → pruning → ordering → smoothing │
│   │                     │  Produces ordered world-coordinate path           │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│        centerline.points_world (N x 2 array)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUNTIME LOOP (10 Hz)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────┐                                                   │
│   │  get_visible_path() │  Extract path segment ahead of vehicle            │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │compute_target_speed │  Curvature-based speed adaptation                 │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │ PurePursuitController│  Compute steering angle to follow path          │
│   │  (from shared/)      │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │   BicycleModel      │  Update vehicle state (x, y, yaw, speed)         │
│   │  (from shared/)      │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                   │
│   │ CircuitVisualizer   │  Real-time matplotlib display                     │
│   └─────────────────────┘                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### `main.py` - Entry Point

The main simulation loop orchestrating all components.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_simulation()` | Main entry point; loads map, extracts centerline, runs control loop |
| `process_raw_grid()` | Converts sparse point cloud to binary navigable grid |
| `get_visible_path()` | Extracts path segment within lookahead horizon |
| `compute_target_speed()` | Computes speed based on upcoming path curvature |
| `find_start_position()` | Determines initial vehicle pose from centerline |

**Configuration Constants:**

```python
# Map processing
GRID_RESOLUTION = 0.5           # meters per cell
SAFETY_MARGIN_M = 0.50          # 50cm safety margin from track edges
CLOSING_RADIUS_M = 2.0          # Morphological closing radius

# Path following
VISIBLE_HORIZON_M = 35.0        # Lookahead visibility distance

# Simulation
SIMULATION_HZ = 10              # Update frequency
DT = 1.0 / SIMULATION_HZ        # Time step (0.1s)

# Speed control
SPEED_DEFAULT_MS = 5.0          # Default cruise speed (m/s)
SPEED_CURVE_MS = 3.5            # Speed in curves (m/s)
CURVATURE_THRESHOLD = 0.05      # Curvature to trigger slow down (1/m)
```

---

### `map_processing/grid_extractor.py` - MCAP Parser

Extracts road point cloud data from ROS2 MCAP bag files.

**Classes:**

#### `GridMapData`
Dataclass containing extracted grid map:

| Field | Type | Description |
|-------|------|-------------|
| `raw_data` | `np.ndarray` | Grid data (NaN for empty cells) |
| `resolution` | `float` | Meters per cell |
| `origin` | `Tuple[float, float]` | World origin (x, y) |
| `dimensions` | `Tuple[int, int]` | Grid size (rows, cols) |
| `length_x`, `length_y` | `float` | World size in meters |
| `layers` | `List[str]` | Available layer names |
| `point_cloud` | `np.ndarray` | Original XYZ points |

#### `GridMapExtractor`
Parses MCAP files and converts PointCloud2 to occupancy grid:

```python
extractor = GridMapExtractor(bag_path, resolution=0.5)
map_data = extractor.extract()
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `extract()` | Main entry; returns `GridMapData` |
| `_parse_pointcloud2()` | Parses CDR-encoded PointCloud2 message |
| `_points_to_grid()` | Converts point cloud to occupancy grid |

**Supported Features:**
- zstd-compressed MCAP files
- PointCloud2 message parsing (x, y, z fields)
- Automatic MCAP file discovery in directories

---

### `map_processing/centerline.py` - Centerline Extraction

Extracts track centerline using morphological skeletonization.

**Classes:**

#### `Centerline`
Dataclass for extracted centerline:

| Field | Type | Description |
|-------|------|-------------|
| `points_pixel` | `np.ndarray` | Pixel coordinates (N x 2) |
| `points_world` | `np.ndarray` | World coordinates (N x 2) |
| `length_m` | `float` | Total path length in meters |
| `is_closed` | `bool` | Whether track is a closed loop |
| `resolution` | `float` | Grid resolution |
| `origin` | `Tuple[float, float]` | World origin |

#### `CenterlineExtractor`
Main extraction class:

```python
extractor = CenterlineExtractor(min_branch_length=10, smoothing_window=5)
centerline = extractor.extract(navigable_grid, resolution, origin, method="skeleton")
centerline = extractor.resample_by_distance(centerline, spacing_m=0.5)
```

**Extraction Methods:**

| Method | Description |
|--------|-------------|
| `skeleton` | Traditional morphological skeletonization |
| `ridge` | Distance transform ridge following |

**Algorithm Pipeline:**

1. **Skeletonization** (`skeletonize` from scikit-image)
   - Input: Binary navigable grid
   - Output: Single-pixel-wide skeleton

2. **Branch Pruning** (`_prune_branches`)
   - Removes short spurious branches (< `min_branch_length` pixels)
   - Preserves main track path

3. **Point Ordering** (`_order_skeleton_points`)
   - Traverses skeleton with direction continuity
   - Handles junctions with scoring:
     - Direction continuity (stay straight): weight 10.0
     - Distance from boundary (prefer main track): weight 3.0
     - Right turn preference (roundabouts): weight 1.5
     - Dead-end penalty: -50.0

4. **Gap Jumping** (`_find_jump_target`)
   - Handles sparse/fragmented skeletons
   - Jumps up to 30m gaps while maintaining direction

5. **Smoothing** (`_smooth_centerline`)
   - Moving average filter
   - Handles both open and closed paths

6. **Coordinate Conversion** (`_to_world_coordinates`)
   - Pixel (row, col) → World (x, y)

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_branch_length` | 10 | Minimum branch length to keep (pixels) |
| `smoothing_window` | 5 | Moving average window size |

---

### `visualization/visualizer.py` - Real-Time Display

Provides real-time matplotlib visualization of the simulation.

**Classes:**

#### `VisualizerConfig`
Configuration for visualization:

| Field | Default | Description |
|-------|---------|-------------|
| `vehicle_length` | 4.0 | Vehicle length for display (m) |
| `vehicle_width` | 2.0 | Vehicle width for display (m) |
| `trail_length` | 200 | Number of past positions to show |

#### `CircuitVisualizer`
Real-time matplotlib renderer:

```python
visualizer = CircuitVisualizer(
    navigable_grid=grid,
    centerline_world=centerline,
    origin=origin,
    resolution=resolution,
    config=viz_config
)

# In loop:
visualizer.update(
    car_x=state.x,
    car_y=state.y,
    car_yaw=state.yaw,
    car_speed=state.speed,
    steering=state.steering,
    target_point=target,
    visible_path=visible_path,
    path_idx=current_idx,
    lap_progress=progress
)
```

**Display Elements:**
- Track/navigable area (binary grid)
- Pre-computed centerline
- Vehicle position and orientation
- Target point (lookahead)
- Visible path segment
- Position trail (history)
- Status text (speed, steering, progress)

---

## Usage

### Basic Usage

```bash
python -m circuit_navigator.main --bag sy27_road_ground_truth2 --laps 2
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bag` | string | `sy27_road_ground_truth2/` | Path to ROS2 MCAP bag |
| `--laps` | int | 2 | Number of laps (0 = infinite) |
| `--headless` | flag | False | Run without visualization |

### Programmatic Usage

```python
from circuit_navigator.map_processing.grid_extractor import GridMapExtractor
from circuit_navigator.map_processing.centerline import CenterlineExtractor

# Extract grid
extractor = GridMapExtractor("path/to/bag", resolution=0.5)
map_data = extractor.extract()

# Process to navigable area
navigable = process_raw_grid(map_data.raw_data, map_data.resolution)

# Extract centerline
cl_extractor = CenterlineExtractor()
centerline = cl_extractor.extract(navigable, map_data.resolution, map_data.origin)

# Resample for smooth following
centerline = cl_extractor.resample_by_distance(centerline, spacing_m=0.5)

print(f"Centerline: {len(centerline.points_world)} points, {centerline.length_m:.1f}m")
```

---

## Algorithm Details

### Grid Processing

```python
def process_raw_grid(raw_grid, resolution):
    # 1. Mark cells with valid data
    has_point = ~np.isnan(raw_grid) & (raw_grid != 0)

    # 2. Morphological closing to fill gaps
    closing_radius_px = int(CLOSING_RADIUS_M / resolution)
    kernel = disk(closing_radius_px)
    closed = closing(has_point, kernel)

    # 3. Remove small isolated regions (< 50 sq meters)
    min_size = int(50 / (resolution * resolution))
    cleaned = remove_small_objects(closed, min_size=min_size)

    # 4. Apply safety margin via erosion
    safety_radius_px = int(SAFETY_MARGIN_M / resolution)
    navigable = binary_erosion(cleaned, disk(safety_radius_px))

    return navigable
```

### Path Visibility

The `get_visible_path()` function extracts the portion of centerline visible to the vehicle:

1. Find nearest point on path (with forward search constraint)
2. Accumulate path distance up to `VISIBLE_HORIZON_M`
3. Return path segment and updated index

**Key feature**: Index only advances forward, preventing oscillation on out-and-back sections.

### Speed Profile

Curvature-based speed adaptation:

```python
def compute_target_speed(path_points):
    # Compute curvature at each point (5-point estimation)
    for i in range(2, len(path_points) - 2):
        angle_change = heading[i+2] - heading[i-2]
        arc_length = distance(p[i-2], p[i+2])
        curvature = abs(angle_change) / arc_length

    # Slow down for curves
    if max(curvatures) > CURVATURE_THRESHOLD:
        return SPEED_CURVE_MS  # 3.5 m/s
    else:
        return SPEED_DEFAULT_MS  # 5.0 m/s
```

---

## Output

### Console Output

```
================================================================
Circuit Navigator - Starting Simulation
================================================================

[1/4] Extracting map from MCAP bag...
  Grid size: (850, 1200)
  Resolution: 0.5 m/cell
  Origin: (-150.0, -200.0)

[2/4] Processing navigable area...
  Navigable cells: 125000 (12.3%)

[3/4] Extracting centerline...
  Centerline points: 2847
  Centerline length: 1423.5 m
  Closed loop: True
  After resampling: 2848 points

[4/4] Initializing simulation...
  Start position: (45.2, 123.4)
  Start heading: 45.0 deg

================================================================
Starting simulation loop (Ctrl+C to stop)
================================================================

Pos: (  45.2,  123.4) | Speed:  5.0 m/s | Steer:   0.0° | Progress:   0.0% | Lap: 0
Pos: (  50.1,  127.8) | Speed:  5.0 m/s | Steer:  -2.3° | Progress:   5.2% | Lap: 0
...

*** Path traversal 1 complete! ***

Pos: (  45.2,  123.4) | Speed:  5.0 m/s | Steer:   0.0° | Progress:   0.0% | Lap: 1
...

Simulation complete!
  Total steps: 2856
  Laps completed: 2
```

---

## File Structure

```
circuit_navigator/
├── __init__.py
├── main.py                      # Entry point and simulation loop
├── README.md                    # This file
│
├── map_processing/
│   ├── __init__.py
│   ├── grid_extractor.py        # MCAP parsing, PointCloud2 → grid
│   └── centerline.py            # Skeletonization-based extraction
│
├── visualization/
│   ├── __init__.py
│   └── visualizer.py            # Real-time matplotlib display
│
└── archived_analysis/           # Development and debug scripts
    ├── analyze_skeleton.py
    ├── analyze_dead_end.py
    ├── analyze_path_end.py
    ├── analyze_second_roundabout.py
    ├── check_skeleton_connectivity.py
    ├── debug_junction_decisions.py
    ├── find_return_path.py
    ├── visualize_centerline.py
    ├── visualize_road.py
    └── ...
```

---

## Dependencies

Uses components from `shared/`:
- `shared.simulation.BicycleModel`
- `shared.simulation.VehicleState`
- `shared.navigation.PurePursuitController`
- `shared.navigation.PurePursuitConfig`

External packages:
- `numpy` - Array operations
- `scipy` - Morphology (`ndimage`)
- `scikit-image` - Skeletonization, morphological closing
- `matplotlib` - Visualization
- `mcap` - MCAP file reading

---

## Limitations

1. **Requires pre-recorded map**: Cannot handle unknown environments
2. **Static obstacles only**: Does not detect dynamic obstacles
3. **Single reference path**: No re-planning capability
4. **Simulated vehicle**: Uses kinematic bicycle model, not real hardware

---

