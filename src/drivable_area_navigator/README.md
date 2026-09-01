# Drivable Area Navigator

**ROS2 Jazzy | Python 3.12.3**

Computes drivable centerlines from road boundary data detected by LiDAR.

---

## Overview

This package receives left/right road boundaries from `drivable_area_detector` and computes a smooth centerline path suitable for vehicle navigation.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DRIVABLE_AREA_NAVIGATOR                   │
│                                                             │
│  /ring_boundaries ──► boundary_receiver_node                │
│   (PointCloud2)              │                              │
│                              ▼                              │
│                     centerline_planner                      │
│                     (midpoint + smoothing)                  │
│                              │                              │
│                              ▼                              │
│                     /planned_centerline ──► RViz2           │
│                         (Path)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
src/drivable_area_navigator/
├── drivable_area_navigator/
│   ├── __init__.py
│   ├── types.py                    # Dataclasses, ROS2 message converters
│   ├── boundary_receiver_node.py   # Main node: boundaries → centerline
│   ├── edge_smoother.py            # Kalman filtering (disabled by default)
│   └── centerline_planner.py       # Centerline computation algorithm
├── config/
│   └── params.yaml                 # Configuration parameters
├── launch/
│   └── navigator.launch.py         # Launch file
├── package.xml
├── setup.py
└── README.md
```

---

## Node: boundary_receiver_node

**Executable:** `boundary_receiver_node`

Receives road boundaries and computes the centerline.

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/ring_boundaries` | PointCloud2 | Subscribe | Boundary points with `side_id` field |
| `/planned_centerline` | Path | Publish | Computed centerline path |
| `/centerline_markers` | MarkerArray | Publish | Visualization markers |

**Pipeline:**
1. Parse PointCloud2 → Extract left (`side_id=1`) / right (`side_id=-1`) points
2. Compute centerline via `LocalCenterlinePlanner`
3. Publish as `nav_msgs/Path`

---

## Centerline Algorithm

The `LocalCenterlinePlanner` computes centerline from boundary edges:

1. **Pair edges** at same forward distances using linear interpolation
2. **Compute midpoint** with adaptive right-bias for roundabout navigation
3. **Smooth path** using Savitzky-Golay filter
4. **Handle edge cases** (single edge, no overlap)

### Adaptive Right-Bias

For counterclockwise roundabout navigation:
- **Normal road:** `alpha = 0.45` (slight right of center)
- **Roundabout detected:** `alpha = 0.15` (strong right bias)
- **Detection trigger:** track width > 1.1× baseline + left asymmetry

### Curvature Computation

The planner can compute path curvature for analysis:

```python
curvature = planner.compute_curvature(centerline)
# Returns (N,) array of curvature values (1/m)
# Higher values = sharper turns
```

---

## Configuration

`config/params.yaml`:

```yaml
topics:
  boundaries_input: "/ring_boundaries"
  centerline_output: "/planned_centerline"

frame_id: "velodyne"

# Edge Smoother (DISABLED - upstream provides temporal smoothing)
edge_smoother:
  enable_kalman: false

# Centerline Planner
centerline_planner:
  min_forward_dist: 2.0       # meters
  max_forward_dist: 20.0      # meters
  num_slices: 40              # sampling points
  safety_margin: 0.5          # meters from edges
  right_bias_alpha: 0.45      # 0.5 = center, <0.5 = right
  adaptive_bias_enabled: true
  baseline_track_width: 6.0   # meters
  track_widening_threshold: 1.1
```

---

## Kalman Filter Evaluation

**Status: DISABLED by default**

The Kalman filter in `edge_smoother.py` is redundant because upstream Node 02 (`ring_boundary_extraction`) already provides:
- RANSAC polynomial fitting (outlier rejection)
- 5-frame temporal buffer (noise averaging)
- Geometric constraints (width enforcement)

The Kalman filter adds no significant improvement and may introduce lag.

---

## Usage

### With Detector Pipeline

```bash
./run_navigator.sh
```

This runs:
1. Patchwork++ (ground segmentation)
2. Drivable Area Detector (boundary extraction)
3. **Boundary Receiver (centerline computation)**
4. RViz2 (visualization)
5. Rosbag playback

### Standalone

```bash
ros2 run drivable_area_navigator boundary_receiver_node --ros-args -p use_sim_time:=true
```

### Launch File

```bash
ros2 launch drivable_area_navigator navigator.launch.py
```

---

## RViz2 Displays

| Display | Topic | Type | Color |
|---------|-------|------|-------|
| Ground Points | `/patchworkpp/ground` | PointCloud2 | Blue |
| Road Surface | `/road_surface` | PointCloud2 | Red |
| Boundaries | `/road_boundaries` | MarkerArray | Cyan/Magenta |
| **Centerline** | `/planned_centerline` | Path | **Yellow** |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `rclpy` | ROS2 Python client |
| `numpy` | Array operations |
| `scipy` | Interpolation, Savitzky-Golay filter |
| `nav_msgs` | Path messages |
| `sensor_msgs` | PointCloud2 messages |
| `geometry_msgs` | Pose, Point, Quaternion |
| `visualization_msgs` | Marker, MarkerArray |

---

## Integration with Detector

```
drivable_area_detector          drivable_area_navigator
┌─────────────────────┐         ┌─────────────────────┐
│ 01_road_extraction  │         │                     │
│        ↓            │         │                     │
│ 02_ring_boundary    │────────►│ boundary_receiver   │
│   (PointCloud2)     │         │        ↓            │
│        ↓            │         │ centerline_planner  │
│ rviz2_publisher     │         │        ↓            │
│   (MarkerArray)     │         │    nav_msgs/Path    │
└─────────────────────┘         └─────────────────────┘
```

**Data Format:**
- Input: PointCloud2 with fields `x, y, z, side_id` (int32)
  - `side_id = 1` → left boundary
  - `side_id = -1` → right boundary
- Output: nav_msgs/Path in `velodyne` frame

---

## Author

**Victor Dessenne** - UTC AI36 Project

**License:** MIT
