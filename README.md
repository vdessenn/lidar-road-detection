# LiDAR Road Detection and Path Planning

Autonomous vehicle navigation from LiDAR point clouds: detect the drivable road surface, extract
its left and right boundaries, and plan a centerline the vehicle can follow — as a ROS 2 Jazzy
pipeline and as standalone Python simulations.

<!-- Drop the illustration in docs/images/ and uncomment:
![Pipeline output in RViz2](docs/images/pipeline.png)
-->

> Started as project AI36 / SY27 at Université de Technologie de Compiègne (autumn 2025).
> Actively maintained — the detection algorithm is still evolving.

---

## Overview

This repository contains a complete pipeline for autonomous navigation from LiDAR point clouds:

| Module | Type | Description |
|--------|------|-------------|
| `patchwork-plusplus/` | ROS2 C++ | Ground segmentation — *external, see Setup* |
| `drivable_area_detector/` | ROS2 Python | Road boundary extraction |
| `drivable_area_navigator/` | ROS2 Python | Centerline planning from boundaries |
| `circuit_navigator/` | Python standalone | Offline simulation with pre-computed path |
| `circuit_navigator_realtime/` | Python standalone | Real-time simulation with LiDAR |
| `tools/` | Python | Parameter tuning utilities |

---

## Quick Start

### Prerequisites

- ROS2 Jazzy
- Python 3.10+
- Colcon build tools

A ready-to-use ROS 2 Jazzy environment is provided in `.devcontainer/` (VS Code → *Reopen in
Container*), which installs the system and Python dependencies for you.

### Third-party sources

Two dependencies are not vendored in this repository. Clone them into the workspace first:

```bash
# Ground segmentation (Lee et al., IROS 2022)
git clone https://github.com/url-kaist/patchwork-plusplus.git patchwork-plusplus

# RViz2 plugin used to hand-pick points while tuning (optional)
git clone https://github.com/drwnz/selected_points_publisher.git src/selected_points_publisher
```

### Build

```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash

# Build all packages
colcon build

# Source workspace
source install/setup.bash
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Full ROS2 Pipeline

The complete pipeline processes LiDAR data through ground segmentation, boundary detection, and centerline planning.

### Option 1: Using the Launch Script (Recommended)

```bash
./run_navigator.sh path/to/rosbag.mcap
```

This starts all nodes automatically:
1. Patchwork++ (ground segmentation)
2. Drivable Area Detector (boundary extraction)
3. Drivable Area Navigator (centerline planning)
4. RViz2 (visualization)
5. Rosbag playback

### Option 2: Manual Launch (3 Terminals)

**Terminal 1 - Patchwork++ (Ground Segmentation)**
```bash
ros2 launch patchworkpp patchworkpp.launch.py \
    cloud_topic:=/hesai/pandar \
    use_sim_time:=true
```

**Terminal 2 - Detector + Navigator**
```bash
ros2 launch drivable_area_navigator full_pipeline.launch.py \
    use_sim_time:=true
```

**Terminal 3 - Rosbag Playback**
```bash
ros2 bag play rosbag.mcap --clock --loop
```

### Option 3: Individual Node Launch

```bash
# Ground segmentation only
ros2 launch patchworkpp patchworkpp.launch.py cloud_topic:=/hesai/pandar

# Detector only (requires Patchwork++ running)
ros2 launch drivable_area_detector detector.launch.py

# Navigator only (requires Detector running)
ros2 launch drivable_area_navigator navigator.launch.py
```

---

## ROS2 Topics

### Patchwork++ (Ground Segmentation)

| Topic | Type | Description |
|-------|------|-------------|
| **Input** | | |
| `/hesai/pandar` | PointCloud2 | Raw LiDAR point cloud |
| **Output** | | |
| `/patchworkpp/ground` | PointCloud2 | Ground points |
| `/patchworkpp/nonground` | PointCloud2 | Non-ground points |

### Drivable Area Detector

| Topic | Type | Description |
|-------|------|-------------|
| **Input** | | |
| `/patchworkpp/ground` | PointCloud2 | Ground points from Patchwork++ |
| **Output** | | |
| `/road_surface` | PointCloud2 | Filtered road surface |
| `/road_boundaries` | MarkerArray | Detected left/right boundaries |

### Drivable Area Navigator

| Topic | Type | Description |
|-------|------|-------------|
| **Input** | | |
| `/road_boundaries` | MarkerArray | Boundaries from detector |
| **Output** | | |
| `/planned_centerline` | Path | Computed centerline path |

---

## Data

LiDAR recordings are **not versioned** — the bags used during development weigh 0.9 GB and 1.7 GB,
well past GitHub's 100 MB per-file limit. Bring your own ROS 2 bag instead.

Expected input: a ROS 2 MCAP bag publishing `sensor_msgs/PointCloud2` on the LiDAR topic
(`/hesai/pandar` by default, `--ros-args -p cloud_topic:=...` to change it). The point cloud must
carry the `intensity` and `ring` fields: road-surface extraction thresholds on intensity, and
boundary extraction walks the scan ring by ring.

The standalone simulators additionally read `/road_whole` and `/grid` when replaying a
pre-segmented ground-truth bag.

---

## Running Standalone Simulations

For development and testing without ROS2:

### Offline Navigation (Pre-computed Centerline)

```bash
python -m circuit_navigator.main --bag sy27_road_ground_truth2 --laps 2
```

| Option | Description |
|--------|-------------|
| `--bag PATH` | Path to MCAP bag directory |
| `--laps N` | Number of laps (0 = infinite) |
| `--headless` | Run without visualization |

### Real-Time Navigation (Simulated LiDAR)

```bash
python -m circuit_navigator_realtime.main --bag sy27_road_ground_truth2 --range 40.0 --laps 2
```

| Option | Description |
|--------|-------------|
| `--bag PATH` | Path to MCAP bag directory |
| `--range METERS` | LiDAR sensor range |
| `--laps N` | Number of laps (0 = infinite) |
| `--headless` | Run without visualization |
| `--no-metrics` | Disable quality metrics display |

---

## Launch Parameters

### Patchwork++ Parameters

```bash
ros2 launch patchworkpp patchworkpp.launch.py \
    cloud_topic:=/hesai/pandar \    # Input topic
    use_sim_time:=true \            # Use rosbag clock
    front_only:=true \              # Filter to front hemisphere (50% faster)
    bag_delay:=1.0 \                # Delay before bag playback
    bag_rate:=1.0                   # Playback rate
```

### Detector Parameters

```bash
ros2 launch drivable_area_detector detector.launch.py \
    use_sim_time:=true \
    params_file:=/path/to/params.yaml \
    enable_static_tf:=true
```

### Navigator Parameters

```bash
ros2 launch drivable_area_navigator navigator.launch.py \
    use_sim_time:=true \
    params_file:=/path/to/params.yaml
```

---

## Configuration Files

| File | Description |
|------|-------------|
| `src/drivable_area_detector/config/params.yaml` | Detector parameters |
| `src/drivable_area_navigator/config/params.yaml` | Navigator parameters |
| `qos_override.yaml` | ROS2 QoS profile overrides |

---

## Tuning Tools

Parameter tuning utilities in `tools/`:

```bash
# Road surface intensity tuning
python tools/road_surface_tuner.py

# Boundary extraction tuning
python tools/boundary_extraction_tuner.py

# Curve fitting tuning
python tools/curve_fitting_tuner.py

# Detection validation
python tools/detection_validator.py
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run_navigator.sh                    # Full pipeline launch script
├── qos_override.yaml                   # ROS2 QoS settings
│
├── .devcontainer/                      # ROS 2 Jazzy dev environment
│
├── patchwork-plusplus/                 # Ground segmentation — cloned, not versioned
│
├── src/
│   ├── drivable_area_detector/        # Boundary Detection (ROS2 Python)
│   │   ├── launch/detector.launch.py
│   │   ├── config/params.yaml
│   │   └── drivable_area_detector/
│   │       ├── 01_road_points_extraction.py
│   │       ├── 02_ring_boundary_extraction.py
│   │       └── rviz2_publisher.py
│   │
│   └── drivable_area_navigator/       # Centerline Planning (ROS2 Python)
│       ├── launch/
│       │   ├── navigator.launch.py
│       │   └── full_pipeline.launch.py
│       ├── config/params.yaml
│       └── drivable_area_navigator/
│           ├── boundary_receiver_node.py
│           ├── centerline_planner.py
│           └── edge_smoother.py
│
├── circuit_navigator/                  # Offline Simulation (Python)
│   ├── main.py
│   └── README.md
│
├── circuit_navigator_realtime/         # Real-Time Simulation (Python)
│   ├── main.py
│   └── README.md
│
├── shared/                             # Common Components
│   ├── navigation/pure_pursuit.py
│   └── simulation/bicycle_model.py
│
├── tools/                              # Tuning Utilities
│   ├── road_surface_tuner.py
│   ├── boundary_extraction_tuner.py
│   ├── curve_fitting_tuner.py
│   └── detection_validator.py
│
└── docs/
    ├── notes/                          # LiDAR & road-marking research notes
    └── images/
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ROS2 REAL-TIME PIPELINE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw LiDAR                                                              │
│   (/hesai/pandar)                                                        │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────┐                                           │
│   │   PATCHWORK++           │  Ground/non-ground segmentation           │
│   │   (patchworkpp)         │  Output: /patchworkpp/ground              │
│   └───────────┬─────────────┘                                           │
│               │                                                          │
│               ▼                                                          │
│   ┌─────────────────────────┐                                           │
│   │   DETECTOR              │  Road surface → Boundary extraction       │
│   │   (drivable_area_       │  • 01_road_points_extraction              │
│   │    detector)            │  • 02_ring_boundary_extraction            │
│   │                         │  Output: /road_boundaries                 │
│   └───────────┬─────────────┘                                           │
│               │                                                          │
│               ▼                                                          │
│   ┌─────────────────────────┐                                           │
│   │   NAVIGATOR             │  Boundary → Centerline planning           │
│   │   (drivable_area_       │  • boundary_receiver_node                 │
│   │    navigator)           │  • centerline_planner                     │
│   │                         │  Output: /planned_centerline              │
│   └─────────────────────────┘                                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                        STANDALONE SIMULATIONS                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   circuit_navigator/          circuit_navigator_realtime/               │
│   (Offline)                   (Real-time)                               │
│                                                                          │
│   MCAP → Grid → Skeleton     MCAP → Grid → LiDAR Sim → Edge Detection  │
│        → Centerline                      → Filtering → Centerline       │
│        → Pure Pursuit                    → Pure Pursuit                 │
│        → Bicycle Model                   → Bicycle Model                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Visualization in RViz2

Add these displays in RViz2:

| Display Type | Topic | Description |
|--------------|-------|-------------|
| PointCloud2 | `/patchworkpp/ground` | Ground points (green) |
| PointCloud2 | `/patchworkpp/nonground` | Non-ground points (red) |
| PointCloud2 | `/road_surface` | Filtered road surface |
| MarkerArray | `/road_boundaries` | Left/right boundaries |
| Path | `/planned_centerline` | Computed centerline |

---

## Troubleshooting

### No data received

Check QoS compatibility:
```bash
ros2 topic info /hesai/pandar --verbose
```

Use QoS override file:
```bash
ros2 bag play rosbag.mcap --qos-profile-overrides-path qos_override.yaml
```

### Nodes not synchronized

Ensure all nodes use simulation time:
```bash
ros2 param set /node_name use_sim_time true
```

### Build errors

```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

---

## Dependencies

### ROS2 Packages
- rclpy, rclcpp
- sensor_msgs, nav_msgs, geometry_msgs
- visualization_msgs
- tf2_ros, tf2_sensor_msgs

### Python Packages
- numpy, scipy
- scikit-learn, scikit-image
- matplotlib
- mcap, zstandard
- libeigen3-dev
- open3d (optional)

---

## References

- [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) - Lee et al., IROS 2022
- Pure Pursuit - Coulter, 1992

---

## Team

Built by three students at Université de Technologie de Compiègne, autumn 2025:

- Lara Coulombe
- Victor Dessenne
- Andrea Pruvost Lamy

---

## License

[MIT](LICENSE)
