#!/bin/bash
# Run the full navigator pipeline
# Usage: ./run_navigator.sh [bag_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="${1:-$SCRIPT_DIR/spanlidar/spanlidar_0.mcap}"

echo "=============================================="
echo "  Drivable Area Navigator Pipeline"
echo "=============================================="
echo "  Bag: $BAG_PATH"
echo "=============================================="

# Source ROS2
source /opt/ros/jazzy/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Patchwork++ (ground segmentation)
echo "[1/4] Starting Patchwork++..."
ros2 launch patchworkpp patchworkpp.launch.py cloud_topic:=/points use_sim_time:=true &
sleep 2

# Start Detector (boundary extraction)
echo "[2/4] Starting Detector..."
ros2 launch drivable_area_detector detector.launch.py use_sim_time:=true &
sleep 2

# Start Navigator (centerline computation)
echo "[3/4] Starting Navigator..."
ros2 run drivable_area_navigator boundary_receiver_node --ros-args -p use_sim_time:=true &
sleep 1

# Start RViz2 in background
echo "[4/4] Starting RViz2..."
ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/src/drivable_area_detector/drivable_area.rviz" --ros-args -p use_sim_time:=true &
sleep 2

echo ""
echo "=============================================="
echo "  All nodes running!"
echo "  Add Path display for /planned_centerline"
echo "  Press Ctrl+C to stop all nodes"
echo "=============================================="
echo ""

# Play rosbag (foreground so Ctrl+C works)
echo "Playing rosbag..."
ros2 bag play "$BAG_PATH" --clock --loop --qos-profile-overrides-path "$SCRIPT_DIR/qos_override.yaml"

# Wait for all background jobs
wait
