# =============================================================================
# Makefile for Drivable Area Detection Pipeline
# Packages: patchwork-plusplus (C++) + drivable_area_detector (Python)
# =============================================================================

SHELL := /bin/bash

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WS_ROOT          := $(shell pwd)
PATCHWORK_WS     := $(WS_ROOT)/patchwork-plusplus/ros
DETECTOR_WS      := $(WS_ROOT)/src
DETECTOR_PKG     := $(DETECTOR_WS)/drivable_area_detector

# Bag files
BAG_SPANLIDAR    := $(WS_ROOT)/rosbag/spanlidar/spanlidar_0.mcap
BAG_SY27         := $(WS_ROOT)/rosbag/sy27_road_ground_truth_live/sy27_road_ground_truth_live_0.mcap

# Default parameters
CLOUD_TOPIC      ?= /points
USE_SIM_TIME     ?= true
FRONT_ONLY       ?= true
BAG_RATE         ?= 1.0          # Playback rate (0.5 = half speed, 1.0 = normal)
BAG_DELAY        ?= 2.0          # Delay between loops (seconds)

# RViz config
RVIZ_CONFIG      := $(DETECTOR_PKG)/drivable_area.rviz

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------
.PHONY: all build build-patchwork build-detector \
        clean clean-patchwork clean-detector clean-all \
        run run-patchwork run-detector run-all run-nodes-only \
        run-bag run-bag-spanlidar run-bag-sy27 \
        run-bag-loop-spanlidar run-bag-loop-sy27 \
        rviz rviz-sim help \
        validate-detection validate-detection-no-rviz validate-detection-latest \
        tune-boundaries tune-curves tune-road-surface tune-markers

# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------
all: build

# -----------------------------------------------------------------------------
# Build targets
# -----------------------------------------------------------------------------
build: build-patchwork build-detector
	@echo "All packages built successfully"

build-patchwork:
	@echo "Building patchwork-plusplus..."
	unset VIRTUAL_ENV && \
	export PATH=$$(echo $$PATH | tr ':' '\n' | grep -v '.venv' | tr '\n' ':') && \
	source /opt/ros/jazzy/setup.bash && \
	colcon build --symlink-install --packages-select patchworkpp --base-paths $(PATCHWORK_WS)
	@echo "patchwork-plusplus built"

build-detector:
	@echo "Building drivable_area_detector..."
	unset VIRTUAL_ENV && \
	export PATH=$$(echo $$PATH | tr ':' '\n' | grep -v '.venv' | tr '\n' ':') && \
	source /opt/ros/jazzy/setup.bash && \
	colcon build --symlink-install --packages-select drivable_area_detector --base-paths $(DETECTOR_WS)
	@echo "drivable_area_detector built"

# -----------------------------------------------------------------------------
# Clean targets
# -----------------------------------------------------------------------------
clean: clean-patchwork clean-detector
	@echo "All packages cleaned"

clean-patchwork:
	@echo "Cleaning patchwork-plusplus..."
	rm -rf $(WS_ROOT)/build/patchworkpp $(WS_ROOT)/install/patchworkpp

clean-detector:
	@echo "Cleaning drivable_area_detector..."
	rm -rf $(WS_ROOT)/build/drivable_area_detector $(WS_ROOT)/install/drivable_area_detector $(WS_ROOT)/log

clean-all: clean
	rm -rf $(WS_ROOT)/build $(WS_ROOT)/install $(WS_ROOT)/log $(WS_ROOT)/core.*

# -----------------------------------------------------------------------------
# Run targets (individual packages)
# -----------------------------------------------------------------------------
run-patchwork:
	@echo "Launching patchwork-plusplus..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	ros2 launch patchworkpp patchworkpp.launch.py \
		cloud_topic:=$(CLOUD_TOPIC) \
		use_sim_time:=$(USE_SIM_TIME) \
		front_only:=$(FRONT_ONLY) \
		base_frame:=velodyne

run-detector:
	@echo "Launching drivable_area_detector..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	ros2 launch drivable_area_detector detector.launch.py \
		use_sim_time:=$(USE_SIM_TIME)

# -----------------------------------------------------------------------------
# Run ALL pipeline (patchwork + detector simultaneously)
# -----------------------------------------------------------------------------
run-all: run
run:
	@echo "Launching full pipeline (patchwork++ + detector)..."
	@echo "   - Cloud topic: $(CLOUD_TOPIC)"
	@echo "   - Use sim time: $(USE_SIM_TIME)"
	@echo ""
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=$(USE_SIM_TIME) \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=$(USE_SIM_TIME) & \
		wait \
	)

# -----------------------------------------------------------------------------
# Run NODES ONLY (no RViz, no bag playback)
# Connects to live LiDAR on topic /points
# -----------------------------------------------------------------------------
run-nodes-only:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║             Launching NODES ONLY (patchwork++ + detector)                ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║   • Cloud topic:  $(CLOUD_TOPIC)"
	@echo "║   • Use sim time: $(USE_SIM_TIME)"
	@echo "║   • No RViz"
	@echo "║   • No bag playback"
	@echo "║"
	@echo "║   Waiting for LiDAR data on topic $(CLOUD_TOPIC)..."
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=$(USE_SIM_TIME) \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=$(USE_SIM_TIME) & \
		wait \
	)

# -----------------------------------------------------------------------------
# Run with bag file playback
# -----------------------------------------------------------------------------
run-bag: run-bag-spanlidar

run-bag-spanlidar:
	@echo "Launching pipeline with spanlidar bag..."
	@echo "   - Playback rate: $(BAG_RATE)x"
	@echo "   - Loop delay: $(BAG_DELAY)s"
	@echo "   - use_sim_time: true (required with --clock)"
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SPANLIDAR) --clock --rate $(BAG_RATE) & \
		sleep 3 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		wait \
	)

run-bag-sy27:
	@echo "Launching pipeline with sy27_road_ground_truth_live bag..."
	@echo "   - Playback rate: $(BAG_RATE)x"
	@echo "   - Loop delay: $(BAG_DELAY)s"
	@echo "   - use_sim_time: true (required with --clock)"
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SY27) --clock --rate $(BAG_RATE) & \
		sleep 3 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		wait \
	)

# -----------------------------------------------------------------------------
# Run with bag file playback (LOOP MODE - with proper time reset handling)
# These targets use a wrapper script to handle clock jumps gracefully
# -----------------------------------------------------------------------------
run-bag-loop-spanlidar:
	@echo "Launching pipeline with spanlidar bag (LOOP MODE)..."
	@echo "   - Playback rate: $(BAG_RATE)x"
	@echo "   - Loop delay: $(BAG_DELAY)s"
	@echo "   - WARNING: Loop mode causes time jumps - RViz may reset"
	@echo ""
	@echo "   TIP: For stable visualization, use 'make run-bag-spanlidar' without loop"
	@echo ""
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SPANLIDAR) --clock --loop --rate $(BAG_RATE) --delay $(BAG_DELAY) & \
		sleep 3 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		wait \
	)

run-bag-loop-sy27:
	@echo "Launching pipeline with sy27 bag (LOOP MODE)..."
	@echo "   - Playback rate: $(BAG_RATE)x"
	@echo "   - Loop delay: $(BAG_DELAY)s"
	@echo "   - WARNING: Loop mode causes time jumps - RViz may reset"
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SY27) --clock --loop --rate $(BAG_RATE) --delay $(BAG_DELAY) & \
		sleep 3 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		wait \
	)

# -----------------------------------------------------------------------------
# RViz visualization
# -----------------------------------------------------------------------------
rviz:
	@echo "Launching RViz (wall clock time)..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	rviz2 -d $(RVIZ_CONFIG)

rviz-sim:
	@echo "Launching RViz with simulated time (for bag playback)..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	rviz2 -d $(RVIZ_CONFIG) --ros-args -p use_sim_time:=true

# -----------------------------------------------------------------------------
# Validation tools
# -----------------------------------------------------------------------------
# NODE options: 01, 02, markers
NODE ?= 01

validate-detection:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║          Detection Validator - Node $(NODE)                               ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║  Controls: SPACE=next | g=GOOD | b=BAD | o=notes | p=continuous | q=quit ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Starting pipeline with rosbag PAUSED (frame-by-frame)..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SPANLIDAR) --clock --start-paused & \
		sleep 2 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		sleep 2 && \
		python3 $(WS_ROOT)/tools/detection_validator.py --node $(NODE) & \
		rviz2 -d $(RVIZ_CONFIG) --ros-args -p use_sim_time:=true & \
		wait \
	)

validate-detection-no-rviz:
	@echo "Starting validation without RViz (Node $(NODE))..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SPANLIDAR) --clock --start-paused & \
		sleep 2 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		sleep 2 && \
		python3 $(WS_ROOT)/tools/detection_validator.py --node $(NODE) & \
		wait \
	)

validate-detection-latest:
	@echo "Resuming latest validation session..."
	source /opt/ros/jazzy/setup.bash && \
	source $(WS_ROOT)/install/setup.bash && \
	( \
		ros2 bag play $(BAG_SPANLIDAR) --clock --start-paused & \
		sleep 2 && \
		ros2 launch patchworkpp patchworkpp.launch.py \
			cloud_topic:=$(CLOUD_TOPIC) \
			use_sim_time:=true \
			front_only:=$(FRONT_ONLY) \
			base_frame:=velodyne & \
		ros2 launch drivable_area_detector detector.launch.py \
			use_sim_time:=true & \
		sleep 2 && \
		python3 $(WS_ROOT)/tools/detection_validator.py --latest & \
		rviz2 -d $(RVIZ_CONFIG) --ros-args -p use_sim_time:=true & \
		wait \
	)

# -----------------------------------------------------------------------------
# Tuning tools
# -----------------------------------------------------------------------------
tune-boundaries:
	@echo "Running boundary extraction tuner with latest session..."
	python3 $(WS_ROOT)/tools/boundary_extraction_tuner.py --latest

tune-curves:
	@echo "Running Node 02 curve fitting tuner with latest session..."
	python3 $(WS_ROOT)/tools/curve_fitting_tuner.py --latest

tune-road-surface:
	@echo "Running road surface (Node 01) filter tuner with latest session..."
	python3 $(WS_ROOT)/tools/road_surface_tuner.py --latest

# Backwards compatibility alias
tune-markers: tune-curves

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║           Drivable Area Detection - Makefile Commands                    ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ BUILD                                                                    ║"
	@echo "║   make build              Build all packages                             ║"
	@echo "║   make build-patchwork    Build patchwork-plusplus only                  ║"
	@echo "║   make build-detector     Build drivable_area_detector only              ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ RUN                                                                      ║"
	@echo "║   make run                Launch full pipeline (both packages)           ║"
	@echo "║   make run-nodes-only     Launch nodes ONLY (no RViz, no bag)            ║"
	@echo "║   make run-patchwork      Launch patchwork-plusplus only                 ║"
	@echo "║   make run-detector       Launch drivable_area_detector only             ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ RUN WITH BAG FILES (SINGLE PLAYBACK - RECOMMENDED)                       ║"
	@echo "║   make run-bag            Launch with spanlidar bag (no loop)            ║"
	@echo "║   make run-bag-spanlidar  Launch with spanlidar bag (no loop)            ║"
	@echo "║   make run-bag-sy27       Launch with sy27 ground truth bag (no loop)    ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ RUN WITH BAG FILES (LOOP MODE - TIME JUMPS EXPECTED)                     ║"
	@echo "║   make run-bag-loop-spanlidar  Loop spanlidar (RViz resets at loop)      ║"
	@echo "║   make run-bag-loop-sy27       Loop sy27 (RViz resets at loop)           ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ VISUALIZATION                                                            ║"
	@echo "║   make rviz               Launch RViz (wall clock time)                  ║"
	@echo "║   make rviz-sim           Launch RViz (simulated time - for bags)        ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ VALIDATION (Frame-by-frame analysis)                                     ║"
	@echo "║   make validate-detection         Validate with RViz (recommended)       ║"
	@echo "║   make validate-detection-no-rviz Validate without RViz (headless)       ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ TUNING (Parameter optimization)                                          ║"
	@echo "║   make tune-road-surface   Tune Node 01 (DBSCAN filter params)           ║"
	@echo "║   make tune-boundaries     Tune Node 02 boundary extraction params       ║"
	@echo "║   make tune-curves         Tune Node 02 curve fitting params             ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ CLEAN                                                                    ║"
	@echo "║   make clean              Clean all build artifacts                      ║"
	@echo "║   make clean-patchwork    Clean patchwork-plusplus only                  ║"
	@echo "║   make clean-detector     Clean drivable_area_detector only              ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════╣"
	@echo "║ PARAMETERS (override with make VAR=value)                                ║"
	@echo "║   CLOUD_TOPIC   = $(CLOUD_TOPIC)                                         ║"
	@echo "║   USE_SIM_TIME  = $(USE_SIM_TIME)                                        ║"
	@echo "║   FRONT_ONLY    = $(FRONT_ONLY)                                          ║"
	@echo "║   BAG_RATE      = $(BAG_RATE) (playback speed multiplier)                ║"
	@echo "║   BAG_DELAY     = $(BAG_DELAY)s (delay between loops)                    ║"
	@echo "║   NODE          = $(NODE) (01 / 02 / markers for validation)             ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  TIME SYNCHRONIZATION NOTES:"
	@echo "   • For bag playback, always use 'make rviz-sim' instead of 'make rviz'"
	@echo "   • Loop mode (--loop) causes 'jump back in time' - RViz will reset"
	@echo "   • Single playback mode is recommended for stable visualization"
	@echo ""
	@echo "  VALIDATION & TUNING WORKFLOW:"
	@echo "   1. Run: make validate-detection NODE=01  (or 02 or markers)"
	@echo "   2. Use SPACE to advance frame, 'g' for GOOD, 'b' for BAD"
	@echo "   3. Press 's' to save labels to NPZ file"
	@echo "   4. Run tuners to find optimal parameters:"
	@echo "      - make tune-road-surface  (Node 01 DBSCAN filter)"
	@echo "      - make tune-boundaries    (Node 02 boundary extraction)"
	@echo "      - make tune-curves        (Node 02 curve fitting)"
	@echo "   5. Update config/params.yaml with recommended values"
	@echo ""