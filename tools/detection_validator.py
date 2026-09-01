#!/usr/bin/env python3
"""
Detection Validator Tool - Frame-by-Frame Mode
===============================================
Interactive tool for validating road detection frame-by-frame.

Usage:
  python3 tools/detection_validator.py --node 01        # Validate Node 01
  python3 tools/detection_validator.py --node 02        # Validate Node 02
  python3 tools/detection_validator.py --node markers   # Validate RViz markers
  python3 tools/detection_validator.py --latest         # Resume last session

Makefile:
  make validate-detection NODE=01
  make validate-detection NODE=02
  make validate-detection NODE=markers

Node Options:
  01      : /road_surface (Road Points Extraction)
  02      : /ring_boundaries (Ring Boundary Extraction)
  markers : /road_boundaries (RViz MarkerArray - Node 03)

Controls:
- SPACE/ENTER: Advance to next frame
- 'g' or 'y': Mark current frame as GOOD and advance
- 'b' or 'n': Mark current frame as BAD and advance
- 'o': Add notes (for markers: shape quality)
- 'p': Toggle continuous playback
- 's': Save current validation session
- 'q': Quit and save

Output:
- JSON file with frame timestamps and validation labels
- NPZ files with point cloud data for each labeled frame
- Analysis report comparing good vs bad frames
"""

import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from rosbag2_interfaces.srv import PlayNext, TogglePaused, Resume, Pause
from std_srvs.srv import Trigger
import numpy as np
import json
import sys
import select
import termios
import tty
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
import threading
from pathlib import Path
import time


# Node configurations: maps --node argument to topics
NODE_CONFIGS = {
    '01': {
        'name': 'Road Points Extraction (Node 01)',
        'topic': '/road_surface',
        'reference_topic': '/patchworkpp/ground',
        'msg_type': 'PointCloud2',
    },
    '02': {
        'name': 'Ring Boundary Extraction (Node 02)',
        'topic': '/ring_boundaries',
        'reference_topic': '/road_surface',
        'msg_type': 'PointCloud2',
    },
    'markers': {
        'name': 'RViz Markers (Node 03)',
        'topic': '/road_boundaries',
        'reference_topic': '/ring_boundaries',
        'msg_type': 'MarkerArray',
    },
}


@dataclass
class FrameData:
    """Point cloud data for a single frame."""
    timestamp_sec: int
    timestamp_nanosec: int
    road_x: np.ndarray = field(default_factory=lambda: np.array([]))
    road_y: np.ndarray = field(default_factory=lambda: np.array([]))
    road_z: np.ndarray = field(default_factory=lambda: np.array([]))
    road_ring: np.ndarray = field(default_factory=lambda: np.array([]))
    road_intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_x: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_y: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_z: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_ring: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_intensity: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ValidationRecord:
    """Single validation record for a frame."""
    timestamp_sec: int
    timestamp_nanosec: int
    label: str
    notes: str = ""  # User notes, shape quality for markers
    num_road_points: int = 0
    num_ground_points: int = 0
    road_intensity_mean: float = 0.0
    road_intensity_std: float = 0.0
    road_intensity_min: float = 0.0
    road_intensity_max: float = 0.0
    ground_intensity_mean: float = 0.0
    ground_intensity_std: float = 0.0
    frame_id: str = ""
    # Marker-specific fields
    num_markers: int = 0
    shape_quality: str = ""  # 'good', 'gaps', 'jagged', 'wrong_width', etc.


class DetectionValidator(Node):
    def __init__(self, node_config: dict, resume_session: Path = None):
        super().__init__('detection_validator')

        self.node_config = node_config
        self.msg_type = node_config['msg_type']
        main_topic = node_config['topic']
        reference_topic = node_config['reference_topic']
        
        self.output_dir = Path('/home/ws/tools/validation_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume existing session or create new one
        if resume_session:
            self.session_dir = resume_session
            self.session_id = resume_session.name.replace('session_', '')
            self._load_session()
        else:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.output_dir / f'session_{self.session_id}'
            self.records: List[ValidationRecord] = []
            self.good_count = 0
            self.bad_count = 0
        
        self.good_dir = self.session_dir / 'good'
        self.bad_dir = self.session_dir / 'bad'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.good_dir.mkdir(exist_ok=True)
        self.bad_dir.mkdir(exist_ok=True)

        self.paused = True
        self.continuous_mode = False
        self.current_frame: Optional[FrameData] = None
        self.current_record: Optional[ValidationRecord] = None
        self.frame_count = len(self.records) if hasattr(self, 'records') else 0
        self.running = True
        self.pending_road: Dict[int, FrameData] = {}
        self.pending_ground: Dict[int, tuple] = {}
        self.waiting_for_frame = False
        self.frame_received = threading.Event()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe based on message type
        if self.msg_type == 'MarkerArray':
            self.road_sub = self.create_subscription(
                MarkerArray, main_topic, self.marker_callback, qos)
        else:
            self.road_sub = self.create_subscription(
                PointCloud2, main_topic, self.road_callback, qos)
        self.ground_sub = self.create_subscription(
            PointCloud2, reference_topic, self.ground_callback, qos)

        # Rosbag control services
        self.play_next_client = self.create_client(PlayNext, '/rosbag2_player/play_next')
        self.toggle_paused_client = self.create_client(TogglePaused, '/rosbag2_player/toggle_paused')
        self.pause_client = self.create_client(Pause, '/rosbag2_player/pause')
        self.resume_client = self.create_client(Resume, '/rosbag2_player/resume')

        self.session_file = self.session_dir / 'validation.json'
        self.old_settings = None
        self._setup_terminal()
        
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info('='*60)
        self.get_logger().info(f'Detection Validator - {node_config["name"]}')
        self.get_logger().info('='*60)
        self.get_logger().info(f'  Topic: {main_topic}')
        self.get_logger().info(f'  Reference: {reference_topic}')
        self.get_logger().info('  SPACE/ENTER: Next frame')
        self.get_logger().info('  g/y: GOOD | b/n: BAD | o: Notes | p: Continuous')
        self.get_logger().info('  s: Save | q: Quit')
        self.get_logger().info(f'  Session: {self.session_dir}')
        if resume_session:
            self.get_logger().info(f'  Resuming: {self.good_count} good, {self.bad_count} bad')
        self.get_logger().info('='*60)
        self.get_logger().info('')
        self.get_logger().info('⏸  Waiting for rosbag... Press SPACE to advance')
        self._check_rosbag_services()

    def _load_session(self):
        """Load existing session from JSON."""
        session_file = self.session_dir / 'validation.json'
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
            self.records = []
            for r in data.get('records', []):
                self.records.append(ValidationRecord(**r))
            self.good_count = data.get('good_count', 0)
            self.bad_count = data.get('bad_count', 0)
        else:
            self.records = []
            self.good_count = 0
            self.bad_count = 0

    def _setup_terminal(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            self.old_settings = None

    def _restore_terminal(self):
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass

    def _check_rosbag_services(self):
        """Check if rosbag services are available."""
        if not self.play_next_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('⚠  Rosbag services not available!')
            self.get_logger().warn('   Start rosbag with: ros2 bag play <bag> --clock --start-paused')
        else:
            self.get_logger().info('✓ Rosbag services connected')

    def _play_next_frame(self):
        """Request rosbag to play the next message."""
        if not self.play_next_client.service_is_ready():
            self.get_logger().warn('Rosbag service not ready')
            return False
        
        self.waiting_for_frame = True
        self.frame_received.clear()
        
        # Call play_next multiple times to get a full frame (ground + road)
        # Each call plays one message from the bag
        request = PlayNext.Request()
        for _ in range(10):  # Play up to 10 messages to get synchronized data
            future = self.play_next_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
            if self.frame_received.is_set():
                break
            time.sleep(0.01)
        
        self.waiting_for_frame = False
        return self.frame_received.is_set()

    def _toggle_continuous(self):
        """Toggle continuous playback mode."""
        self.continuous_mode = not self.continuous_mode
        
        if self.continuous_mode:
            # Resume rosbag playback
            if self.resume_client.service_is_ready():
                request = Resume.Request()
                self.resume_client.call_async(request)
            self.get_logger().info('▶  CONTINUOUS MODE - Press p to pause')
        else:
            # Pause rosbag playback
            if self.pause_client.service_is_ready():
                request = Pause.Request()
                self.pause_client.call_async(request)
            self.get_logger().info('⏸  FRAME-BY-FRAME MODE - Press SPACE to advance')

    def _keyboard_loop(self):
        while self.running:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    self._handle_key(sys.stdin.read(1))
            except:
                pass

    def _handle_key(self, key: str):
        if key == ' ' or key == '\n' or key == '\r':
            # Advance to next frame
            if not self.continuous_mode:
                self.get_logger().info('→ Next frame...')
                self._play_next_frame()
        elif key in 'pP':
            self._toggle_continuous()
        elif key in 'gyGY':
            self._label_current('good')
            if not self.continuous_mode:
                self._play_next_frame()
        elif key in 'bnBN':
            self._label_current('bad')
            if not self.continuous_mode:
                self._play_next_frame()
        elif key in 'oO':
            self._add_notes()
        elif key in 'sS':
            self._save_session()
        elif key in 'qQ':
            self.get_logger().info('Quitting...')
            self.running = False  # Signal main loop to exit

    def _add_notes(self):
        """Add notes to current record (for markers: shape quality)."""
        if self.current_record is None:
            self.get_logger().warn('No frame to add notes to')
            return
        
        # Temporarily restore terminal for input
        self._restore_terminal()
        try:
            if self.msg_type == 'MarkerArray':
                print('\nShape quality (good/gaps/jagged/wrong_width/other): ', end='', flush=True)
            else:
                print('\nNotes: ', end='', flush=True)
            notes = input().strip()
            if notes:
                if self.msg_type == 'MarkerArray':
                    self.current_record.shape_quality = notes
                self.current_record.notes = notes
                self.get_logger().info(f'Added notes: {notes}')
        finally:
            self._setup_terminal()

    def _parse_pointcloud(self, msg: PointCloud2) -> tuple:
        if len(msg.data) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        data = np.frombuffer(msg.data, dtype=np.uint8)
        points = data.reshape(-1, msg.point_step)
        x = points[:, 0:4].view(np.float32).ravel()
        y = points[:, 4:8].view(np.float32).ravel()
        z = points[:, 8:12].view(np.float32).ravel()
        intensity = points[:, 12:16].view(np.float32).ravel() if msg.point_step >= 16 else np.zeros_like(x)
        ring = points[:, 16:18].view(np.uint16).ravel() if msg.point_step >= 18 else np.zeros_like(x)
        return x, y, z, intensity, ring

    def _get_timestamp_key(self, msg: PointCloud2) -> int:
        return msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec

    def _label_current(self, label: str):
        if self.current_frame is None:
            self.get_logger().warn('No frame to label')
            return

        self.current_record.label = label
        self.records.append(self.current_record)

        # Export NPZ
        output_dir = self.good_dir if label == 'good' else self.bad_dir
        np.savez_compressed(
            output_dir / f'frame_{self.frame_count:06d}.npz',
            road_x=self.current_frame.road_x,
            road_y=self.current_frame.road_y,
            road_z=self.current_frame.road_z,
            road_ring=self.current_frame.road_ring,
            road_intensity=self.current_frame.road_intensity,
            ground_x=self.current_frame.ground_x,
            ground_y=self.current_frame.ground_y,
            ground_z=self.current_frame.ground_z,
            ground_ring=self.current_frame.ground_ring,
            ground_intensity=self.current_frame.ground_intensity,
            timestamp_sec=self.current_frame.timestamp_sec,
            timestamp_nanosec=self.current_frame.timestamp_nanosec
        )

        if label == 'good':
            self.good_count += 1
        else:
            self.bad_count += 1

        self.get_logger().info(
            f'[{"✓" if label == "good" else "✗"}] Frame {self.frame_count} → {label.upper()} | '
            f'Road: {len(self.current_frame.road_x)} | G:{self.good_count} B:{self.bad_count}'
        )
        self.current_frame = None
        self.current_record = None

    def marker_callback(self, msg: MarkerArray):
        """Handle MarkerArray messages for RViz marker validation."""
        self.frame_count += 1
        
        # Extract marker info
        num_markers = len(msg.markers)
        total_points = sum(len(m.points) for m in msg.markers)
        
        # Use first marker's header for timestamp
        if num_markers > 0:
            ts_sec = msg.markers[0].header.stamp.sec
            ts_nanosec = msg.markers[0].header.stamp.nanosec
            frame_id = msg.markers[0].header.frame_id
        else:
            ts_sec = 0
            ts_nanosec = 0
            frame_id = ''
        
        # Create a minimal frame data (markers don't have x,y,z arrays)
        frame = FrameData(
            timestamp_sec=ts_sec,
            timestamp_nanosec=ts_nanosec,
            road_x=np.array([]), road_y=np.array([]),
            road_z=np.array([]), road_intensity=np.array([])
        )
        
        self.current_frame = frame
        self.current_record = ValidationRecord(
            timestamp_sec=ts_sec,
            timestamp_nanosec=ts_nanosec,
            label='unlabeled',
            num_markers=num_markers,
            num_road_points=total_points,  # Repurpose as total marker points
            frame_id=frame_id
        )
        
        # Signal that we received a frame
        self.frame_received.set()
        
        # Display marker info
        self.get_logger().info(
            f'Frame {self.frame_count:4d} | Markers: {num_markers} | Points: {total_points} | '
            f'[g=GOOD, b=BAD, o=SHAPE, SPACE=skip]'
        )

    def road_callback(self, msg: PointCloud2):
        self.frame_count += 1
        x, y, z, intensity, ring = self._parse_pointcloud(msg)
        ts_key = self._get_timestamp_key(msg)
        
        frame = FrameData(
            timestamp_sec=msg.header.stamp.sec,
            timestamp_nanosec=msg.header.stamp.nanosec,
            road_x=x, road_y=y, road_z=z, road_intensity=intensity, road_ring=ring,
        )
        
        if ts_key in self.pending_ground:
            gx, gy, gz, gi, gr = self.pending_ground.pop(ts_key)
            frame.ground_x, frame.ground_y, frame.ground_z, frame.ground_intensity, frame.ground_ring = gx, gy, gz, gi, gr
        else:
            self.pending_road[ts_key] = frame
        
        self.current_frame = frame
        self.current_record = ValidationRecord(
            timestamp_sec=msg.header.stamp.sec,
            timestamp_nanosec=msg.header.stamp.nanosec,
            label='unlabeled',
            num_road_points=len(x),
            num_ground_points=len(frame.ground_x),
            road_intensity_mean=float(np.mean(intensity)) if len(intensity) > 0 else 0.0,
            road_intensity_std=float(np.std(intensity)) if len(intensity) > 0 else 0.0,
            road_intensity_min=float(np.min(intensity)) if len(intensity) > 0 else 0.0,
            road_intensity_max=float(np.max(intensity)) if len(intensity) > 0 else 0.0,
            frame_id=msg.header.frame_id
        )

        # Signal that we received a frame
        self.frame_received.set()
        
        # Display frame info
        y_range = f"[{np.min(y):.1f}, {np.max(y):.1f}]" if len(y) > 0 else "[]"
        self.get_logger().info(
            f'Frame {self.frame_count:4d} | Road: {len(x):4d} pts | Y: {y_range} | '
            f'[g=GOOD, b=BAD, SPACE=skip]'
        )

    def ground_callback(self, msg: PointCloud2):
        x, y, z, intensity, ring = self._parse_pointcloud(msg)
        ts_key = self._get_timestamp_key(msg)
        
        if ts_key in self.pending_road:
            frame = self.pending_road.pop(ts_key)
            frame.ground_x, frame.ground_y, frame.ground_z, frame.ground_intensity, frame.ground_ring = x, y, z, intensity, ring
            self.current_frame = frame
            if self.current_record:
                self.current_record.num_ground_points = len(x)
                self.current_record.ground_intensity_mean = float(np.mean(intensity)) if len(intensity) > 0 else 0.0
        else:
            self.pending_ground[ts_key] = (x, y, z, intensity, ring)
        
        # Cleanup old pending data
        for d in (self.pending_road, self.pending_ground):
            if len(d) > 10:
                del d[min(d.keys())]

    def _save_session(self):
        session_data = {
            'session_id': self.session_id,
            'node_config': self.node_config,
            'total_frames': self.frame_count,
            'good_count': self.good_count,
            'bad_count': self.bad_count,
            'records': [asdict(r) for r in self.records]
        }
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        self.get_logger().info(f'Saved to {self.session_file}')

    def _generate_analysis(self):
        if not self.records:
            return
        good = [r for r in self.records if r.label == 'good']
        bad = [r for r in self.records if r.label == 'bad']
        
        analysis = {
            'good': {'count': len(good), 
                     'road_pts': np.mean([r.num_road_points for r in good]) if good else 0,
                     'intensity': np.mean([r.road_intensity_mean for r in good]) if good else 0},
            'bad': {'count': len(bad),
                    'road_pts': np.mean([r.num_road_points for r in bad]) if bad else 0,
                    'intensity': np.mean([r.road_intensity_mean for r in bad]) if bad else 0}
        }
        
        with open(self.session_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.get_logger().info('='*40)
        self.get_logger().info(f'GOOD: {len(good)} frames, {analysis["good"]["road_pts"]:.0f} pts, int={analysis["good"]["intensity"]:.1f}')
        self.get_logger().info(f'BAD:  {len(bad)} frames, {analysis["bad"]["road_pts"]:.0f} pts, int={analysis["bad"]["intensity"]:.1f}')

    def destroy_node(self):
        self._restore_terminal()
        if self.records:
            self._save_session()
            self._generate_analysis()
        super().destroy_node()


def find_latest_session() -> Path:
    """Find the most recent validation session directory."""
    output_dir = Path('/home/ws/tools/validation_output')
    if not output_dir.exists():
        return None
    
    sessions = sorted(output_dir.glob('session_*'), reverse=True)
    for session in sessions:
        if (session / 'validation.json').exists():
            return session
    return sessions[0] if sessions else None


def main():
    parser = argparse.ArgumentParser(
        description='Detection Validator - Frame-by-Frame Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/detection_validator.py --node 01        # Validate road extraction
  python3 tools/detection_validator.py --node 02        # Validate ring boundaries
  python3 tools/detection_validator.py --node markers   # Validate RViz markers
  python3 tools/detection_validator.py --latest         # Resume last session
"""
    )
    parser.add_argument(
        '--node', '-n',
        choices=['01', '02', 'markers'],
        default='01',
        help='Pipeline node to validate: 01=road_surface, 02=ring_boundaries, markers=road_boundaries (default: 01)'
    )
    parser.add_argument(
        '--latest', '-l',
        action='store_true',
        help='Resume the most recent validation session'
    )
    
    args = parser.parse_args()
    
    # Get node configuration
    node_config = NODE_CONFIGS[args.node]
    
    # Find session to resume
    resume_session = None
    if args.latest:
        resume_session = find_latest_session()
        if resume_session:
            print(f'Resuming session: {resume_session}')
        else:
            print('No previous session found, starting new session')
    
    rclpy.init()
    node = DetectionValidator(node_config, resume_session)
    try:
        # Use spin_once with timeout so we can check running flag
        while node.running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()