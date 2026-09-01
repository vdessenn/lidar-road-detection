#!/usr/bin/env python3
"""
Full Pipeline launch file - launches detector + navigator together.

Pipeline:
1. Patchwork++ (ground segmentation) - External, run separately
2. Detector nodes (from drivable_area_detector)
3. Navigator nodes (from drivable_area_navigator)

Usage:
    # Terminal 1: Patchwork++
    ros2 launch patchwork-plusplus patchworkpp.launch.py

    # Terminal 2: Full pipeline (detector + navigator)
    ros2 launch drivable_area_navigator full_pipeline.launch.py

    # Terminal 3: Rosbag playback
    ros2 bag play rosbag/spanlidar/spanlidar_0.mcap --clock
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package share directories
    detector_pkg_share = FindPackageShare('drivable_area_detector')
    navigator_pkg_share = FindPackageShare('drivable_area_navigator')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time (from rosbag)'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    # Include detector launch
    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([detector_pkg_share, 'launch', 'detector.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Include navigator launch
    navigator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([navigator_pkg_share, 'launch', 'navigator.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    return LaunchDescription([
        use_sim_time_arg,
        detector_launch,
        navigator_launch,
    ])
