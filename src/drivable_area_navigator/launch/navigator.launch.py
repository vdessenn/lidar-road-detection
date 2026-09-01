#!/usr/bin/env python3
"""
Navigator launch file - launches the centerline computation node.

Nodes:
1. boundary_receiver_node - Parses boundaries, computes centerline
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('drivable_area_navigator')

    # Path to params file
    default_params_file = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time (from rosbag)'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to parameters file'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    # Boundary Receiver + Centerline Planner Node
    boundary_receiver_node = Node(
        package='drivable_area_navigator',
        executable='boundary_receiver_node',
        name='boundary_receiver',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        boundary_receiver_node,
    ])
