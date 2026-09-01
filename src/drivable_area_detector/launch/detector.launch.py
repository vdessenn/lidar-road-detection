"""
Launch file pour la pipeline complète de détection de zone circulable (v02)
Pipeline simplifié: Patchwork++ → Extraction → Détection marquages → Limites → Visualisation

Nodes 01-03 (spatial filter, height transform, RANSAC) REMOVED - replaced by Patchwork++
New input: /patchworkpp/ground (from external Patchwork++ node)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package path
    pkg_share = FindPackageShare('drivable_area_detector')
    
    # Config file path
    config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'params.yaml'
    ])

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from rosbag'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=config_file,
        description='Full path to parameters YAML file'
    )

    # Static TF between pandora (LiDAR frame in bag) and base_link
    static_tf_enabled_arg = DeclareLaunchArgument(
        'enable_static_tf',
        default_value='true',
        description='Publish static transform pandora -> base_link'
    )
    
    static_tf_xyz = [
        DeclareLaunchArgument('static_tf_x', default_value='0.0'),
        DeclareLaunchArgument('static_tf_y', default_value='0.0'),
        DeclareLaunchArgument('static_tf_z', default_value='1.8'),
        DeclareLaunchArgument('static_tf_roll', default_value='0.0'),
        DeclareLaunchArgument('static_tf_pitch', default_value='0.0'),
        DeclareLaunchArgument('static_tf_yaw', default_value='0.0'),
    ]


    # ==========================================================================
    # v03 MINIMAL PIPELINE (3 nodes: 01→02→RANSAC)
    # - Patchwork++ handles ground segmentation (external C++ node)
    # - Node 01: Road surface extraction (intensity filter + noise filter)
    # - Node 02: Ring boundary extraction (y_min/y_max per ring)
    # - RANSAC viz: Polynomial fitting + RViz2 visualization
    #
    # REMOVED: nodes 03 (curves_smoothing) + 04 (marker_array) - redundant
    # ==========================================================================

    # [01] Road surface extraction (intensity filtering from /patchworkpp/ground)
    node_01 = Node(
        package='drivable_area_detector',
        executable='01_road_points_extraction_node',
        name='road_points_extraction_node',
        parameters=[LaunchConfiguration('params_file'),
                   {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # [02] Ring boundary extraction (y_min/y_max per x-bin)
    node_02 = Node(
        package='drivable_area_detector',
        executable='02_ring_boundary_extraction_node',
        name='ring_boundary_extraction_node',
        parameters=[LaunchConfiguration('params_file'),
                   {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # [03] Curves smoothing - REMOVED (redundant with RANSAC smoothing)
    # [04] Marker array - REMOVED (replaced by rviz2_publisher)

    # RANSAC Polynomial Fitting + RViz2 Publisher
    node_ransac_viz = Node(
        package='drivable_area_detector',
        executable='road_markers_publisher',
        name='road_markers_publisher',
        parameters=[LaunchConfiguration('params_file'),
                   {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # RViz2 avec config automatique (commenté pour phase 1)
    rviz_config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'drivable_area.rviz'
    ])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {
                'qos_overrides': {
                    # FIX P0: Unified QoS overrides for entire pipeline (all BEST_EFFORT)
                    '/points': {'reliability': 'best_effort'},
                    '/ROI_spatialfilter_points': {'reliability': 'best_effort'},
                    '/height_transformed_points': {'reliability': 'best_effort'},
                    '/voxel_points': {'reliability': 'best_effort'},
                    '/ground_segmentation_points': {'reliability': 'best_effort'},
                    '/road_surface': {'reliability': 'best_effort'},
                    '/detected_road_markings': {'reliability': 'best_effort'},
                    '/clustered_road_markings': {'reliability': 'best_effort'},
                    '/road_boundaries': {'reliability': 'best_effort'},
                    '/smoothed_road_boundaries': {'reliability': 'best_effort'},
                    '/road_boundaries': {'reliability': 'best_effort'}
                }
        }
        ],
        output='screen'
    )

    # Static transform publisher pandora -> base_link
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_velodyne_base_link',
        arguments=[
            LaunchConfiguration('static_tf_x'),
            LaunchConfiguration('static_tf_y'),
            LaunchConfiguration('static_tf_z'),
            LaunchConfiguration('static_tf_roll'),
            LaunchConfiguration('static_tf_pitch'),
            LaunchConfiguration('static_tf_yaw'),
            'velodyne',
            'base_link'
        ],
        condition=IfCondition(LaunchConfiguration('enable_static_tf')),
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        static_tf_enabled_arg,
        *static_tf_xyz,
        # Simplified pipeline: /patchworkpp/ground → [01] → [02] → [RANSAC viz]
        node_01,
        node_02,
        node_ransac_viz,  # RANSAC polynomial fitting + RViz2 visualization
        rviz_node,
        static_tf_node,
    ])