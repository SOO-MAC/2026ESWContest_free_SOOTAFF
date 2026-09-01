#!/usr/bin/env python3

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

from launch.substitutions import (
    LaunchConfiguration,
)

from launch_ros.actions import Node

from launch_ros.descriptions import (
    ParameterFile,
)


def generate_launch_description():

    # ============================================================
    # Package paths
    # ============================================================

    nav2_bringup_dir = (
        get_package_share_directory(
            'nav2_bringup'
        )
    )

    scout_package_dir = (
        get_package_share_directory(
            'my_scout_control'
        )
    )

    # ============================================================
    # Launch arguments
    # ============================================================

    map_yaml = LaunchConfiguration(
        'map'
    )

    params_file = LaunchConfiguration(
        'params_file'
    )

    use_sim_time = LaunchConfiguration(
        'use_sim_time'
    )

    autostart = LaunchConfiguration(
        'autostart'
    )

    # ============================================================
    # Parameter file
    # ============================================================

    configured_params = ParameterFile(
        params_file,
        allow_substs=True
    )

    # ============================================================
    # Saved-map localization
    #
    # Official Nav2 localization launch:
    #   map_server
    #   AMCL
    #   localization lifecycle manager
    # ============================================================

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_bringup_dir,
                'launch',
                'localization_launch.py'
            )
        ),

        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': autostart,
            'use_composition': 'False',
            'use_respawn': 'False',
            'container_name': 'nav2_container',
        }.items(),
    )

    # ============================================================
    # Controller
    #
    # Nav2 velocity command:
    # /cmd_vel -> /cmd_vel_raw
    # ============================================================

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],

        remappings=[
            (
                'cmd_vel',
                '/cmd_vel_raw'
            ),
        ],
    )

    # ============================================================
    # Path smoother
    #
    # NOTE:
    # nav2_smoother = path smoothing
    # velocity_smoother.py = velocity smoothing
    #
    # 서로 다른 역할.
    # ============================================================

    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],
    )

    # ============================================================
    # Planner
    # ============================================================

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],
    )

    # ============================================================
    # Behavior server
    #
    # Spin / Backup 등의 recovery 동작도
    # 반드시 custom smoother 쪽으로 보냄.
    # ============================================================

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],

        remappings=[
            (
                'cmd_vel',
                '/cmd_vel_raw'
            ),
        ],
    )

    # ============================================================
    # BT Navigator
    # ============================================================

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],
    )

    # ============================================================
    # Waypoint follower
    #
    # 향후 MODE 3 순찰 등에 사용 가능.
    # ============================================================

    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',

        parameters=[
            configured_params,
            {
                'use_sim_time':
                use_sim_time
            },
        ],
    )

    # ============================================================
    # Nav2 Lifecycle Manager
    #
    # 기본 Humble navigation launch와 달리
    # nav2_velocity_smoother는 여기 넣지 않음.
    # ============================================================

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',

        parameters=[
            {
                'use_sim_time':
                use_sim_time,

                'autostart':
                autostart,

                'node_names': [
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
                ],
            },
        ],
    )

    # ============================================================
    # Our Scout control stack
    #
    # velocity_smoother
    # cmd_vel_gate
    # scout_control_node
    # ============================================================

    scout_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                scout_package_dir,
                'launch',
                'scout_control.launch.py'
            )
        )
    )

    # ============================================================
    # Arguments
    # ============================================================

    declare_map = DeclareLaunchArgument(
        'map',
        description=(
            'Full path to the saved '
            'OccupancyGrid map YAML'
        )
    )

    declare_params = DeclareLaunchArgument(
        'params_file',

        default_value=os.path.join(
            scout_package_dir,
            'config',
            'nav2_params.yaml'
        ),

        description=(
            'Nav2 parameter YAML'
        )
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false'
    )

    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true'
    )

    # ============================================================
    # Launch
    # ============================================================

    return LaunchDescription([
        declare_map,
        declare_params,
        declare_use_sim_time,
        declare_autostart,

        localization,

        controller_server,
        smoother_server,
        planner_server,
        behavior_server,
        bt_navigator,
        waypoint_follower,

        lifecycle_manager,

        scout_control,
    ])
