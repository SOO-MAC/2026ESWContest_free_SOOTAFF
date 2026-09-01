import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    package_share = get_package_share_directory(
        'my_scout_control'
    )

    params_file = os.path.join(
        package_share,
        'config',
        'scout_control.yaml'
    )

    velocity_smoother = Node(
        package='my_scout_control',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[params_file],
    )

    cmd_vel_gate = Node(
        package='my_scout_control',
        executable='cmd_vel_gate',
        name='cmd_vel_gate',
        output='screen',
        parameters=[params_file],
    )

    scout_control = Node(
        package='my_scout_control',
        executable='scout_control',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        velocity_smoother,
        cmd_vel_gate,
        scout_control,
    ])
