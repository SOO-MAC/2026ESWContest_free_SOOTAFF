import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    안정적인 지도 작성 전용 launch.

    주의:
    - scout_base는 별도 터미널에서 먼저 실행해야 한다.
    - odom -> base_link 정적 TF는 절대 만들지 않는다.
    - SLAM 입력은 /merged_scan이 아니라 RPLiDAR의 /scan만 사용한다.
    """
    my_mapping_share = get_package_share_directory("my_mapping_pkg")
    rplidar_share = get_package_share_directory("rplidar_ros")
    slam_toolbox_share = get_package_share_directory("slam_toolbox")
    
    table_mapper_node = Node(
        package="my_mapping_pkg",
        executable="table_mapper",
        name="table_mapper",
        output="screen",
        parameters=[
            os.path.join(
                my_mapping_share,
                "config",
                "table_mapper.yaml",
            )
        ],
    )

    slam_params_file = os.path.join(
        my_mapping_share,
        "config",
        "my_mapper.yaml",
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                rplidar_share,
                "launch",
                "rplidar_a2m8_launch.py",
            )
        ),
        launch_arguments={
            "serial_port": "/dev/ttyUSB0",
            "frame_id": "laser_1",
            "inverted": "true",
            "baudrate": "115200",
            "scan_mode": "Standard",
        }.items(),
    )

    # 실제 장착 위치가 다르면 x/y/z 및 roll/pitch/yaw를 실측값으로 수정한다.
    # robot_state_publisher가 같은 TF를 이미 발행한다면 이 노드는 삭제한다.
    tf_base_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_base_to_laser",
        arguments=[
            "--x", "0.20",
            "--y", "0.0",
            "--z", "0.28",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "3.14159265",
            "--frame-id", "base_link",
            "--child-frame-id", "laser_1",
        ],
        output="screen",
    )

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                slam_toolbox_share,
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            "slam_params_file": slam_params_file,
            "use_sim_time": "false",
        }.items(),
    )

    return LaunchDescription(
        [
            rplidar_launch,
            tf_base_to_laser,
            slam_toolbox_launch,
            table_mapper_node,
        ]
    )
