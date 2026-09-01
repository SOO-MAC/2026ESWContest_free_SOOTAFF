#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def realsense_include(use_realsense, realsense_launch_file):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                realsense_launch_file,
            ])
        ),
        condition=IfCondition(use_realsense),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
        }.items(),
    )


def generate_launch_description():
    use_realsense = LaunchConfiguration('use_realsense')
    realsense_launch_file = LaunchConfiguration('realsense_launch_file')

    weights_path = LaunchConfiguration('weights_path')
    handeye_path = LaunchConfiguration('handeye_path')
    color_topic = LaunchConfiguration('color_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    show_debug_windows = LaunchConfiguration('show_debug_windows')
    use_temp_aruco_if_missing = LaunchConfiguration('use_temp_aruco_if_missing')
    aruco_dictionary = LaunchConfiguration('aruco_dictionary')
    target_aruco_id = LaunchConfiguration('target_aruco_id')
    conf_thres = LaunchConfiguration('conf_thres')
    auto_start = LaunchConfiguration('auto_start')

    detection_node = Node(
        package='mission_1',
        executable='task_1_detection_node',
        name='mission1_detection_node',
        output='screen',
        parameters=[{
            'weights_path': weights_path,
            'conf_thres': conf_thres,
            'handeye_path': handeye_path,
            'color_topic': color_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'show_debug_windows': show_debug_windows,
            'use_temp_aruco_if_missing': use_temp_aruco_if_missing,
            'aruco_dictionary': aruco_dictionary,
            'target_aruco_id': target_aruco_id,
        }],
    )

    manager_node = Node(
        package='mission_1',
        executable='task_1_manager_node',
        name='mission1_task_manager_node',
        output='screen',
        parameters=[{
            'auto_start': auto_start,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_realsense', default_value='true'),
        DeclareLaunchArgument('realsense_launch_file', default_value='rs_launch.py'),
        DeclareLaunchArgument('weights_path', default_value='/home/seungwon/soomac_ws/src/mission_1/mission_1/best.pt'),
        DeclareLaunchArgument('handeye_path', default_value='/home/seungwon/soomac_ws/src/mission_1/mission_1/hand_eye_result.json'),
        DeclareLaunchArgument('color_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('show_debug_windows', default_value='true'),
        DeclareLaunchArgument('use_temp_aruco_if_missing', default_value='true'),
        DeclareLaunchArgument('aruco_dictionary', default_value='DICT_4X4_50'),
        DeclareLaunchArgument('target_aruco_id', default_value='-1'),
        DeclareLaunchArgument('conf_thres', default_value='0.60'),
        DeclareLaunchArgument('auto_start', default_value='true'),
        realsense_include(use_realsense, realsense_launch_file),
        detection_node,
        manager_node,
    ])
