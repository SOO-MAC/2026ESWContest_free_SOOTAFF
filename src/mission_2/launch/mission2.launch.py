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
    conf_thres = LaunchConfiguration('conf_thres')
    color_topic = LaunchConfiguration('color_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    show_debug_window = LaunchConfiguration('show_debug_window')

    target_waypoint = LaunchConfiguration('target_waypoint')
    presentation_wait_sec = LaunchConfiguration('presentation_wait_sec')
    max_detect_retry = LaunchConfiguration('max_detect_retry')
    max_pick_place_retry = LaunchConfiguration('max_pick_place_retry')
    max_recovery_retry = LaunchConfiguration('max_recovery_retry')

    detection_node = Node(
        package='mission_2',
        executable='task_2_detection_node',
        name='mission2_detection_node',
        output='screen',
        parameters=[{
            'weights_path': weights_path,
            'handeye_path': handeye_path,
            'conf_thres': conf_thres,
            'color_topic': color_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'show_debug_window': show_debug_window,
        }],
    )

    manager_node = Node(
        package='mission_2',
        executable='task_2_manager_node',
        name='mission2_task_manager_node',
        output='screen',
        parameters=[{
            'target_waypoint': target_waypoint,
            'presentation_wait_sec': presentation_wait_sec,
            'max_detect_retry': max_detect_retry,
            'max_pick_place_retry': max_pick_place_retry,
            'max_recovery_retry': max_recovery_retry,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_realsense', default_value='true'),
        DeclareLaunchArgument('realsense_launch_file', default_value='rs_launch.py'),
        DeclareLaunchArgument('weights_path', default_value='/home/ryeong/runs/segment/multi_seg/yolo11n_task_2_ver1-6/weights/best.pt'),
        DeclareLaunchArgument('handeye_path', default_value='/home/ryeong/calib/hand_eye_result.json'),
        DeclareLaunchArgument('conf_thres', default_value='0.83'),
        DeclareLaunchArgument('color_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('show_debug_window', default_value='true'),
        DeclareLaunchArgument('target_waypoint', default_value='presentation_zone'),
        DeclareLaunchArgument('presentation_wait_sec', default_value='10.0'),
        DeclareLaunchArgument('max_detect_retry', default_value='3'),
        DeclareLaunchArgument('max_pick_place_retry', default_value='2'),
        DeclareLaunchArgument('max_recovery_retry', default_value='2'),
        realsense_include(use_realsense, realsense_launch_file),
        detection_node,
        manager_node,
    ])
