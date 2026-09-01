import os



from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node





def generate_launch_description() -> LaunchDescription:



    realsense_share = get_package_share_directory("realsense2_camera")



    realsense_launch = IncludeLaunchDescription(

        PythonLaunchDescriptionSource(

            os.path.join(

                realsense_share,

                "launch",

                "rs_launch.py",

            )

        ),

        launch_arguments={

            "align_depth.enable": "true",

            "enable_sync": "true",

            "pointcloud.enable": "false",

        }.items(),

    )




    # 실제 장착 위치가 다르면 반드시 실측값으로 바꾼다.

    # URDF가 같은 TF를 이미 발행한다면 이 노드는 삭제한다.

    tf_base_to_camera = Node(

        package="tf2_ros",

        executable="static_transform_publisher",

        name="tf_base_to_camera",

        arguments=[

            "--x", "-0.10",

            "--y", "0.0",

            "--z", "0.70",

            "--roll", "0.0",

            "--pitch", "0.0",

            "--yaw", "0.0",

            "--frame-id", "base_link",

            "--child-frame-id", "camera_link",

        ],

        output="screen",

    )



    depth_to_scan_node = Node(

        package="depthimage_to_laserscan",

        executable="depthimage_to_laserscan_node",

        name="depthimage_to_laserscan_node",

        remappings=[

            ("depth", "/camera/camera/depth/image_rect_raw"),

            ("depth_camera_info", "/camera/camera/depth/camera_info"),

            ("scan", "/depth_scan"),

        ],

        parameters=[

            {

                "scan_height": 50,

                "scan_time": 0.033,

                "range_min": 0.30,

                "range_max": 3.50,

                # Humble의 정확한 파라미터 이름은 output_frame이다.

                "output_frame": "camera_depth_frame",

            }

        ],

        output="screen",

    )



    yolo_node = Node(

        package="yolo_detector",

        executable="yolo_node",

        name="yolo_detector",

        parameters=[

            {

                "model_path": "yolov8n.pt",

                "confidence_threshold": 0.50,

                "min_depth_m": 0.30,

                "max_depth_m": 4.00,

                "max_rgb_depth_time_gap_sec": 0.20,

            }

        ],

        output="screen",

    )



    scan_merger_node = Node(

        package="scan_merger_pkg",

        executable="scan_merger",

        name="scan_merger",

        output="screen",

    )



    return LaunchDescription(

        [

            realsense_launch,

            tf_base_to_camera,

            depth_to_scan_node,

            yolo_node,

            scan_merger_node,

        ]

    )
