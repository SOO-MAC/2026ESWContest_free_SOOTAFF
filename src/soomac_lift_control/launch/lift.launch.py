from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration("port")
    baudrate = LaunchConfiguration("baudrate")
    fake = LaunchConfiguration("fake")
    start_delay_sec = LaunchConfiguration("start_delay_sec")
    lift_up_sec = LaunchConfiguration("lift_up_sec")
    lift_down_sec = LaunchConfiguration("lift_down_sec")

    return LaunchDescription([
        DeclareLaunchArgument(
            "port",
            default_value="/dev/ttyACM0",
            description="Arduino serial port",
        ),

        DeclareLaunchArgument(
            "baudrate",
            default_value="115200",
            description="Arduino serial baudrate",
        ),

        DeclareLaunchArgument(
            "fake",
            default_value="false",
            description="Run without Arduino",
        ),

        DeclareLaunchArgument(
            "start_delay_sec",
            default_value="2.0",
            description="Delay before lift starts moving",
        ),

        DeclareLaunchArgument(
            "lift_up_sec",
            default_value="10.0",
            description="Lift up movement time",
        ),

        DeclareLaunchArgument(
            "lift_down_sec",
            default_value="3.0",
            description="Lift down movement time",
        ),

        Node(
            package="soomac_lift_control",
            executable="lift_serial_server",
            name="lift_serial_server",
            output="screen",
            arguments=[
                "--port", port,
                "--baudrate", baudrate,
                "--start-delay-sec", start_delay_sec,
                "--lift-up-sec", lift_up_sec,
                "--lift-down-sec", lift_down_sec,
            ],
            condition=UnlessCondition(fake),
        ),

        Node(
            package="soomac_lift_control",
            executable="lift_serial_server",
            name="lift_serial_server_fake",
            output="screen",
            arguments=[
                "--fake",
                "--port", port,
                "--baudrate", baudrate,
                "--start-delay-sec", start_delay_sec,
                "--lift-up-sec", lift_up_sec,
                "--lift-down-sec", lift_down_sec,
            ],
            condition=IfCondition(fake),
        ),
    ])
