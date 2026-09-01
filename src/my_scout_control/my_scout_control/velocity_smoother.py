#!/usr/bin/env python3
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class VelocitySmoother(Node):

    def __init__(self) -> None:
        super().__init__('velocity_smoother')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter(
            'input_topic',
            '/cmd_vel_raw'
        )

        self.declare_parameter(
            'output_topic',
            '/cmd_vel_pre_gate'
        )

        self.declare_parameter(
            'update_period_sec',
            0.05
        )

        self.declare_parameter(
            'linear_step',
            0.02
        )

        self.declare_parameter(
            'angular_step',
            0.02
        )

        self.declare_parameter(
            'command_timeout_sec',
            0.5
        )

        self.input_topic = str(
            self.get_parameter('input_topic').value
        )

        self.output_topic = str(
            self.get_parameter('output_topic').value
        )

        self.update_period = float(
            self.get_parameter(
                'update_period_sec'
            ).value
        )

        self.linear_step = float(
            self.get_parameter(
                'linear_step'
            ).value
        )

        self.angular_step = float(
            self.get_parameter(
                'angular_step'
            ).value
        )

        self.command_timeout = float(
            self.get_parameter(
                'command_timeout_sec'
            ).value
        )

        # ------------------------------------------------------------
        # ROS communication
        # ------------------------------------------------------------

        self.subscription = self.create_subscription(
            Twist,
            self.input_topic,
            self.command_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            self.output_topic,
            10
        )

        # ------------------------------------------------------------
        # Velocity state
        # ------------------------------------------------------------

        self.current_linear = 0.0
        self.target_linear = 0.0

        self.current_angular = 0.0
        self.target_angular = 0.0

        self.last_command_time = (
            self.get_clock().now()
        )

        # ------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------

        self.timer = self.create_timer(
            self.update_period,
            self.timer_callback
        )

        self.get_logger().info(
            'Velocity smoother started: '
            f'{self.input_topic} -> '
            f'{self.output_topic}'
        )

    # ================================================================
    # Command input
    # ================================================================

    def command_callback(
        self,
        msg: Twist
    ) -> None:

        self.target_linear = float(
            msg.linear.x
        )

        self.target_angular = float(
            msg.angular.z
        )

        self.last_command_time = (
            self.get_clock().now()
        )

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def move_toward(
        current: float,
        target: float,
        step: float
    ) -> float:

        if current < target:
            return min(
                current + step,
                target
            )

        if current > target:
            return max(
                current - step,
                target
            )

        return current

    # ================================================================
    # Main smoothing loop
    # ================================================================

    def timer_callback(self) -> None:

        elapsed = (
            self.get_clock().now()
            - self.last_command_time
        ).nanoseconds / 1e9

        # ------------------------------------------------------------
        # Fail-safe:
        # 일정 시간 동안 새 속도 명령이 없으면 목표 속도를 0으로 설정
        # ------------------------------------------------------------

        if elapsed > self.command_timeout:
            self.target_linear = 0.0
            self.target_angular = 0.0

        # ------------------------------------------------------------
        # Linear velocity smoothing
        # ------------------------------------------------------------

        self.current_linear = (
            self.move_toward(
                self.current_linear,
                self.target_linear,
                self.linear_step
            )
        )

        # ------------------------------------------------------------
        # Angular velocity smoothing
        # ------------------------------------------------------------

        self.current_angular = (
            self.move_toward(
                self.current_angular,
                self.target_angular,
                self.angular_step
            )
        )

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        command = Twist()

        command.linear.x = (
            self.current_linear
        )

        command.angular.z = (
            self.current_angular
        )

        self.publisher.publish(command)

    # ================================================================
    # Stop
    # ================================================================

    def stop(self) -> None:

        self.target_linear = 0.0
        self.target_angular = 0.0

        self.current_linear = 0.0
        self.current_angular = 0.0

        self.publisher.publish(
            Twist()
        )


def main(
    args: Optional[list[str]] = None
) -> None:

    rclpy.init(args=args)

    node = VelocitySmoother()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
