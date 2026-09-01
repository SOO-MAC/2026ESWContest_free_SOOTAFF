#!/usr/bin/env python3
"""Final Scout velocity interlock.

Only this node should publish to the Scout driver's /cmd_vel topic.
The upstream Nav2/velocity-smoother output must be remapped to
/cmd_vel_pre_gate.

A non-zero Twist is forwarded only when:
  * /scout/move_enabled is True,
  * /scout/emergency_stop is False, and
  * the upstream command is fresh.
Otherwise a zero Twist is continuously published.
"""

from __future__ import annotations

import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class CmdVelGate(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_gate')

        self.declare_parameter('input_topic', '/cmd_vel_pre_gate')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('move_enabled_topic', '/scout/move_enabled')
        self.declare_parameter('emergency_stop_topic', '/scout/emergency_stop')
        self.declare_parameter('publish_period_sec', 0.05)
        self.declare_parameter('command_timeout_sec', 0.25)

        self.enabled = False
        self.move_state_received = False
        self.emergency_stop = False
        self.last_command = Twist()
        self.last_command_time = 0.0

        self.publisher = self.create_publisher(
            Twist,
            str(self.get_parameter('output_topic').value),
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('input_topic').value),
            self.command_callback,
            20,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('move_enabled_topic').value),
            self.enabled_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('emergency_stop_topic').value),
            self.emergency_callback,
            10,
        )
        self.create_timer(
            float(self.get_parameter('publish_period_sec').value),
            self.timer_callback,
        )
        self.publish_zero()
        self.get_logger().info(
            'cmd_vel gate ready; default state is BLOCKED until move_enabled=True.'
        )

    def command_callback(self, msg: Twist) -> None:
        self.last_command = msg
        self.last_command_time = time.monotonic()

    def enabled_callback(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        self.move_state_received = True
        if not self.enabled:
            self.publish_zero()

    def emergency_callback(self, msg: Bool) -> None:
        self.emergency_stop = bool(msg.data)
        if self.emergency_stop:
            self.publish_zero()

    def timer_callback(self) -> None:
        timeout = float(self.get_parameter('command_timeout_sec').value)
        command_fresh = (
            self.last_command_time > 0.0
            and time.monotonic() - self.last_command_time <= timeout
        )
        if self.enabled and not self.emergency_stop and command_fresh:
            self.publisher.publish(self.last_command)
        else:
            self.publish_zero()

    def publish_zero(self) -> None:
        self.publisher.publish(Twist())


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = CmdVelGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
