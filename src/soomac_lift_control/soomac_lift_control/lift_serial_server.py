#!/usr/bin/env python3

import argparse
import threading
import time
from enum import Enum

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy,
    HistoryPolicy,
)

from std_msgs.msg import Bool
from std_srvs.srv import Trigger


try:
    import serial
except ImportError:
    serial = None


class LiftState(Enum):
    DOWN = "down"
    WAITING_UP = "waiting_up"
    MOVING_UP = "moving_up"
    UP = "up"
    WAITING_DOWN = "waiting_down"
    MOVING_DOWN = "moving_down"
    STOPPED = "stopped"


class LiftSerialServer(Node):
    def __init__(
        self,
        port: str,
        baudrate: int,
        fake: bool,
        start_delay_sec: float,
        lift_up_sec: float,
        lift_down_sec: float,
    ):
        super().__init__("lift_serial_server")

        self.port = port
        self.baudrate = baudrate
        self.fake = fake

        # 현재 리프트 동작 시간은 launch 파일 또는 실행 인자로 설정한다.
        # 현재 프로젝트 기준:
        #   UP   = 10.0 sec
        #   DOWN = 3.0 sec
        self.start_delay_sec = start_delay_sec
        self.lift_up_sec = lift_up_sec
        self.lift_down_sec = lift_down_sec

        self.arduino = None

        self.serial_lock = threading.Lock()
        self.motion_lock = threading.Lock()
        self.stop_event = threading.Event()

        # 처음 시작할 때는 리프트가 내려가 있다고 가정
        self.state = LiftState.DOWN

        self.callback_group = ReentrantCallbackGroup()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Scout 이동 가능 여부
        self.move_enabled_pub = self.create_publisher(
            Bool,
            "/scout/move_enabled",
            qos,
        )

        # Robot arm 작업 가능 여부
        self.robot_arm_work_enabled_pub = self.create_publisher(
            Bool,
            "/robot_arm/work_enabled",
            qos,
        )

        # Lift services
        self.create_service(
            Trigger,
            "/lift/up",
            self.handle_lift_up,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            "/lift/down",
            self.handle_lift_down,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            "/lift/stop",
            self.handle_lift_stop,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            "/lift/home",
            self.handle_lift_home,
            callback_group=self.callback_group,
        )

        # Robot arm service client
        # 리프트 UP 완료 후 이 서비스를 호출해서 로봇팔 작업 시작 신호를 보냄
        self.arm_begin_seat_client = self.create_client(
            Trigger,
            "/arm/begin_seat",
            callback_group=self.callback_group,
        )

        if self.fake:
            self.get_logger().warn("FAKE MODE ON: Arduino 없이 동작 시뮬레이션만 합니다.")
        else:
            self.connect_arduino()

        self.publish_move_enabled()
        self.publish_robot_arm_work_enabled()

        self.get_logger().info("Lift Serial Server Started")
        self.get_logger().info("Service ready: /lift/up")
        self.get_logger().info("Service ready: /lift/down")
        self.get_logger().info("Service ready: /lift/stop")
        self.get_logger().info("Service ready: /lift/home")
        self.get_logger().info("Topic ready: /scout/move_enabled")
        self.get_logger().info("Topic ready: /robot_arm/work_enabled")
        self.get_logger().info("Client ready: /arm/begin_seat")
        self.get_logger().info(
            f"Timing: delay={self.start_delay_sec}s, "
            f"up={self.lift_up_sec}s, down={self.lift_down_sec}s"
        )

    def connect_arduino(self):
        if serial is None:
            self.arduino = None
            self.get_logger().error("python3-serial이 설치되어 있지 않습니다.")
            return

        try:
            self.arduino = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.2,
                write_timeout=1.0,
            )

            # Arduino는 USB Serial 연결 시 리셋될 수 있음
            time.sleep(2.0)

            self.arduino.reset_input_buffer()
            self.arduino.reset_output_buffer()

            self.get_logger().info(
                f"Arduino connected: {self.port}, baudrate={self.baudrate}"
            )

        except Exception as error:
            self.arduino = None
            self.get_logger().error(f"Arduino connection failed: {error}")

    def publish_move_enabled(self):
        msg = Bool()

        # Scout 이동 가능 조건:
        # 리프트가 완전히 DOWN 상태일 때만 true
        msg.data = self.state == LiftState.DOWN

        self.move_enabled_pub.publish(msg)
        self.get_logger().info(f"/scout/move_enabled = {msg.data}")

    def publish_robot_arm_work_enabled(self):
        msg = Bool()

        # Robot arm 작업 가능 조건:
        # 리프트가 완전히 UP 상태일 때만 true
        msg.data = self.state == LiftState.UP

        self.robot_arm_work_enabled_pub.publish(msg)
        self.get_logger().info(f"/robot_arm/work_enabled = {msg.data}")

    def set_state(self, state: LiftState):
        self.state = state
        self.get_logger().info(f"Lift state: {self.state.value}")
        self.publish_move_enabled()
        self.publish_robot_arm_work_enabled()

    def send_serial(self, command: str):
        command = command.strip() + "\n"

        if self.fake:
            self.get_logger().info(f"[FAKE SERIAL SEND] {command.strip()}")
            return True, "fake serial sent"

        with self.serial_lock:
            if self.arduino is None or not self.arduino.is_open:
                self.get_logger().warn("Arduino disconnected. Reconnecting...")
                self.connect_arduino()

            if self.arduino is None or not self.arduino.is_open:
                return False, "Arduino not connected"

            try:
                self.get_logger().info(f"[SERIAL SEND] {command.strip()}")

                self.arduino.write(command.encode("utf-8"))
                self.arduino.flush()

                time.sleep(0.1)

                replies = []
                start_time = time.time()

                while time.time() - start_time < 0.3:
                    if self.arduino.in_waiting > 0:
                        line = (
                            self.arduino.readline()
                            .decode("utf-8", errors="ignore")
                            .strip()
                        )
                        if line:
                            replies.append(line)
                    else:
                        time.sleep(0.03)

                if replies:
                    self.get_logger().info("[ARDUINO] " + " / ".join(replies))

                return True, "serial sent"

            except Exception as error:
                self.get_logger().error(f"Serial write failed: {error}")
                return False, str(error)

    def wait_with_stop_check(self, seconds: float):
        start_time = time.time()

        while time.time() - start_time < seconds:
            if self.stop_event.is_set():
                return False
            time.sleep(0.05)

        return True

    def call_arm_begin_seat(self):
        # 로봇팔 서비스가 켜져 있는지 확인
        if not self.arm_begin_seat_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("/arm/begin_seat service is not ready.")
            return False

        request = Trigger.Request()
        future = self.arm_begin_seat_client.call_async(request)
        future.add_done_callback(self.handle_arm_begin_seat_response)

        self.get_logger().info("Trigger sent: /arm/begin_seat")
        return True

    def handle_arm_begin_seat_response(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"/arm/begin_seat call failed: {error}")
            return

        self.get_logger().info(
            f"/arm/begin_seat response: "
            f"success={response.success}, message='{response.message}'"
        )

    def run_motion(self, direction: str):
        with self.motion_lock:
            self.stop_event.clear()

            if direction == "up":
                self.set_state(LiftState.WAITING_UP)
                self.get_logger().info("Lift UP request received.")
                self.get_logger().info(f"{self.start_delay_sec}초 대기 후 UP 시작")

                if not self.wait_with_stop_check(self.start_delay_sec):
                    self.send_serial("STOP")
                    self.set_state(LiftState.STOPPED)
                    return False, "lift up canceled before start"

                ok, msg = self.send_serial("UP")
                if not ok:
                    self.set_state(LiftState.STOPPED)
                    return False, msg

                self.set_state(LiftState.MOVING_UP)
                self.get_logger().info(f"Lift UP moving for {self.lift_up_sec} sec")

                if not self.wait_with_stop_check(self.lift_up_sec):
                    self.send_serial("STOP")
                    self.set_state(LiftState.STOPPED)
                    return False, "lift up stopped"

                self.send_serial("STOP")
                self.set_state(LiftState.UP)

                # 핵심 추가:
                # 리프트 UP 완료 후 로봇팔 작업 시작 Trigger 전송
                arm_trigger_sent = self.call_arm_begin_seat()

                if arm_trigger_sent:
                    return True, "lift up complete, /arm/begin_seat trigger sent"

                return True, "lift up complete, but /arm/begin_seat not ready"

            if direction == "down":
                self.set_state(LiftState.WAITING_DOWN)
                self.get_logger().info("Lift DOWN request received.")
                self.get_logger().info(f"{self.start_delay_sec}초 대기 후 DOWN 시작")

                if not self.wait_with_stop_check(self.start_delay_sec):
                    self.send_serial("STOP")
                    self.set_state(LiftState.STOPPED)
                    return False, "lift down canceled before start"

                ok, msg = self.send_serial("DOWN")
                if not ok:
                    self.set_state(LiftState.STOPPED)
                    return False, msg

                self.set_state(LiftState.MOVING_DOWN)
                self.get_logger().info(f"Lift DOWN moving for {self.lift_down_sec} sec")

                if not self.wait_with_stop_check(self.lift_down_sec):
                    self.send_serial("STOP")
                    self.set_state(LiftState.STOPPED)
                    return False, "lift down stopped"

                self.send_serial("STOP")
                self.set_state(LiftState.DOWN)

                # DOWN 완료 후에는 로봇팔 시작 Trigger를 보내지 않는다.
                # 이때는 /scout/move_enabled = true가 되면서 Scout 이동 가능 상태가 됨.
                return True, "lift down complete"

            return False, "unknown direction"

    def handle_lift_up(self, request, response):
        success, message = self.run_motion("up")
        response.success = success
        response.message = message
        return response

    def handle_lift_down(self, request, response):
        success, message = self.run_motion("down")
        response.success = success
        response.message = message
        return response

    def handle_lift_stop(self, request, response):
        self.stop_event.set()

        ok, msg = self.send_serial("STOP")
        self.set_state(LiftState.STOPPED)

        response.success = ok
        response.message = "lift stopped" if ok else msg
        return response

    def handle_lift_home(self, request, response):
        # 홈 센서가 없으므로 home은 실제 원점복귀 보장 X
        # 안전하게 정지만 수행
        self.stop_event.set()

        ok, msg = self.send_serial("STOP")
        self.set_state(LiftState.STOPPED)

        response.success = ok
        response.message = "home sensor not available. lift stopped only."
        return response

    def destroy_node(self):
        if self.arduino is not None and self.arduino.is_open:
            self.arduino.close()

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)

    parser.add_argument(
        "--fake",
        action="store_true",
        help="Arduino 없이 fake 모드로 실행",
    )

    parser.add_argument("--start-delay-sec", type=float, default=2.0)
    parser.add_argument("--lift-up-sec", type=float, default=10.0)
    parser.add_argument("--lift-down-sec", type=float, default=3.0)

    # ros2 launch가 자동으로 붙이는 --ros-args 처리용
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = LiftSerialServer(
        port=args.port,
        baudrate=args.baudrate,
        fake=args.fake,
        start_delay_sec=args.start_delay_sec,
        lift_up_sec=args.lift_up_sec,
        lift_down_sec=args.lift_down_sec,
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
