#!/usr/bin/env python3
"""
Mission 2 Task Manager Node - 마이크 전달/회수

흐름
  1. /scout/move 로 SCOUT을 지정 좌표로 이동 (my_scout_control 연동)
  2. /mission2/start_detect
  3. Hand 검출 대기
  4. Hand distance_m 을 피드백으로 SCOUT을 0.3m까지 접근
     (P 제어, /cmd_vel_raw -> velocity_smoother -> cmd_vel_gate -> /cmd_vel)
  5. /arm/deliver_mic : 제어단이 바구니에서 마이크를 집어 손 위치에 전달
  6. 발표 종료 대기
  7. /arm/return_mic  : 제어단이 테이블 위 마이크를 집어 바구니로 회수
  8. /arm/home, SCOUT 복귀

★ 왜 arm 서비스가 Trigger인가 (mission1/3의 PickPlace와 다름)
  mission2는 pick/place 위치가 "고정 바구니" 아니면 "제어단이 직접 검출하는 손/마이크"라
  Task Manager가 넘겨줄 좌표가 없다. 제어단이 /mission2/detections를 직접 구독해
  손/마이크 좌표를 얻고, 바구니 위치는 자체 상수로 갖는다.

★ 접근 제어에 distance_m만 쓰는 이유
  distance_m은 카메라-물체 직선거리라서 flange 좌표계와 무관하다.
  접근 중에는 SCOUT만 움직이고 팔은 정지해 있으므로 stamp 게이팅이 필요 없다.
  (pose를 썼다면 게이팅이 필요했을 것)
"""

import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from soomac_interfaces.msg import Detection, DetectionArray, MissionEvent, MissionResult
from soomac_interfaces.srv import MoveScout, StartMission2


MISSION_NAME = "mission2"
TARGET_WAYPOINT = "seat_1"

# ---- 0.3m 접근 제어 ----
APPROACH_TARGET_M = 0.30
APPROACH_TOLERANCE_M = 0.03
APPROACH_KP = 0.4
APPROACH_MAX_SPEED = 0.15        # m/s, SCOUT 안전 속도 상한
APPROACH_TIMEOUT_SEC = 20.0
APPROACH_PERIOD_SEC = 0.1
MAX_MISSED_FRAMES = 5            # hand 연속 미검출 이 횟수면 정지

# ---- SCOUT (my_scout_control) ----
SCOUT_MOVE_SERVICE = "/scout/move"
SCOUT_CANCEL_SERVICE = "/scout/cancel"
SCOUT_MOVE_ENABLED_TOPIC = "/scout/move_enabled"
# /scout/move는 blocking: 이동허가(move_enabled) 대기(무한) + Nav2 주행(최대 180초)
SCOUT_SERVICE_TIMEOUT_SEC = 300.0
MAX_SCOUT_RETRY = 2

# 접근 제어 발행 토픽.
# 절대 /cmd_vel 로 직접 쏘지 말 것: my_scout_control의 cmd_vel_gate가 /cmd_vel의
# 유일한 publisher이고, 차단 상태에서는 0을 계속 발행하므로 두 publisher가 충돌한다.
# /cmd_vel_raw -> velocity_smoother -> cmd_vel_gate -> /cmd_vel 파이프라인의 입구로 쏜다.
# 이렇게 하면 move_enabled / emergency_stop 안전장치가 접근 제어에도 그대로 적용된다.
APPROACH_CMD_TOPIC = "/cmd_vel_raw"

MIC_VERIFY_WAIT_SEC = 3.0        # 회수 후 새 검출 프레임을 기다릴 최대 시간
MAX_RETURN_RETRY = 2             # 회수 검증 실패 시 재회수 시도 횟수

DETECTION_FRESH_SEC = 0.5
DETECT_WAIT_SEC = 3.0
MAX_DETECT_RETRY = 3
ARM_SERVICE_TIMEOUT_SEC = 180.0
QUICK_SERVICE_TIMEOUT_SEC = 10.0


class Mission2TaskManagerNode(Node):
    def __init__(self):
        super().__init__("mission2_task_manager_node")

        self.declare_parameter("auto_start", False)
        # False면 scout 서비스가 없어도 이동을 생략하고 진행 (팔 단독 벤치 테스트용)
        self.declare_parameter("scout_required", True)
        self.declare_parameter("approach_cmd_topic", APPROACH_CMD_TOPIC)
        self.declare_parameter("target_waypoint", TARGET_WAYPOINT)
        self.declare_parameter("presentation_wait_sec", 10.0)

        self.target_waypoint = str(self.get_parameter("target_waypoint").value)
        self.presentation_wait_sec = float(self.get_parameter("presentation_wait_sec").value)

        # ---- 검출 캐시 ----
        self.cache_lock = threading.Lock()
        self.latest = {}                 # category -> Detection
        self.latest_time = {}            # category -> wall clock
        self.frame_ok = False
        # 회수 검증용: "직전 한 프레임에 무엇이 있었나"를 그대로 보관한다.
        # latest/latest_time은 마지막으로 '보인' 시점만 남기므로 부재를 판단할 수 없다.
        self.last_msg_stamp_ns = 0
        self.last_msg_categories = set()

        self.create_subscription(DetectionArray, "/mission2/detections", self.on_detections, 10)

        self.scout_required = bool(self.get_parameter("scout_required").value)

        # GUI -> Mission2 시작 / 질문 종료 서비스
        self.mission_lock = threading.Lock()
        self.mission_running = False

        self.question_end_event = threading.Event()
        self.waiting_for_question_end = False

        self.start_task_service = self.create_service(
            StartMission2, "/mission2/start_task", self.on_start_task)
        self.question_end_service = self.create_service(
            Trigger, "/mission2/question_end", self.on_question_end)

        # 접근 제어 명령 (파이프라인 입구. 상단 APPROACH_CMD_TOPIC 주석 참조)
        self.cmd_vel_pub = self.create_publisher(
            Twist, str(self.get_parameter("approach_cmd_topic").value), 10)

        # 이동 허가 상태 (리프트 하강 완료 신호). 게이트가 차단 중이면 접근 명령이 무시된다.
        self.scout_move_enabled = None   # None = 미수신
        self.create_subscription(Bool, SCOUT_MOVE_ENABLED_TOPIC, self.on_move_enabled, 10)

        self.start_detect_client = self.create_client(Trigger, "/mission2/start_detect")
        self.scout_move_client = self.create_client(MoveScout, SCOUT_MOVE_SERVICE)
        self.scout_cancel_client = self.create_client(Trigger, SCOUT_CANCEL_SERVICE)
        self.deliver_client = self.create_client(Trigger, "/arm/deliver_mic")
        self.return_client = self.create_client(Trigger, "/arm/return_mic")
        self.home_client = self.create_client(Trigger, "/arm/home")

        self.status_pub = self.create_publisher(MissionEvent, "/mission2/task_status", 10)
        self.result_pub = self.create_publisher(MissionResult, "/mission2/mission_result", 10)

        if bool(self.get_parameter("auto_start").value):
            threading.Thread(target=self.run_mission2, daemon=True).start()

        self.get_logger().info("Mission2 Task Manager 시작")

    def on_start_task(self, request, response):
        table_id = int(request.table_id)

        if table_id <= 0:
            response.success = False
            response.message = "잘못된 table_id입니다."
            return response

        with self.mission_lock:
            if self.mission_running:
                response.success = False
                response.message = "Mission2가 이미 실행 중입니다."
                return response

            self.mission_running = True

        self.target_waypoint = f"seat_{table_id}"

        self.question_end_event.clear()
        self.waiting_for_question_end = False

        threading.Thread(target=self._run_mission2_from_service, daemon=True).start()

        response.success = True
        response.message = f"Mission2 시작 요청: seat_{table_id}"
        return response

    def on_question_end(self, request, response):
        del request

        with self.mission_lock:
            running = self.mission_running

        if not running:
            response.success = False
            response.message = "Mission2가 실행 중이 아닙니다."
            return response

        if not self.waiting_for_question_end:
            response.success = False
            response.message = "아직 질문 종료를 받을 단계가 아닙니다."
            return response

        self.question_end_event.set()

        response.success = True
        response.message = "질문 종료 신호를 받았습니다."
        return response

    def _run_mission2_from_service(self):
        try:
            self.run_mission2()
        except Exception as error:
            self.get_logger().error(f"Mission2 예외 발생: {error}")
            self.fail_mission(f"Mission2 예외 발생: {error}")
        finally:
            self.waiting_for_question_end = False
            self.question_end_event.clear()

            with self.mission_lock:
                self.mission_running = False

    def on_move_enabled(self, msg: Bool):
        self.scout_move_enabled = bool(msg.data)

    def on_detections(self, msg: DetectionArray):
        now = time.time()
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        cats = {d.category for d in msg.detections}
        with self.cache_lock:
            self.frame_ok = (msg.header.frame_id == "flange")
            self.last_msg_stamp_ns = stamp_ns
            self.last_msg_categories = cats
            for category in (Detection.CATEGORY_HAND, Detection.CATEGORY_MIC):
                items = [d for d in msg.detections if d.category == category]
                if items:
                    # 여러 개면 가장 가까운 것 (요청자 근처로 이미 이동했다는 전제)
                    self.latest[category] = min(items, key=lambda d: d.distance_m)
                    self.latest_time[category] = now

    def get_fresh(self, category: str, timeout_sec: float = DETECTION_FRESH_SEC) -> Optional[Detection]:
        with self.cache_lock:
            det = self.latest.get(category)
            t = self.latest_time.get(category, 0.0)
        if det is None or (time.time() - t) > timeout_sec:
            return None
        return det

    def detect_with_retry(self, category: str) -> Optional[Detection]:
        """DETECT_WAIT_SEC 동안 기다리기를 MAX_DETECT_RETRY번 반복한다."""
        for attempt in range(1, MAX_DETECT_RETRY + 1):
            self.get_logger().info(f"{category} 인식 대기 {attempt}/{MAX_DETECT_RETRY}")
            deadline = time.time() + DETECT_WAIT_SEC
            while time.time() < deadline:
                det = self.get_fresh(category)
                if det is not None:
                    return det
                time.sleep(0.05)
        self.get_logger().error(f"{category} 최종 인식 실패")
        return None

    def category_present_after(self, category: str, gate_ns: int,
                               timeout_sec: float = MIC_VERIFY_WAIT_SEC):
        """
        gate_ns 이후에 발행된 프레임에서 해당 category가 보이는지 확인한다.
        반환: True(보임) / False(안 보임) / None(새 프레임을 못 받음)

        팔이 움직인 뒤라 이전 프레임 결과는 의미가 없으므로 stamp 게이팅이 필수다.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.cache_lock:
                if self.last_msg_stamp_ns > gate_ns:
                    return category in self.last_msg_categories
            time.sleep(0.05)
        return None

    def verify_mic_returned(self) -> bool:
        """
        회수 검증: 테이블에서 마이크가 사라졌는지 확인한다.
        아직 보이면 /arm/return_mic 을 다시 호출해 재회수를 시도한다.

        주의: 이 검증은 회수 동작이 끝난 뒤 팔의 최종 자세에서 '테이블이 여전히
              시야에 들어온다'는 전제에서만 유효하다. 팔이 테이블을 못 보는 자세로
              끝나면 '미검출 = 회수 성공'으로 잘못 판정할 수 있으므로,
              제어단과 회수 후 복귀 자세를 맞춰둘 것.
        """
        for attempt in range(1, MAX_RETURN_RETRY + 2):
            gate_ns = self.get_clock().now().nanoseconds
            present = self.category_present_after(Detection.CATEGORY_MIC, gate_ns)

            if present is None:
                self.publish_status("회수 검증: 새 검출 프레임 없음 - 판정 불가",
                                    event="verify_fail")
                return False
            if not present:
                self.publish_status("회수 검증 성공: 테이블에서 마이크 미검출",
                                    event="verify_success")
                return True

            if attempt > MAX_RETURN_RETRY:
                break
            self.publish_status(
                f"회수 검증 실패: 마이크가 아직 보임 -> 재회수 {attempt}/{MAX_RETURN_RETRY}",
                event="verify_retry")
            if not self.call_trigger(self.return_client, "/arm/return_mic",
                                     timeout_sec=ARM_SERVICE_TIMEOUT_SEC):
                return False

        self.publish_status("회수 검증 최종 실패: 마이크가 계속 남아있음", event="verify_fail")
        return False

    def spin_future(self, future, name: str, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error(f"{name} 서비스 timeout ({timeout_sec:.0f}s)")
                return False
            time.sleep(0.02)
        return True

    def call_trigger(self, client, name: str, timeout_sec: float = QUICK_SERVICE_TIMEOUT_SEC) -> bool:
        if not client.wait_for_service(timeout_sec=QUICK_SERVICE_TIMEOUT_SEC):
            self.get_logger().warn(f"[STUB] {name} 서비스 없음 (제어단 미구현)")
            return False
        future = client.call_async(Trigger.Request())
        if not self.spin_future(future, name, timeout_sec):
            return False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"{name} 응답 오류: {e}")
            return False
        if not res.success:
            self.get_logger().warn(f"{name} 실패: {res.message}")
        return bool(res.success)

    def approach_to_hand(self) -> bool:
        """hand distance_m이 0.3m에 수렴할 때까지 /cmd_vel로 SCOUT 접근."""
        # 게이트가 차단 중이면 접근 명령이 /cmd_vel까지 도달하지 않는다.
        # 여기서 미리 알려주면 "명령은 나가는데 로봇이 안 움직이는" 상황을 빨리 파악할 수 있다.
        if self.scout_move_enabled is False:
            self.publish_status(
                "경고: move_enabled=False - 접근 명령이 cmd_vel_gate에서 차단됨 "
                "(리프트 하강 신호 확인)", event="approach_warn")
        elif self.scout_move_enabled is None:
            self.get_logger().info("move_enabled 상태 미수신 (게이트 미실행이면 무시)")

        self.publish_status(f"접근 시작 (목표 {APPROACH_TARGET_M:.2f}m)", event="approach_start")
        start = time.time()
        missed = 0

        while time.time() - start < APPROACH_TIMEOUT_SEC:
            hand = self.get_fresh(Detection.CATEGORY_HAND)
            if hand is None:
                missed += 1
                if missed >= MAX_MISSED_FRAMES:
                    self.stop_scout()
                    self.publish_status("hand 미검출 지속 -> 접근 중단", event="approach_fail")
                    return False
                time.sleep(APPROACH_PERIOD_SEC)
                continue

            missed = 0
            error = hand.distance_m - APPROACH_TARGET_M
            if abs(error) <= APPROACH_TOLERANCE_M:
                self.stop_scout()
                self.publish_status(f"목표 거리 도달 ({hand.distance_m:.2f}m)",
                                    event="approach_done")
                return True

            speed = max(-APPROACH_MAX_SPEED, min(APPROACH_MAX_SPEED, APPROACH_KP * error))
            twist = Twist()
            twist.linear.x = float(speed)
            self.cmd_vel_pub.publish(twist)
            time.sleep(APPROACH_PERIOD_SEC)

        self.stop_scout()
        self.publish_status("접근 timeout", event="approach_fail")
        return False

    def stop_scout(self):
        self.cmd_vel_pub.publish(Twist())

    def run_mission2(self):
        self.get_logger().info("===== Mission 2 시작: 마이크 전달/회수 =====")
        self.publish_status(f"이동 시작: {self.target_waypoint}")

        if not self.move_scout(self.target_waypoint):
            return self.fail_mission("SCOUT 이동 실패")

        if not self.call_trigger(self.start_detect_client, "/mission2/start_detect"):
            self.publish_status("start_detect 응답 없음 (계속 진행)")

        # ---- 전달 ----
        self.tts("손바닥을 보여주십시오.")
        if self.detect_with_retry(Detection.CATEGORY_HAND) is None:
            return self.fail_mission("hand 인식 실패")

        if not self.approach_to_hand():
            return self.fail_mission("0.3m 접근 실패")

        with self.cache_lock:
            frame_ok = self.frame_ok
        if not frame_ok:
            return self.fail_mission("frame=camera 강등 상태 (Hand-Eye 미적용) - 전달 불가")

        if not self.call_trigger(self.deliver_client, "/arm/deliver_mic",
                                 timeout_sec=ARM_SERVICE_TIMEOUT_SEC):
            return self.fail_mission("마이크 전달 실패")
        self.publish_result("delivery", True, "마이크 전달 성공")

        # ---- 발표 / Q&A 대기 ----
        self.waiting_for_question_end = True
        self.publish_status("Q&A 대기 - GUI의 질문 종료 버튼을 기다립니다.",
                            event="question_wait")
        self.tts("발표가 끝나면 마이크를 테이블 위에 올려주십시오.")

        self.question_end_event.wait()

        self.waiting_for_question_end = False
        self.publish_status("질문 종료 신호 수신 - 마이크 회수를 시작합니다.",
                            event="question_end")

        # ---- 회수 ----
        mic = self.detect_with_retry(Detection.CATEGORY_MIC)
        if mic is None:
            self.tts("마이크가 보이지 않습니다. 테이블 위에 올려주십시오.")
            mic = self.detect_with_retry(Detection.CATEGORY_MIC)
        if mic is None:
            return self.fail_mission("마이크 인식 실패")

        yaw_info = f", yaw={mic.yaw_deg:+.0f}deg" if mic.yaw_valid else " (yaw 무효)"
        self.publish_status(f"마이크 감지 {mic.distance_m:.2f}m{yaw_info} -> 회수")

        if not self.call_trigger(self.return_client, "/arm/return_mic",
                                 timeout_sec=ARM_SERVICE_TIMEOUT_SEC):
            return self.fail_mission("마이크 회수 실패")

        # 제어단이 success를 줘도 실제로 치워졌는지는 눈으로 확인한다
        # (헛집기 / 옮기다 떨어뜨림을 성공으로 넘기지 않기 위함)
        if not self.verify_mic_returned():
            self.publish_result("return", False, "마이크 회수 검증 실패")
            return self.fail_mission("마이크 회수 검증 실패")
        self.publish_result("return", True, "마이크 회수 성공")

        self.call_trigger(self.home_client, "/arm/home")
        self.move_scout("home")
        self.publish_mission_result(True, "Mission2 마이크 전달/회수 성공")
        self.get_logger().info("===== Mission 2 전체 성공 =====")

    # ------------------------------------------------------------------
    # SCOUT 이동 (my_scout_control 연동)
    # ------------------------------------------------------------------
    def move_scout(self, target: str) -> bool:
        """
        /scout/move (MoveScout) 호출. Nav2 완주까지 blocking이라 오래 걸린다.
        target: seat_N / table_N / pickup / home 등 (venue_tables.yaml 기준)
        """
        if not self.scout_move_client.wait_for_service(timeout_sec=QUICK_SERVICE_TIMEOUT_SEC):
            if not self.scout_required:
                self.get_logger().warn(
                    f"[SKIP] /scout/move 없음 - scout_required=False라 이동 생략 ({target})")
                return True
            self.get_logger().error("/scout/move 서비스 없음 (scout_control_node 실행 확인)")
            return False

        for attempt in range(1, MAX_SCOUT_RETRY + 1):
            self.publish_status(f"SCOUT 이동 요청: {target} (시도 {attempt}/{MAX_SCOUT_RETRY})")
            req = MoveScout.Request()
            req.target = target
            future = self.scout_move_client.call_async(req)

            if not self.spin_future(future, f"/scout/move({target})", SCOUT_SERVICE_TIMEOUT_SEC):
                self.cancel_scout()
                return False

            try:
                res = future.result()
            except Exception as e:
                self.get_logger().error(f"/scout/move 응답 오류: {e}")
                return False

            if res.success:
                self.publish_status(f"SCOUT 이동 완료: {res.message}")
                return True
            self.get_logger().warn(f"/scout/move 실패: {res.message}")

        return False

    def cancel_scout(self):
        """진행 중인 Nav2 이동 취소 (best-effort)."""
        if not self.scout_cancel_client.wait_for_service(timeout_sec=2.0):
            return
        self.scout_cancel_client.call_async(Trigger.Request())
        self.get_logger().warn("/scout/cancel 요청 발행 (이동 중단)")

    def tts(self, text: str):
        # TODO: TTS 패키지 미제공. 현재는 상태 토픽 발행만 한다 (event="tts").
        # 스피커 노드가 생기면 /mission2/task_status 의 event=="tts" 를 구독해
        # detail을 재생하거나, 여기서 해당 서비스를 호출하도록 교체할 것.
        self.publish_status(f"[TTS] {text}", event="tts")

    def publish_status(self, message: str, event: str = "status"):
        msg = MissionEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mission = MISSION_NAME
        msg.event = event
        msg.detail = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)

    def publish_result(self, phase: str, success: bool, message: str):
        msg = MissionResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mission = MISSION_NAME
        msg.phase = phase
        msg.success = success
        msg.message = message
        self.result_pub.publish(msg)

    def publish_mission_result(self, success: bool, message: str):
        self.publish_result("mission", success, message)

    def fail_mission(self, message: str):
        self.get_logger().error(message)
        self.publish_status(message, event="mission_fail")
        self.stop_scout()
        self.call_trigger(self.home_client, "/arm/home")
        self.publish_mission_result(False, message)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = Mission2TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
