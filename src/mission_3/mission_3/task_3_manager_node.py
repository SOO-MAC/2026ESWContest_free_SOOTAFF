#!/usr/bin/env python3
"""
Mission 3 Task Manager Node - 행사장 테이블 뒷정리

흐름
  1. /scout/move 로 SCOUT을 테이블 앞으로 이동 (my_scout_control 연동)
  2. /arm/begin_seat        : 제어단 ArUco world 캐시 무효화
  3. /mission3/start_detect : 검출 노드 상태 리셋
  4. /arm/pick_place(name_tag, PLACE_ARUCO_OFFSET, offset=0)
       제어단이 name_tag를 집어 ArUco 위치(원래 자리)에 되돌려 놓는다
  5. 귀중품 우선 수거 -> 쓰레기 수거, 각각 nothing_detected 가 올 때까지 반복
       PLACE_FIXED 모드: 제어단이 category별 고정 수거함에 놓는다
  6. /arm/home, SCOUT 복귀

★ 종료 판정을 TM이 개수로 세지 않는다
  PickPlace 응답의 nothing_detected 가 "스캔 자세에서 그 category가 하나도 안 보임"을
  뜻하므로, 그 신호가 오면 해당 category는 다 치운 것이다.
  (제어단이 실제로 본 결과라 TM이 따로 세는 것보다 정확하다)

★ /mission3/detections 구독 용도
  frame_id 확인(Hand-Eye 강등 감지)과 로그용. pick 대상 선정에는 쓰지 않는다.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional

import yaml

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point

from soomac_interfaces.msg import Detection, DetectionArray, MissionEvent, MissionResult
from soomac_interfaces.srv import MoveScout, PickPlace


MISSION_NAME = "mission3"
# target_waypoint는 기존 launch 호환용으로 남겨둔다.
# 실제 Mission3 순회 대상은 venue_tables.yaml의
# seat_to_table에 연결된 테이블에서 동적으로 결정한다.
TARGET_WAYPOINT = "table_1"

NAME_TAG_OFFSET = [0.0, 0.0, 0.0]   # ArUco 위치 그대로

ARM_SERVICE_TIMEOUT_SEC = 180.0
QUICK_SERVICE_TIMEOUT_SEC = 10.0
MAX_ARM_RETRY = 2
MAX_COLLECT_ITEMS = 30              # 무한루프 방지 상한
FIRST_DETECTION_WAIT_SEC = 60.0     # 검출 노드 첫 메시지 대기 (YOLO 로딩 시간 감안)


# ---- SCOUT (my_scout_control) ----
SCOUT_MOVE_SERVICE = "/scout/move"
SCOUT_CANCEL_SERVICE = "/scout/cancel"
# /scout/move는 blocking: 이동허가(move_enabled) 대기(무한) + Nav2 주행(최대 180초).
# TM은 이보다 넉넉히 기다린 뒤 포기하고 cancel을 보낸다.
SCOUT_SERVICE_TIMEOUT_SEC = 300.0
MAX_SCOUT_RETRY = 2


class Mission3TaskManagerNode(Node):
    def __init__(self):
        super().__init__("mission3_task_manager_node")

        self.declare_parameter("auto_start", False)
        # False면 scout 서비스가 없어도 이동을 생략하고 진행 (팔 단독 벤치 테스트용)
        self.declare_parameter("scout_required", True)
        self.declare_parameter("venue_yaml", "~/.ros/venue_tables.yaml")
        self.declare_parameter("target_waypoint", TARGET_WAYPOINT)
        self.target_waypoint = str(self.get_parameter("target_waypoint").value)

        self.cache_lock = threading.Lock()
        self.frame_ok = False
        self.last_counts = {}
        self.got_first_detection = False

        self.create_subscription(DetectionArray, "/mission3/detections", self.on_detections, 10)

        self.start_detect_client = self.create_client(Trigger, "/mission3/start_detect")
        self.begin_seat_client = self.create_client(Trigger, "/arm/begin_seat")
        self.pick_place_client = self.create_client(PickPlace, "/arm/pick_place")
        self.home_client = self.create_client(Trigger, "/arm/home")
        self.scout_move_client = self.create_client(MoveScout, SCOUT_MOVE_SERVICE)
        self.scout_cancel_client = self.create_client(Trigger, SCOUT_CANCEL_SERVICE)

        self.status_pub = self.create_publisher(MissionEvent, "/mission3/task_status", 10)
        self.result_pub = self.create_publisher(MissionResult, "/mission3/mission_result", 10)

        self.scout_required = bool(self.get_parameter("scout_required").value)

        # GUI -> Mission3 시작 서비스
        self.mission_lock = threading.Lock()
        self.mission_running = False

        self.start_task_service = self.create_service(
            Trigger, "/mission3/start_task", self.on_start_task)

        if bool(self.get_parameter("auto_start").value):
            threading.Thread(target=self.run_mission3, daemon=True).start()

        self.get_logger().info("Mission3 Task Manager 시작")

    def on_start_task(self, request, response):
        del request

        with self.mission_lock:
            if self.mission_running:
                response.success = False
                response.message = "Mission3가 이미 실행 중입니다."
                return response

            self.mission_running = True

        threading.Thread(target=self._run_mission3_from_service, daemon=True).start()

        response.success = True
        response.message = "Mission3 시작 요청을 받았습니다."
        return response

    def _run_mission3_from_service(self):
        try:
            self.run_mission3()
        except Exception as error:
            self.get_logger().error(f"Mission3 예외 발생: {error}")
            self.fail_mission(f"Mission3 예외 발생: {error}")
        finally:
            with self.mission_lock:
                self.mission_running = False

    def on_detections(self, msg: DetectionArray):
        counts = {}
        for det in msg.detections:
            counts[det.category] = counts.get(det.category, 0) + 1
        with self.cache_lock:
            self.frame_ok = (msg.header.frame_id == "flange")
            self.last_counts = counts
            self.got_first_detection = True

    def frame_is_flange(self) -> bool:
        with self.cache_lock:
            return self.frame_ok

    def wait_first_detection(self, timeout_sec: float = FIRST_DETECTION_WAIT_SEC) -> bool:
        """
        검출 노드의 첫 메시지를 기다린다.
        검출 노드는 YOLO 로딩에 수십 초가 걸리므로, 이 대기가 없으면
        frame_ok 초기값(False) 때문에 미션이 시작하자마자 실패한다.
        """
        deadline = time.time() + timeout_sec
        warned = False
        while time.time() < deadline:
            with self.cache_lock:
                if self.got_first_detection:
                    return True
            if not warned:
                self.publish_status("검출 노드 첫 메시지 대기 중...")
                warned = True
            time.sleep(0.1)
        return False

    def snapshot_counts(self) -> dict:
        with self.cache_lock:
            return dict(self.last_counts)

    def call_trigger(self, client, name: str, timeout_sec: float = QUICK_SERVICE_TIMEOUT_SEC) -> bool:
        if not client.wait_for_service(timeout_sec=QUICK_SERVICE_TIMEOUT_SEC):
            self.get_logger().warn(f"[STUB] {name} 서비스 없음 (제어단 미구현)")
            return False
        future = client.call_async(Trigger.Request())
        if not self.spin_future(future, name, timeout_sec):
            return False
        try:
            return bool(future.result().success)
        except Exception as e:
            self.get_logger().error(f"{name} 응답 오류: {e}")
            return False

    def spin_future(self, future, name: str, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error(f"{name} 서비스 timeout ({timeout_sec:.0f}s)")
                return False
            time.sleep(0.02)
        return True

    def pick_place_with_retry(self, category: str, place_mode: int,
                              place_offset: Optional[List[float]] = None):
        """
        /arm/pick_place 요청 + 재시도. 반환은 (success, nothing_detected).

        nothing_detected는 "스캔 자세에서 그 category가 하나도 안 보인다"는 제어단 신호다.
        재시도해도 같은 답이 나오므로 바로 빠져나와서, 상위가 수거 종료로 해석하게 둔다.
        """
        name = f"/arm/pick_place({category})"
        offset = place_offset if place_offset is not None else [0.0, 0.0, 0.0]

        for attempt in range(1, MAX_ARM_RETRY + 1):
            if not self.pick_place_client.wait_for_service(timeout_sec=QUICK_SERVICE_TIMEOUT_SEC):
                self.get_logger().warn("[STUB] /arm/pick_place 서비스 없음 (제어단 미구현)")
                self.get_logger().warn(
                    f"{category} pick&place 실패 {attempt}/{MAX_ARM_RETRY}: service unavailable")
                continue

            req = PickPlace.Request()
            req.header.stamp = self.get_clock().now().to_msg()
            req.category = category
            req.place_mode = place_mode
            req.place_offset = Point(x=float(offset[0]), y=float(offset[1]), z=float(offset[2]))
            req.place_yaw_deg = 0.0

            future = self.pick_place_client.call_async(req)
            if not self.spin_future(future, name, ARM_SERVICE_TIMEOUT_SEC):
                self.get_logger().warn(
                    f"{category} pick&place 실패 {attempt}/{MAX_ARM_RETRY}: timeout")
                continue

            try:
                res = future.result()
            except Exception as e:
                self.get_logger().error(f"{name} 응답 오류: {e}")
                self.get_logger().warn(
                    f"{category} pick&place 실패 {attempt}/{MAX_ARM_RETRY}: {e}")
                continue

            if res.success:
                picked = res.picked
                if picked is not None and picked.class_name:
                    yaw = f", yaw={picked.yaw_deg:+.0f}deg" if picked.yaw_valid else ""
                    self.publish_status(f"{category} 수거 완료: {picked.class_name}{yaw}")
                return True, False

            if res.nothing_detected:
                return False, True

            self.get_logger().warn(
                f"{category} pick&place 실패 {attempt}/{MAX_ARM_RETRY}: {res.message}")

        return False, False

    def load_service_table_waypoints(self) -> list[str]:
        """
        행사에서 실제 쓰는 테이블 목록을 읽는다.

        tables는 맵에서 검출된 전체 물리 테이블이고, seat_to_table은 GUI 좌석이
        연결된 실제 운용 테이블이다. Mission1/3은 seat_to_table에 등장하는
        table ID만 중복 제거해서 순서대로 돈다.
        """
        path = Path(str(self.get_parameter("venue_yaml").value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"venue YAML 없음: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        tables = payload.get("tables", [])
        if not isinstance(tables, list) or not tables:
            raise ValueError("venue YAML에 tables가 없습니다.")
        valid_table_ids = {
            int(table["id"]) for table in tables
            if isinstance(table, dict) and "id" in table
        }

        seat_to_table = payload.get("seat_to_table", {})
        if not isinstance(seat_to_table, dict) or not seat_to_table:
            raise ValueError("venue YAML의 seat_to_table이 비어 있습니다.")

        service_table_ids = set()
        for raw_table_id in seat_to_table.values():
            try:
                table_id = int(raw_table_id)
            except (TypeError, ValueError):
                raise ValueError("seat_to_table에 잘못된 table ID가 있습니다.")
            if table_id not in valid_table_ids:
                raise ValueError(
                    f"seat_to_table이 존재하지 않는 table_{table_id}을 가리킵니다.")
            service_table_ids.add(table_id)

        if not service_table_ids:
            raise ValueError("운용할 테이블이 없습니다.")

        return [f"table_{table_id}" for table_id in sorted(service_table_ids)]

    def run_mission3(self):
        self.get_logger().info("===== Mission 3 시작: 전체 테이블 뒷정리 =====")

        try:
            table_waypoints = self.load_service_table_waypoints()
        except Exception as error:
            return self.fail_mission(f"운용 테이블 목록 로드 실패: {error}")

        total_tables = len(table_waypoints)
        self.publish_status("Mission3 순회 테이블: " + ", ".join(table_waypoints),
                            event="table_list")

        for index, target in enumerate(table_waypoints, start=1):
            self.publish_status(f"[{index}/{total_tables}] {target} 이동 시작",
                                event="table_move_start")

            if not self.move_scout(target):
                return self.fail_mission(f"{target} SCOUT 이동 실패")

            # 이전 테이블 detection 상태 제거
            with self.cache_lock:
                self.frame_ok = False
                self.last_counts = {}
                self.got_first_detection = False

            if not self.call_trigger(self.begin_seat_client, "/arm/begin_seat"):
                self.publish_status("begin_seat 응답 없음 (계속 진행)")

            if not self.call_trigger(self.start_detect_client, "/mission3/start_detect"):
                self.publish_status("start_detect 응답 없음 (계속 진행)")

            if not self.wait_first_detection():
                return self.fail_mission(
                    f"{target}: 검출 노드 메시지 수신 실패 (detection_node 실행 확인)")

            if not self.frame_is_flange():
                return self.fail_mission(
                    f"{target}: frame=camera 강등 상태 (Hand-Eye 미적용) - pick 불가")

            # 1. name_tag를 ArUco 자리로 원위치
            ok, nothing = self.pick_place_with_retry(
                Detection.CATEGORY_NAME_TAG,
                PickPlace.Request.PLACE_ARUCO_OFFSET,
                NAME_TAG_OFFSET,
            )
            if not ok:
                reason = "name_tag 미검출" if nothing else "name_tag 반환 실패"
                return self.fail_mission(f"{target}: {reason}")
            self.publish_status(f"{target}: name_tag 원위치 완료")

            # 2. 귀중품 먼저 챙기고 그 다음 쓰레기
            for category in (Detection.CATEGORY_VALUABLES, Detection.CATEGORY_TRASH):
                if not self.collect_all(category):
                    return self.fail_mission(f"{target}: {category} 수거 실패")

            self.publish_status(f"[{index}/{total_tables}] {target} 정리 완료",
                                event="table_done")

        # 모든 운용 테이블 완료 후 복귀
        self.call_trigger(self.home_client, "/arm/home")
        if not self.move_scout("home"):
            return self.fail_mission("전체 테이블 정리 후 SCOUT home 복귀 실패")

        self.publish_mission_result(True, f"Mission3 전체 테이블 {total_tables}개 뒷정리 성공")
        self.get_logger().info("===== Mission 3 전체 성공 =====")

    def collect_all(self, category: str) -> bool:
        """
        해당 category가 안 보일 때까지 계속 수거.
        종료 판정은 제어단이 준 nothing_detected 신호로 한다.
        """
        self.publish_status(f"{category} 수거 시작", event="collect_start")
        collected = 0

        while collected < MAX_COLLECT_ITEMS:
            if not self.frame_is_flange():
                self.get_logger().error("frame=camera 강등 - 수거 중단")
                return False

            ok, nothing = self.pick_place_with_retry(
                category, PickPlace.Request.PLACE_FIXED)

            if nothing:
                self.publish_status(f"{category} 더 이상 없음 - 총 {collected}개 수거",
                                    event="collect_done")
                return True
            if not ok:
                return False
            collected += 1

        self.get_logger().warn(f"{category} 수거 상한({MAX_COLLECT_ITEMS}) 도달 - 중단")
        return True

    # ------------------------------------------------------------------
    # SCOUT 이동 (my_scout_control 연동)
    # ------------------------------------------------------------------
    def move_scout(self, target: str) -> bool:
        """
        /scout/move (MoveScout) 호출. Nav2 완주까지 blocking이라 오래 걸린다
        (이동허가 대기 + 주행. scout 쪽 nav timeout 180초).

        target 문자열은 scout_control_node의 target_to_pose가 해석한다:
          seat_N / table_N / pickup / home / stage_front 등 (venue_tables.yaml 기준)
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
                # TM은 포기하지만 Scout는 아직 주행/허가대기 중일 수 있다 -> 정리
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

    def publish_status(self, message: str, event: str = "status"):
        msg = MissionEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mission = MISSION_NAME
        msg.event = event
        msg.detail = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)

    def publish_mission_result(self, success: bool, message: str):
        msg = MissionResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mission = MISSION_NAME
        msg.phase = "mission"
        msg.success = success
        msg.message = message
        self.result_pub.publish(msg)

    def fail_mission(self, message: str):
        self.get_logger().error(message)
        self.publish_status(message, event="mission_fail")
        self.call_trigger(self.home_client, "/arm/home")
        self.publish_mission_result(False, message)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = Mission3TaskManagerNode()
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