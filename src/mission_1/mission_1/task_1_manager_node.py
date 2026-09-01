#!/usr/bin/env python3
"""
Mission 1 Task Manager Node - 테이블 자리 세팅

흐름 (자리마다 반복)
  1. /scout/move 로 SCOUT을 자리 앞으로 이동 (my_scout_control 연동)
  1-1. 리프트 상승 (mock)
  2. /arm/begin_seat  : 제어단 ArUco world 캐시 무효화
  3. /mission1/start_detect : 검출 노드 OCR 추적 상태 리셋
  4. /arm/pick_place x5 : name_tag -> snack x3 -> bottle
       - Task Manager는 "무엇을(category) 어디에(ArUco 기준 offset)" 만 지정
       - 스캔 자세 이동/검출/FK/IK/파지는 전부 제어단 내부에서 처리
  5. /arm/move_to_overview_pose 후 개수 검증
  6. 부족하면 부족분만 재배치 (최대 MAX_RECOVERY_RETRY회)
  7. 리프트 하강 후 다음 자리로

★ 왜 pick 대상 좌표를 안 넘기나
  제어단이 스캔 자세로 이동한 뒤에야 물체가 보인다. 따라서 TM은 요청 시점에
  좌표를 알 수 없다. 대신 응답의 picked로 실제 집은 물체를 확인한다.

★ 왜 place를 offset으로 넘기나
  TM이 아는 좌표는 flange 기준이라 팔이 움직이면 무효가 된다.
  제어단이 캐싱한 ArUco world 좌표에 offset을 더해 최종 위치를 만든다.
  offset은 base 축 기준으로 그대로 더한다 (마커 회전 미적용).

★ /mission1/detections 구독 용도
  검증 단계(overview 자세에서 개수 세기)에만 쓴다. pick 대상 선정에는 쓰지 않는다.
  pose가 flange 기준이라 팔이 움직인 뒤의 값은 무효이므로, overview 자세 도달
  시각 이후 header.stamp를 가진 메시지만 수용한다(stamp 게이팅).
"""

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point

from soomac_interfaces.msg import Detection, DetectionArray, MissionEvent, MissionResult
from soomac_interfaces.srv import MoveScout, PickPlace


MISSION_NAME = "mission1"

# Mission1 순회 대상은 venue_tables.yaml의
# seat_to_table에 연결된 실제 테이블에서 동적으로 결정한다.

# ArUco 기준 배치 offset. 단위 m, base 축 기준.
NAME_TAG_OFFSET = [0.0, 0.0, 0.0]
SNACK_OFFSETS = [
    [-0.10, 0.08, 0.0],
    [-0.10, 0.0, 0.0],
    [-0.10, -0.08, 0.0],
]
BOTTLE_OFFSET = [0.0, 0.10, 0.0]

REQUIRED_COUNTS = {
    Detection.CATEGORY_NAME_TAG: 1,
    Detection.CATEGORY_SNACK: 3,
    Detection.CATEGORY_BOTTLE: 1,
}

ARM_SERVICE_TIMEOUT_SEC = 180.0   # pick&place는 스캔 이동+파지까지 포함해 오래 걸린다
QUICK_SERVICE_TIMEOUT_SEC = 10.0
OVERVIEW_SETTLE_SEC = 0.5         # overview 자세 도달 후 새 프레임이 들어올 여유
VERIFY_WAIT_SEC = 3.0             # 게이팅 통과 메시지를 기다릴 최대 시간
FIRST_DETECTION_WAIT_SEC = 60.0   # 검출 노드 첫 메시지 대기 (YOLO/OCR 로딩 시간 감안)
MAX_ARM_RETRY = 2
MAX_RECOVERY_RETRY = 2


# ---- SCOUT (my_scout_control) ----
SCOUT_MOVE_SERVICE = "/scout/move"
SCOUT_CANCEL_SERVICE = "/scout/cancel"
# /scout/move는 blocking: 이동허가(move_enabled) 대기(무한) + Nav2 주행(최대 180초).
# TM은 이보다 넉넉히 기다린 뒤 포기하고 cancel을 보낸다.
SCOUT_SERVICE_TIMEOUT_SEC = 300.0
MAX_SCOUT_RETRY = 2


class Mission1TaskManagerNode(Node):
    def __init__(self):
        super().__init__("mission1_task_manager_node")

        self.declare_parameter("auto_start", False)
        # False면 scout 서비스가 없어도 이동을 생략하고 진행 (팔 단독 벤치 테스트용)
        self.declare_parameter("scout_required", True)
        self.declare_parameter("venue_yaml", "~/.ros/venue_tables.yaml")

        # ---- 검증 전용 검출 캐시 ----
        self.cache_lock = threading.Lock()
        self.latest_counts: Dict[str, int] = {}
        self.latest_stamp_ns: int = 0
        self.frame_ok = False
        self.got_first_detection = False

        self.create_subscription(DetectionArray, "/mission1/detections", self.on_detections, 10)

        # ---- 서비스 클라이언트 ----
        self.start_detect_client = self.create_client(Trigger, "/mission1/start_detect")
        self.begin_seat_client = self.create_client(Trigger, "/arm/begin_seat")
        self.pick_place_client = self.create_client(PickPlace, "/arm/pick_place")
        self.overview_client = self.create_client(Trigger, "/arm/move_to_overview_pose")
        self.home_client = self.create_client(Trigger, "/arm/home")
        self.scout_move_client = self.create_client(MoveScout, SCOUT_MOVE_SERVICE)
        self.scout_cancel_client = self.create_client(Trigger, SCOUT_CANCEL_SERVICE)

        self.status_pub = self.create_publisher(MissionEvent, "/mission1/task_status", 10)
        self.result_pub = self.create_publisher(MissionResult, "/mission1/mission_result", 10)

        self.scout_required = bool(self.get_parameter("scout_required").value)

        # GUI -> Mission1 시작 서비스
        self.mission_lock = threading.Lock()
        self.mission_running = False

        self.start_task_service = self.create_service(
            Trigger, "/mission1/start_task", self.on_start_task)

        if bool(self.get_parameter("auto_start").value):
            threading.Thread(target=self.run_mission1, daemon=True).start()

        self.get_logger().info("Mission1 Task Manager 시작")

    def on_start_task(self, request, response):
        del request

        with self.mission_lock:
            if self.mission_running:
                response.success = False
                response.message = "Mission1이 이미 실행 중입니다."
                return response

            self.mission_running = True

        threading.Thread(target=self._run_mission1_from_service, daemon=True).start()

        response.success = True
        response.message = "Mission1 시작 요청을 받았습니다."
        return response

    def _run_mission1_from_service(self):
        try:
            self.run_mission1()
        except Exception as error:
            self.get_logger().error(f"Mission1 예외 발생: {error}")
            self.fail_mission(f"Mission1 예외 발생: {error}")
        finally:
            with self.mission_lock:
                self.mission_running = False

    # ------------------------------------------------------------------
    # 검출 구독 (검증 전용)
    # ------------------------------------------------------------------
    def on_detections(self, msg: DetectionArray):
        counts: Dict[str, int] = {}
        for det in msg.detections:
            counts[det.category] = counts.get(det.category, 0) + 1
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.cache_lock:
            self.latest_counts = counts
            self.latest_stamp_ns = stamp_ns
            self.frame_ok = (msg.header.frame_id == "flange")
            self.got_first_detection = True

    def frame_is_flange(self) -> bool:
        with self.cache_lock:
            return self.frame_ok

    def wait_first_detection(self, timeout_sec: float = FIRST_DETECTION_WAIT_SEC) -> bool:
        """
        검출 노드의 첫 메시지를 기다린다.
        검출 노드는 YOLO/PaddleOCR 로딩에 수십 초가 걸리므로, 이 대기가 없으면
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

    def counts_after(self, gate_ns: int, timeout_sec: float) -> Optional[Dict[str, int]]:
        """
        gate_ns 이후에 발행된 메시지의 개수 결과를 기다린다.
        팔이 움직인 직후 이전 자세 기준 데이터를 쓰지 않기 위한 stamp 게이팅.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.cache_lock:
                if self.latest_stamp_ns > gate_ns:
                    return dict(self.latest_counts)
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------
    # 서비스 호출 헬퍼
    # ------------------------------------------------------------------
    def call_trigger(self, client, name: str, timeout_sec: float = QUICK_SERVICE_TIMEOUT_SEC) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_sec):
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
        """미션 스레드에서 future 완료를 대기. (rclpy.spin은 메인 스레드가 돌린다)"""
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error(f"{name} 서비스 timeout ({timeout_sec:.0f}s)")
                return False
            time.sleep(0.02)
        return True

    def pick_place_with_retry(self, category: str, place_offset: List[float],
                              place_mode: int = PickPlace.Request.PLACE_ARUCO_OFFSET,
                              place_yaw_deg: float = 0.0) -> bool:
        """
        /arm/pick_place 요청 + 재시도.

        대상 좌표는 넘기지 않는다 - 제어단이 스캔 자세로 이동해 직접 검출한다.
        실제로 뭘 집었는지는 응답의 picked로만 확인할 수 있다.
        """
        name = f"/arm/pick_place({category})"

        for attempt in range(1, MAX_ARM_RETRY + 1):
            self.get_logger().info(f"{category} pick&place 시도 {attempt}/{MAX_ARM_RETRY}")

            if not self.pick_place_client.wait_for_service(timeout_sec=QUICK_SERVICE_TIMEOUT_SEC):
                self.get_logger().warn("[STUB] /arm/pick_place 서비스 없음 (제어단 미구현)")
                self.get_logger().warn(f"{category} pick&place 실패: service unavailable")
                continue

            req = PickPlace.Request()
            req.header.stamp = self.get_clock().now().to_msg()
            req.category = category
            req.place_mode = place_mode
            req.place_offset = Point(x=float(place_offset[0]),
                                     y=float(place_offset[1]),
                                     z=float(place_offset[2]))
            req.place_yaw_deg = float(place_yaw_deg)

            future = self.pick_place_client.call_async(req)
            if not self.spin_future(future, name, ARM_SERVICE_TIMEOUT_SEC):
                self.get_logger().warn(f"{category} pick&place 실패: timeout")
                continue

            try:
                res = future.result()
            except Exception as e:
                self.get_logger().error(f"{name} 응답 오류: {e}")
                self.get_logger().warn(f"{category} pick&place 실패: {e}")
                continue

            if res.success:
                picked = res.picked
                self.publish_status(
                    f"{category} 배치 완료 (실제 집은 것: {picked.class_name}, "
                    f"conf={picked.confidence:.2f})")
                if picked.category and picked.category != category:
                    self.get_logger().warn(
                        f"요청 category={category} 인데 picked.category={picked.category} - 확인 필요")
                return True

            if res.nothing_detected:
                # 물체가 아예 없으면 재시도해도 같은 결과 -> 즉시 포기
                self.get_logger().error(f"{category} 미검출 (적재 공간 확인 필요)")
                return False

            self.get_logger().warn(f"{category} pick&place 실패: {res.message}")

        return False

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

    # ------------------------------------------------------------------
    # Mission 흐름
    # ------------------------------------------------------------------
    def run_mission1(self):
        self.get_logger().info("===== Mission 1 시작: 테이블 자리 세팅 =====")

        try:
            table_waypoints = self.load_service_table_waypoints()
        except Exception as error:
            return self.fail_mission(f"운용 테이블 목록 로드 실패: {error}")

        self.publish_status("Mission1 순회 테이블: " + ", ".join(table_waypoints),
                            event="table_list")

        for waypoint in table_waypoints:
            # seat_no라는 이름은 예전 코드와의 호환 때문에 남긴 것이고 값은 실제 table ID다.
            seat_no = int(waypoint.rsplit("_", 1)[1])
            self.publish_status(f"{seat_no}번 테이블 이동: {waypoint}")

            # 이전 테이블의 detection을 재사용하지 않는다.
            with self.cache_lock:
                self.latest_counts = {}
                self.latest_stamp_ns = 0
                self.frame_ok = False
                self.got_first_detection = False

            if not self.move_scout(waypoint):
                return self.fail_mission(f"{seat_no}번 자리 SCOUT 이동 실패")

            if not self.lift_control("up"):
                return self.fail_mission(f"{seat_no}번 자리 리프트 상승 실패")

            # 자리가 바뀌었으므로 제어단의 ArUco world 캐시를 버리게 한다
            if not self.call_trigger(self.begin_seat_client, "/arm/begin_seat"):
                self.publish_status("begin_seat 응답 없음 (계속 진행)")

            if not self.call_trigger(self.start_detect_client, "/mission1/start_detect"):
                self.publish_status("start_detect 응답 없음 (계속 진행)")

            # 검출 노드가 아직 안 떴을 수 있다 (YOLO/OCR 로딩 지연)
            if not self.wait_first_detection():
                return self.fail_mission("검출 노드 메시지 수신 실패 (detection_node 실행 확인)")

            if not self.setup_one_seat(seat_no):
                return self.fail_mission(f"{seat_no}번 자리 물체 배치 실패")

            ok, counts = self.verify_seat()
            if not ok:
                self.get_logger().warn(f"{seat_no}번 자리 검증 실패 {counts} -> 복구 시도")
                recovered = False
                for attempt in range(1, MAX_RECOVERY_RETRY + 1):
                    self.publish_status(f"{seat_no}번 자리 복구 {attempt}/{MAX_RECOVERY_RETRY}")
                    if not self.recover_missing(counts):
                        continue
                    ok, counts = self.verify_seat()
                    if ok:
                        recovered = True
                        break
                if not recovered:
                    return self.fail_mission(f"{seat_no}번 자리 최종 복구 실패 {counts}")

            self.publish_status(f"{seat_no}번 자리 검증 성공 {counts}")

            if not self.lift_control("down"):
                return self.fail_mission(f"{seat_no}번 자리 리프트 하강 실패")

        self.call_trigger(self.home_client, "/arm/home")
        self.move_scout("home")
        self.publish_mission_result(True, "Mission1 전체 테이블 세팅 성공")
        self.get_logger().info("===== Mission 1 전체 성공 =====")

    def setup_one_seat(self, seat_no: int) -> bool:
        """배치 순서: name_tag -> snack x3 -> bottle (스캔 자세는 제어단이 알아서 전환)"""
        self.publish_status(f"{seat_no}번 자리 물체 배치 시작")

        if not self.frame_is_flange():
            self.get_logger().error("frame=camera 강등 상태 (Hand-Eye 미적용) - pick 불가")
            return False

        if not self.pick_place_with_retry(Detection.CATEGORY_NAME_TAG, NAME_TAG_OFFSET):
            return False

        for i, offset in enumerate(SNACK_OFFSETS):
            if not self.pick_place_with_retry(Detection.CATEGORY_SNACK, offset):
                self.get_logger().error(f"snack {i + 1}번째 배치 실패")
                return False

        if not self.pick_place_with_retry(Detection.CATEGORY_BOTTLE, BOTTLE_OFFSET):
            return False

        self.publish_status(f"{seat_no}번 자리 물체 배치 완료")
        return True

    def verify_seat(self):
        """overview 자세로 이동 후 개수 검증. 반환 (ok, counts)"""
        if not self.call_trigger(self.overview_client, "/arm/move_to_overview_pose",
                                 timeout_sec=ARM_SERVICE_TIMEOUT_SEC):
            return False, {}

        # overview 자세 도달 이후에 발행된 프레임만 신뢰한다 (flange 좌표계가 바뀌었으므로)
        gate_ns = self.get_clock().now().nanoseconds
        time.sleep(OVERVIEW_SETTLE_SEC)

        counts = self.counts_after(gate_ns, VERIFY_WAIT_SEC)
        if counts is None:
            self.get_logger().error("overview 이후 검출 메시지 없음")
            return False, {}

        self.publish_status(f"검증 인식 결과: {counts}")
        for category, need in REQUIRED_COUNTS.items():
            if counts.get(category, 0) < need:
                return False, counts
        return True, counts

    def recover_missing(self, counts: Dict[str, int]) -> bool:
        """부족한 category만 부족한 개수만큼 재배치."""
        for category, need in REQUIRED_COUNTS.items():
            missing = need - counts.get(category, 0)
            if missing <= 0:
                continue
            self.get_logger().warn(f"{category} {missing}개 부족 -> 재배치")

            for i in range(missing):
                if category == Detection.CATEGORY_NAME_TAG:
                    offset = NAME_TAG_OFFSET
                elif category == Detection.CATEGORY_BOTTLE:
                    offset = BOTTLE_OFFSET
                else:
                    # 이미 놓인 개수 다음 자리부터 채운다
                    idx = min(counts.get(category, 0) + i, len(SNACK_OFFSETS) - 1)
                    offset = SNACK_OFFSETS[idx]

                if not self.pick_place_with_retry(category, offset):
                    return False
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

    def lift_control(self, command: str) -> bool:
        """리프트 상승/하강. command: "up" | "down". 제어단 미개발로 mock."""
        self.get_logger().info(f"[STUB] 리프트 제어 요청: {command}")
        time.sleep(0.2)
        return True

    # ------------------------------------------------------------------
    # 상태/결과 publish
    # ------------------------------------------------------------------
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
        self.lift_control("down")
        self.publish_mission_result(False, message)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = Mission1TaskManagerNode()
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
