#!/usr/bin/env python3

"""
Scout Mini Nav2 movement service.

Responsibilities
----------------
- Provide /scout/move service
- Convert target name to PoseStamped using venue_tables.yaml
- Send goal to Nav2
- Wait for /scout/move_enabled before movement
- Cancel navigation if movement permission is removed
- Support emergency stop
- Support pickup / table / seat / named locations
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional

import rclpy
import yaml

from geometry_msgs.msg import PoseStamped

from nav2_simple_commander.robot_navigator import (
    BasicNavigator,
    TaskResult,
)

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from soomac_interfaces.srv import MoveScout


# ================================================================
# Scout Control Node
# ================================================================

class ScoutControlNode(Node):

    def __init__(
        self,
        navigator: BasicNavigator,
    ) -> None:

        super().__init__('scout_control_node')

        self.navigator = navigator
        self.callback_group = ReentrantCallbackGroup()

        # --------------------------------------------------------
        # Parameters
        # --------------------------------------------------------

        self.declare_parameter(
            'venue_yaml',
            '~/.ros/venue_tables.yaml',
        )

        self.declare_parameter(
            'move_service',
            '/scout/move',
        )

        self.declare_parameter(
            'cancel_service',
            '/scout/cancel',
        )

        self.declare_parameter(
            'move_enabled_topic',
            '/scout/move_enabled',
        )

        self.declare_parameter(
            'emergency_stop_topic',
            '/scout/emergency_stop',
        )

        self.declare_parameter(
            'navigation_timeout_sec',
            180.0,
        )

        # 0.0:
        # move_enabled=True가 될 때까지 계속 대기
        self.declare_parameter(
            'wait_for_move_enabled_timeout_sec',
            0.0,
        )

        # Scout가 실제 pickup 위치에 놓였을 때만 True 사용
        self.declare_parameter(
            'set_initial_pose_from_pickup',
            False,
        )

        # --------------------------------------------------------
        # Lift
        #
        # Scout가 작업 위치에 도착하면
        # Lift 최대 높이 상승을 요청한다.
        # --------------------------------------------------------

        self.declare_parameter(
            'lift_up_service',
            '/lift/up',
        )

        self.declare_parameter(
            'lift_service_timeout_sec',
            60.0,
        )

        # --------------------------------------------------------
        # State
        # --------------------------------------------------------

        self.move_enabled = False
        self.move_state_received = False

        self.emergency_stop = False
        self.cancel_requested = False

        # 동시에 두 개의 이동 서비스가 Nav2를 사용하는 것 방지
        self.navigation_lock = threading.Lock()

        # --------------------------------------------------------
        # Lift service client
        # --------------------------------------------------------

        self.lift_up_client = self.create_client(
            Trigger,
            str(
                self.get_parameter(
                    'lift_up_service'
                ).value
            ),
            callback_group=self.callback_group,
        )

        # --------------------------------------------------------
        # Subscribers
        # --------------------------------------------------------

        self.create_subscription(
            Bool,
            str(
                self.get_parameter(
                    'move_enabled_topic'
                ).value
            ),
            self.move_enabled_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            Bool,
            str(
                self.get_parameter(
                    'emergency_stop_topic'
                ).value
            ),
            self.emergency_stop_callback,
            10,
            callback_group=self.callback_group,
        )

        # --------------------------------------------------------
        # Services
        # --------------------------------------------------------

        self.create_service(
            MoveScout,
            str(
                self.get_parameter(
                    'move_service'
                ).value
            ),
            self.move_callback,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            str(
                self.get_parameter(
                    'cancel_service'
                ).value
            ),
            self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            'Scout control ready: '
            '/scout/move -> venue pose -> Nav2'
        )

    # ============================================================
    # Move permission callback
    # ============================================================

    def move_enabled_callback(
        self,
        msg: Bool,
    ) -> None:

        previous = self.move_enabled

        self.move_enabled = bool(msg.data)
        self.move_state_received = True

        if previous != self.move_enabled:

            if self.move_enabled:

                self.get_logger().info(
                    'Scout movement ENABLED'
                )

            else:

                self.get_logger().warning(
                    'Scout movement DISABLED'
                )

    # ============================================================
    # Emergency stop callback
    # ============================================================

    def emergency_stop_callback(
        self,
        msg: Bool,
    ) -> None:

        self.emergency_stop = bool(msg.data)

        if not self.emergency_stop:
            return

        self.get_logger().error(
            'Emergency stop received.'
        )

        self.cancel_requested = True

        try:
            self.navigator.cancelTask()

        except Exception:
            pass

    # ============================================================
    # Cancel service
    # ============================================================

    def cancel_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:

        self.cancel_requested = True

        try:
            self.navigator.cancelTask()

        except Exception:
            pass

        response.success = True
        response.message = (
            'Scout navigation cancel requested.'
        )

        return response

    # ============================================================
    # Move service
    # ============================================================

    def call_lift_up(
        self,
    ) -> tuple[bool, str]:

        service_name = str(
            self.get_parameter(
                'lift_up_service'
            ).value
        )

        timeout_sec = float(
            self.get_parameter(
                'lift_service_timeout_sec'
            ).value
        )

        self.get_logger().info(
            f'Lift UP 요청: {service_name}'
        )

        # --------------------------------------------------------
        # Lift Node가 실행 중인지 확인
        # --------------------------------------------------------

        if not self.lift_up_client.wait_for_service(
            timeout_sec=timeout_sec
        ):

            return (
                False,
                f'{service_name} 서비스를 찾을 수 없습니다.'
            )

        # --------------------------------------------------------
        # 최대 높이 상승 요청
        # --------------------------------------------------------

        future = self.lift_up_client.call_async(
            Trigger.Request()
        )

        start = time.monotonic()

        # Lift Node가 최대 높이에 도달하고
        # Trigger 응답을 보낼 때까지 기다린다.
        while rclpy.ok() and not future.done():

            if (
                self.emergency_stop
                or self.cancel_requested
            ):

                return (
                    False,
                    'Lift UP 대기 중 취소/비상정지'
                )

            if (
                timeout_sec > 0.0
                and (
                    time.monotonic() - start
                    > timeout_sec
                )
            ):

                return (
                    False,
                    f'{service_name} 응답 시간 초과'
                )

            time.sleep(0.02)

        if not future.done():

            return (
                False,
                'Lift UP 응답 실패'
            )

        try:

            result = future.result()

        except Exception as error:  # noqa: BLE001

            return (
                False,
                f'Lift UP 호출 오류: {error}'
            )

        if result is None:

            return (
                False,
                'Lift UP 응답 없음'
            )

        if not result.success:

            return (
                False,
                result.message
                or 'Lift UP 실패'
            )

        self.get_logger().info(
            f'Lift UP 완료: {result.message}'
        )

        return (
            True,
            result.message,
        )


    def move_callback(
        self,
        request: MoveScout.Request,
        response: MoveScout.Response,
    ) -> MoveScout.Response:

        target = str(
            request.target
        ).strip()

        if not target:

            response.success = False
            response.message = (
                'target이 비어 있습니다.'
            )

            return response

        # --------------------------------------------------------
        # 이미 다른 이동 요청 실행 중인지 확인
        # --------------------------------------------------------

        if not self.navigation_lock.acquire(
            blocking=False
        ):

            response.success = False

            response.message = (
                'Scout가 이미 다른 이동 요청을 '
                '수행 중입니다.'
            )

            return response

        try:

            self.cancel_requested = False

            # ----------------------------------------------------
            # Emergency stop
            # ----------------------------------------------------

            if self.emergency_stop:

                response.success = False

                response.message = (
                    'emergency_stop=True 상태라 '
                    '이동할 수 없습니다.'
                )

                return response

            # ----------------------------------------------------
            # venue_tables.yaml은 여기서 읽는다.
            #
            # 즉 navigation launch 시 파일이 없어도
            # Scout Control Node 자체는 실행 가능하다.
            # ----------------------------------------------------

            try:

                venue_path = str(
                    self.get_parameter(
                        'venue_yaml'
                    ).value
                )

                venue = load_venue(
                    venue_path
                )

                label, goal = target_to_pose(
                    self.navigator,
                    venue,
                    target,
                )

            except Exception as error:

                response.success = False

                response.message = (
                    f'target 변환 실패: {error}'
                )

                return response

            self.get_logger().info(
                f'Move request: {target} '
                f'-> {label} '
                f'('
                f'{goal.pose.position.x:.3f}, '
                f'{goal.pose.position.y:.3f}'
                f')'
            )

            # ----------------------------------------------------
            # Navigation
            # ----------------------------------------------------

            success, message = (
                self.navigate_with_interlock(
                    goal,
                    label,
                )
            )

            response.success = success

            # ------------------------------------------------
            # 작업 위치 도착 성공 시 Lift 최대 높이 상승
            #
            # target_to_pose()에서 작업 위치는
            # table_* label로 변환된다.
            # pickup/home은 pickup_zone이므로 Lift를 안 올린다.
            # ------------------------------------------------

            if (
                success
                and label.startswith('table_')
            ):

                lift_ok, lift_message = (
                    self.call_lift_up()
                )

                if not lift_ok:

                    response.success = False

                    response.message = (
                        f'Scout 이동 성공, '
                        f'Lift UP 실패: '
                        f'{lift_message}'
                    )

                    return response

                message = (
                    f'{message}; '
                    'Lift 최대 높이 상승 완료'
                )

            response.message = message
            return response

        finally:

            self.navigation_lock.release()

    # ============================================================
    # Abort condition
    # ============================================================

    def should_abort(self) -> bool:

        return (
            self.emergency_stop
            or self.cancel_requested
            or not rclpy.ok()
        )

    # ============================================================
    # Wait for movement permission
    # ============================================================

    def wait_until_move_enabled(
        self,
        context: str,
    ) -> tuple[bool, str]:

        timeout_sec = float(
            self.get_parameter(
                'wait_for_move_enabled_timeout_sec'
            ).value
        )

        start_time = time.monotonic()
        last_log_time = 0.0

        while (
            rclpy.ok()
            and not self.move_enabled
        ):

            if self.should_abort():

                return (
                    False,
                    f'{context}: 이동 취소/비상정지'
                )

            if timeout_sec > 0.0:

                elapsed = (
                    time.monotonic()
                    - start_time
                )

                if elapsed > timeout_sec:

                    return (
                        False,
                        f'{context}: '
                        'move_enabled 대기 시간 초과'
                    )

            now = time.monotonic()

            if now - last_log_time >= 2.0:

                if not self.move_state_received:

                    reason = (
                        '/scout/move_enabled '
                        '신호 미수신'
                    )

                else:

                    reason = (
                        'move_enabled=False'
                    )

                self.get_logger().info(
                    f'Waiting: {context} - {reason}'
                )

                last_log_time = now

            # MultiThreadedExecutor가
            # move_enabled callback을 다른 thread에서 처리
            time.sleep(0.05)

        if self.should_abort():

            return (
                False,
                f'{context}: 이동 취소/비상정지'
            )

        return True, ''

    # ============================================================
    # Cancel Nav2 task
    # ============================================================

    def cancel_nav_and_wait(
        self,
    ) -> None:

        self.navigator.cancelTask()

        deadline = (
            time.monotonic() + 3.0
        )

        while (
            rclpy.ok()
            and not self.navigator.isTaskComplete()
        ):

            if time.monotonic() >= deadline:
                break

            time.sleep(0.02)

    # ============================================================
    # Navigation + movement interlock
    # ============================================================

    def navigate_with_interlock(
        self,
        goal: PoseStamped,
        label: str,
    ) -> tuple[bool, str]:

        navigation_timeout = float(
            self.get_parameter(
                'navigation_timeout_sec'
            ).value
        )

        overall_start = time.monotonic()

        while rclpy.ok():

            # ----------------------------------------------------
            # 이동 전 리프트 하강 / 이동 허가 확인
            # ----------------------------------------------------

            enabled, reason = (
                self.wait_until_move_enabled(
                    label
                )
            )

            if not enabled:

                return False, reason

            # ----------------------------------------------------
            # 전체 Navigation timeout 확인
            # ----------------------------------------------------

            if navigation_timeout > 0.0:

                elapsed = (
                    time.monotonic()
                    - overall_start
                )

                if elapsed > navigation_timeout:

                    return (
                        False,
                        f'{label}: 이동 시간 초과'
                    )

            # ----------------------------------------------------
            # Nav2 Goal
            # ----------------------------------------------------

            goal.header.stamp = (
                self.navigator
                .get_clock()
                .now()
                .to_msg()
            )

            self.get_logger().info(
                f'NAVIGATING: {label}'
            )

            self.navigator.goToPose(
                goal
            )

            paused_by_interlock = False

            # ----------------------------------------------------
            # Goal 실행 중
            # ----------------------------------------------------

            while (
                rclpy.ok()
                and not self.navigator.isTaskComplete()
            ):

                # ------------------------------------------------
                # emergency / cancel
                # ------------------------------------------------

                if self.should_abort():

                    self.cancel_nav_and_wait()

                    return (
                        False,
                        f'{label}: 이동 취소/비상정지'
                    )

                # ------------------------------------------------
                # 리프트가 올라가거나 이동 허가 해제
                # ------------------------------------------------

                if not self.move_enabled:

                    self.cancel_nav_and_wait()

                    self.get_logger().warning(
                        f'PAUSED: {label} - '
                        'move_enabled=False'
                    )

                    paused_by_interlock = True
                    break

                # ------------------------------------------------
                # timeout
                # ------------------------------------------------

                if (
                    navigation_timeout > 0.0
                    and
                    time.monotonic()
                    - overall_start
                    > navigation_timeout
                ):

                    self.cancel_nav_and_wait()

                    return (
                        False,
                        f'{label}: 이동 시간 초과'
                    )

                time.sleep(0.05)

            # ----------------------------------------------------
            # 이동 허가 해제로 취소된 경우
            #
            # 다시 move_enabled=True가 되면
            # 같은 목적지 재시도
            # ----------------------------------------------------

            if paused_by_interlock:
                continue

            # ----------------------------------------------------
            # Nav2 result
            # ----------------------------------------------------

            result = (
                self.navigator.getResult()
            )

            if result == TaskResult.SUCCEEDED:

                self.get_logger().info(
                    f'ARRIVED: {label}'
                )

                return (
                    True,
                    f'{label} 이동 성공'
                )

            if result == TaskResult.CANCELED:

                return (
                    False,
                    f'{label} 이동 취소'
                )

            return (
                False,
                f'{label} 이동 실패'
            )

        return (
            False,
            f'{label}: ROS 종료'
        )


# ================================================================
# Venue YAML loader
# ================================================================

def load_venue(
    path_text: str,
) -> dict[str, Any]:

    path = Path(
        path_text
    ).expanduser()

    if not path.is_file():

        raise FileNotFoundError(
            '행사장 좌표 파일이 없습니다: '
            f'{path}'
        )

    with path.open(
        'r',
        encoding='utf-8',
    ) as stream:

        venue = yaml.safe_load(
            stream
        )

    if not isinstance(
        venue,
        dict,
    ):

        raise ValueError(
            'venue YAML 최상위 형식이 '
            'dict가 아닙니다.'
        )

    if not isinstance(
        venue.get('pickup_zone'),
        dict,
    ):

        raise ValueError(
            'venue YAML에 '
            'pickup_zone이 없습니다.'
        )

    if not isinstance(
        venue.get('tables'),
        list,
    ):

        raise ValueError(
            'venue YAML에 '
            'tables가 없습니다.'
        )

    return venue


# ================================================================
# Dict -> PoseStamped
# ================================================================

def pose_from_dict(
    navigator: BasicNavigator,
    frame_id: str,
    pose_data: dict[str, Any],
) -> PoseStamped:

    pose = PoseStamped()

    pose.header.frame_id = frame_id

    pose.header.stamp = (
        navigator
        .get_clock()
        .now()
        .to_msg()
    )

    pose.pose.position.x = float(
        pose_data['x']
    )

    pose.pose.position.y = float(
        pose_data['y']
    )

    yaw = float(
        pose_data['yaw']
    )

    pose.pose.orientation.z = (
        math.sin(
            yaw / 2.0
        )
    )

    pose.pose.orientation.w = (
        math.cos(
            yaw / 2.0
        )
    )

    return pose


# ================================================================
# Find table
# ================================================================

def table_by_id(
    venue: dict[str, Any],
    table_id: int,
) -> dict[str, Any]:

    for table in venue['tables']:

        if int(
            table['id']
        ) == table_id:

            return table

    raise KeyError(
        f'{table_id}번 테이블이 없습니다.'
    )


# ================================================================
# Seat -> Table
# ================================================================

def table_id_from_seat(
    venue: dict[str, Any],
    seat_id: int,
) -> int:

    mapping = venue.get(
        'seat_to_table'
    )

    if mapping is None:

        raise KeyError(
            'venue YAML에 seat_to_table이 없습니다.'
        )

    if not isinstance(
        mapping,
        dict,
    ):

        raise ValueError(
            'seat_to_table 형식은 '
            'dict여야 합니다.'
        )

    possible_keys = [
        str(seat_id),
        f'seat_{seat_id}',
    ]

    value = None

    for key in possible_keys:

        if key in mapping:

            value = mapping[key]
            break

    # YAML에서 숫자 key로 저장한 경우
    if (
        value is None
        and seat_id in mapping
    ):

        value = mapping[seat_id]

    if value is None:

        raise KeyError(
            f'{seat_id}번 좌석의 '
            '테이블 정보가 없습니다.'
        )

    if isinstance(
        value,
        str,
    ):

        match = re.fullmatch(
            r'table_?(\d+)',
            value.strip().lower(),
        )

        if match is not None:

            return int(
                match.group(1)
            )

    return int(value)


# ================================================================
# Named location
#
# home
# stage_front
# patrol_1
# patrol_2
# ...
# ================================================================

def named_location_by_name(
    venue: dict[str, Any],
    name: str,
) -> Optional[dict[str, Any]]:

    locations = venue.get(
        'named_locations',
        {},
    )

    if not isinstance(
        locations,
        dict,
    ):

        return None

    normalized_name = (
        name
        .strip()
        .lower()
        .replace('-', '_')
        .replace(' ', '_')
    )

    for key, value in locations.items():

        normalized_key = (
            str(key)
            .strip()
            .lower()
            .replace('-', '_')
            .replace(' ', '_')
        )

        if normalized_key == normalized_name:

            if not isinstance(
                value,
                dict,
            ):

                raise ValueError(
                    f'{key} 좌표 형식이 '
                    '올바르지 않습니다.'
                )

            return value

    return None


# ================================================================
# Target -> Nav2 Pose
# ================================================================

def target_to_pose(
    navigator: BasicNavigator,
    venue: dict[str, Any],
    target: str,
) -> tuple[str, PoseStamped]:

    frame_id = str(
        venue.get(
            'frame_id',
            'map',
        )
    )

    normalized = (
        target
        .strip()
        .lower()
        .replace('-', '_')
        .replace(' ', '_')
    )

    # ------------------------------------------------------------
    # Pickup
    # ------------------------------------------------------------

    if normalized in {
        'pickup',
        'pickup_zone',
        'start',
    }:

        return (
            'pickup_zone',
            pose_from_dict(
                navigator,
                frame_id,
                venue['pickup_zone'],
            ),
        )

    # ------------------------------------------------------------
    # Named locations
    #
    # home
    # stage_front
    # patrol_1
    # ...
    # ------------------------------------------------------------

    named_location = (
        named_location_by_name(
            venue,
            normalized,
        )
    )

    if named_location is not None:

        return (
            normalized,
            pose_from_dict(
                navigator,
                frame_id,
                named_location,
            ),
        )

    # ------------------------------------------------------------
    # Table
    #
    # table_1
    # table1
    # ------------------------------------------------------------

    table_match = re.fullmatch(
        r'table_?(\d+)',
        normalized,
    )

    if table_match is not None:

        table_id = int(
            table_match.group(1)
        )

        table = table_by_id(
            venue,
            table_id,
        )

        approach = table.get(
            'approach'
        )

        if not isinstance(
            approach,
            dict,
        ):

            raise ValueError(
                f'{table_id}번 테이블에 '
                'approach 좌표가 없습니다.'
            )

        return (
            f'table_{table_id}',
            pose_from_dict(
                navigator,
                frame_id,
                approach,
            ),
        )

    # ------------------------------------------------------------
    # 숫자만 보내도 table 번호로 처리
    #
    # "1" -> table_1
    # ------------------------------------------------------------

    if normalized.isdigit():

        table_id = int(normalized)

        table = table_by_id(
            venue,
            table_id,
        )

        approach = table.get(
            'approach'
        )

        if not isinstance(
            approach,
            dict,
        ):

            raise ValueError(
                f'{table_id}번 테이블에 '
                'approach 좌표가 없습니다.'
            )

        return (
            f'table_{table_id}',
            pose_from_dict(
                navigator,
                frame_id,
                approach,
            ),
        )

    # ------------------------------------------------------------
    # Seat
    #
    # seat_1
    # seat1
    # ------------------------------------------------------------

    seat_match = re.fullmatch(
        r'seat_?(\d+)',
        normalized,
    )

    if seat_match is not None:

        seat_id = int(
            seat_match.group(1)
        )

        table_id = table_id_from_seat(
            venue,
            seat_id,
        )

        table = table_by_id(
            venue,
            table_id,
        )

        approach = table.get(
            'approach'
        )

        if not isinstance(
            approach,
            dict,
        ):

            raise ValueError(
                f'{table_id}번 테이블에 '
                'approach 좌표가 없습니다.'
            )

        return (
            f'seat_{seat_id} -> '
            f'table_{table_id}',
            pose_from_dict(
                navigator,
                frame_id,
                approach,
            ),
        )

    raise ValueError(
        '지원하지 않는 target입니다: '
        f'{target}. '
        '예: pickup, table_1, seat_1, '
        'home, stage_front, patrol_1'
    )


# ================================================================
# Main
# ================================================================

def main(
    args: Optional[list[str]] = None,
) -> None:

    rclpy.init(args=args)

    navigator: Optional[
        BasicNavigator
    ] = None

    control: Optional[
        ScoutControlNode
    ] = None

    executor: Optional[
        MultiThreadedExecutor
    ] = None

    try:

        navigator = BasicNavigator()

        control = ScoutControlNode(
            navigator
        )

        # --------------------------------------------------------
        # venue_tables.yaml
        #
        # 중요:
        # 파일이 없어도 Scout Control / Nav2는 실행 가능.
        #
        # 단,
        # set_initial_pose_from_pickup=True라면
        # pickup 좌표가 필요하므로 시작할 때 YAML을 읽음.
        # --------------------------------------------------------

        venue_path = str(
            control.get_parameter(
                'venue_yaml'
            ).value
        )

        if bool(
            control.get_parameter(
                'set_initial_pose_from_pickup'
            ).value
        ):

            venue = load_venue(
                venue_path
            )

            frame_id = str(
                venue.get(
                    'frame_id',
                    'map',
                )
            )

            pickup_pose = pose_from_dict(
                navigator,
                frame_id,
                venue['pickup_zone'],
            )

            navigator.setInitialPose(
                pickup_pose
            )

            control.get_logger().warning(
                'Initial pose = pickup_zone. '
                'Scout가 실제 pickup_zone에 '
                '있을 때만 사용해야 합니다.'
            )

        else:

            venue_file = Path(
                venue_path
            ).expanduser()

            if not venue_file.is_file():

                control.get_logger().warning(
                    'venue_tables.yaml이 아직 없습니다: '
                    f'{venue_file}. '
                    'Nav2와 Scout Control은 계속 실행합니다. '
                    '/scout/move 사용 전 table_mapper로 '
                    'venue_tables.yaml을 생성해야 합니다.'
                )

        # --------------------------------------------------------
        # Nav2 lifecycle
        #
        # navigation.launch.py의 lifecycle_manager가
        # Nav2 활성화를 담당한다.
        #
        # 여기서 waitUntilNav2Active()를 다시 호출하지 않는다.
        # --------------------------------------------------------

        control.get_logger().info(
            'Scout Control ready. '
            'Nav2 lifecycle is managed externally.'
        )

        # --------------------------------------------------------
        # Multi-thread executor
        #
        # /scout/move service가 Nav2 결과를 기다리는 중에도
        # move_enabled와 emergency_stop callback은
        # 계속 처리되어야 함.
        # --------------------------------------------------------

        executor = MultiThreadedExecutor(
            num_threads=3
        )

        executor.add_node(
            control
        )

        executor.spin()

    except KeyboardInterrupt:

        pass

    finally:

        if navigator is not None:

            try:
                navigator.cancelTask()

            except Exception:
                pass

            navigator.destroyNode()

        if (
            executor is not None
            and control is not None
        ):

            try:
                executor.remove_node(
                    control
                )

            except Exception:
                pass

        if control is not None:

            control.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':
    main()
