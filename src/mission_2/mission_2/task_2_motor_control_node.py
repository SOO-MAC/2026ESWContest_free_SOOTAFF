#!/usr/bin/env python3
"""Mission 2 motor control: Dynamixel trajectory and gripper execution."""
from __future__ import annotations
import threading
import time
from typing import Iterable

import numpy as np
import rclpy
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray
from std_srvs.srv import Trigger
from soomac_interfaces.srv import DeliverMike, ExecutePickPlace, MikeGripper, MoveToTicks
from control_config.robot_config import *
from control_config.task_2_config import (
    WAYPOINT_TRACK_STALL_TIMEOUT_SEC,
    WAYPOINT_TRACK_SAMPLE_DT_SEC,
    WAYPOINT_TRACK_PROGRESS_EPS_RATIO,
    COLLISION_RECOVERY_WAIT_SEC,
    COLLISION_RECOVERY_PROFILE_VELOCITY,
    COLLISION_RECOVERY_MAX_TICK_STEP,
    COLLISION_RECOVERY_COMMAND_DT_SEC,
    COLLISION_RECOVERY_TIMEOUT_SEC,
)


class CollisionDetected(RuntimeError):
    """Task 1 collision v5와 동일한 waypoint tracking stall 예외."""

    def __init__(
        self,
        phase_name: str,
        goal_ticks: np.ndarray,
        present_ticks: np.ndarray,
        error_ticks: np.ndarray,
        thresholds: np.ndarray,
        stalled_sec: float,
    ) -> None:
        self.phase_name = str(phase_name)
        self.goal_ticks = np.asarray(goal_ticks, dtype=np.int64).reshape(5)
        self.present_ticks = np.asarray(present_ticks, dtype=np.int64).reshape(5)
        self.error_ticks = np.asarray(error_ticks, dtype=np.int64).reshape(5)
        self.thresholds = np.asarray(thresholds, dtype=np.int64).reshape(5)
        self.stalled_sec = float(stalled_sec)

        super().__init__(
            f"충돌 감지 | phase={self.phase_name} | "
            f"{self.stalled_sec:.2f}s 동안 waypoint 추종 진전 없음 | "
            f"goal={self.goal_ticks.tolist()} | "
            f"present={self.present_ticks.tolist()} | "
            f"error={self.error_ticks.tolist()} | "
            f"limit={self.thresholds.tolist()}"
        )


def signed_int32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value

def check_write(packet: PacketHandler, comm_result: int, dxl_error: int, label: str) -> None:
    if comm_result != COMM_SUCCESS:
        raise RuntimeError(f"{label} 통신 실패: {packet.getTxRxResult(comm_result)}")
    if dxl_error != 0:
        raise RuntimeError(f"{label} 패킷 오류: {packet.getRxPacketError(dxl_error)}")

def write1(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int, value: int, label: str) -> None:
    comm, error = packet.write1ByteTxRx(port, dxl_id, address, int(value))
    check_write(packet, comm, error, label)

def write4(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int, value: int, label: str) -> None:
    comm, error = packet.write4ByteTxRx(port, dxl_id, address, int(value) & 0xFFFFFFFF)
    check_write(packet, comm, error, label)

def read4_retry(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int, label: str) -> int:
    last_comm = COMM_SUCCESS
    last_error = 0
    for attempt in range(1, ARM_READ_RETRIES + 1):
        value, comm, error = packet.read4ByteTxRx(port, dxl_id, address)
        last_comm, last_error = comm, error
        if comm == COMM_SUCCESS and error == 0:
            return int(value)
        if attempt < ARM_READ_RETRIES:
            time.sleep(ARM_READ_RETRY_DT_SEC)

    comm_text = packet.getTxRxResult(last_comm) if last_comm != COMM_SUCCESS else "COMM_SUCCESS"
    error_text = packet.getRxPacketError(last_error) if last_error else "None"
    raise RuntimeError(f"{label} 최종 실패: comm={comm_text}, packet={error_text}")

def set_torque(port: PortHandler, packet: PacketHandler, ids: Iterable[int], enabled: bool) -> None:
    value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
    failures = []
    for dxl_id in ids:
        try:
            write1(
                port,
                packet,
                int(dxl_id),
                ADDR_TORQUE_ENABLE,
                value,
                f"ID {dxl_id} Torque {'ON' if enabled else 'OFF'}",
            )
        except Exception as exc:  # best-effort shutdown must continue
            failures.append(str(exc))
    if failures:
        raise RuntimeError(" | ".join(failures))

def read_arm_ticks(port: PortHandler, packet: PacketHandler) -> np.ndarray:
    return np.array(
        [
            signed_int32(
                read4_retry(
                    port,
                    packet,
                    dxl_id,
                    ADDR_PRESENT_POSITION,
                    f"ID {dxl_id} Present Position",
                )
            )
            for dxl_id in ARM_IDS
        ],
        dtype=np.int64,
    )

def parse_tick_path(
    joint_ticks,
    point_count: int,
) -> list[np.ndarray]:
    point_count = int(point_count)
    flat_ticks = np.asarray(joint_ticks, dtype=np.int64)
    if point_count <= 0 or flat_ticks.size != point_count * 5:
        raise ValueError(
            "invalid joint_ticks: "
            f"point_count={point_count}, size={flat_ticks.size}"
        )
    return [row.copy() for row in flat_ticks.reshape(point_count, 5)]


def quintic_smoothstep(u: float) -> float:
    """Quintic time scaling with zero velocity/acceleration at both ends."""
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def build_phase_commands(
    ticks: list[np.ndarray],
    held_j5_tick: int | None = None,
    min_commands: int = PHASE_MIN_COMMANDS,
    max_commands: int = PHASE_MAX_COMMANDS,
) -> list[np.ndarray]:
    """Planner path에 quintic을 적용하고 필요 시 J5를 고정한다."""
    if not ticks:
        return []

    path = np.asarray(ticks, dtype=np.int64).reshape(-1, 5).copy()

    if held_j5_tick is not None:
        path[:, 4] = int(held_j5_tick)

    if len(path) == 1:
        return [path[0].copy()]

    cumulative = np.zeros(len(path), dtype=float)
    cumulative[1:] = np.cumsum(
        np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1)
    )
    total_length = float(cumulative[-1])

    if total_length <= 1.0e-12:
        return [path[-1].copy()]

    # 각 관절이 phase 전체에서 실제로 이동한 누적 tick 양.
    joint_travel = np.sum(np.abs(np.diff(path, axis=0)), axis=0)
    required = int(np.max(np.ceil(joint_travel / PHASE_MAX_TICK_STEP)))

    # planner waypoint 개수보다 적게 만들지 않는다.
    count = max(int(min_commands), required, len(path) - 1)
    count = min(count, int(max_commands))

    commands: list[np.ndarray] = []

    for index in range(1, count + 1):
        u = index / count
        s = quintic_smoothstep(u)
        distance = s * total_length

        upper = min(
            int(np.searchsorted(cumulative, distance, side="right")), len(path) - 1
        )
        lower = max(upper - 1, 0)
        segment_length = cumulative[upper] - cumulative[lower]

        if distance <= 0.0:
            sampled = path[0]
        elif distance >= total_length:
            sampled = path[-1]
        elif segment_length <= 1.0e-12:
            sampled = path[upper]
        else:
            alpha = (distance - cumulative[lower]) / segment_length
            sampled = (1.0 - alpha) * path[lower] + alpha * path[upper]

        command = np.rint(sampled).astype(np.int64)

        if held_j5_tick is not None:
            command[4] = int(held_j5_tick)

        commands.append(command)

    commands[-1] = path[-1].copy()
    return commands


def stream_commands(
    arm_writer: ArmSyncWriter,
    commands: list[np.ndarray],
) -> None:
    """고정 주기로 command sequence를 streaming한다."""
    next_send_time = time.monotonic()

    for command in commands:
        arm_writer.send(command)

        next_send_time += ARM_STREAM_DT_SEC
        remaining = next_send_time - time.monotonic()

        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_send_time = time.monotonic()


def rotate_j5_only(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    current_ticks: np.ndarray,
    target_j5_tick: int,
    phase_name: str = "J5_ROTATE",
) -> np.ndarray:
    """PLACE_ABOVE 도착 후 J1~J4를 유지하고 J5만 회전한다."""
    start = np.asarray(current_ticks, dtype=np.int64).reshape(5).copy()

    delta = int(target_j5_tick) - int(start[4])

    if delta == 0:
        return start

    next_send_time = time.monotonic()

    for index in range(1, TRANSPORT_YAW_COMMANDS + 1):
        u = index / TRANSPORT_YAW_COMMANDS
        blend = quintic_smoothstep(u)

        command = start.copy()
        command[4] = int(round(start[4] + blend * delta))

        arm_writer.send(command)

        next_send_time += ARM_STREAM_DT_SEC
        remaining = next_send_time - time.monotonic()

        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_send_time = time.monotonic()

    goal = start.copy()
    goal[4] = int(target_j5_tick)

    yaw_thresholds = np.array(
        [1000000, 1000000, 1000000, 1000000, ARM_ALIGN_THRESHOLDS[4]],
        dtype=np.int64,
    )

    return wait_waypoint_tracking(
        port,
        packet,
        arm_writer,
        goal,
        phase_name=phase_name,
        timeout_sec=ARM_WAYPOINT_TIMEOUT_SEC,
        thresholds=yaw_thresholds,
    )


class ArmSyncWriter:
    """Reuse one GroupSyncWrite object for all J1~J5 commands."""

    def __init__(self, port: PortHandler, packet: PacketHandler) -> None:
        self.packet = packet
        self.group = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)

    def send(self, ticks: np.ndarray) -> None:
        ticks = np.asarray(ticks, dtype=np.int64).reshape(5)
        try:
            for dxl_id, tick in zip(ARM_IDS, ticks):
                unsigned = int(tick) & 0xFFFFFFFF
                data = [
                    unsigned & 0xFF,
                    (unsigned >> 8) & 0xFF,
                    (unsigned >> 16) & 0xFF,
                    (unsigned >> 24) & 0xFF,
                ]
                if not self.group.addParam(
                    int(dxl_id),
                    data,
                ):
                    raise RuntimeError(
                        f"GroupSyncWrite addParam 실패: "
                        f"ID={dxl_id}, tick={int(tick)}"
                    )

            comm_result = self.group.txPacket()
            if comm_result != COMM_SUCCESS:
                raise RuntimeError(
                    "GroupSyncWrite Goal Position 통신 실패: "
                    f"{self.packet.getTxRxResult(comm_result)}"
                )
        finally:
            self.group.clearParam()


def set_arm_profile_velocity(
    port: PortHandler,
    packet: PacketHandler,
    velocity: int,
) -> None:
    base = max(1, int(velocity))

    for index, dxl_id in enumerate(ARM_IDS):
        applied = base

        if index == 1:  # J2
            applied = max(
                1,
                int(round(base * J2_PROFILE_VELOCITY_SCALE)),
            )
        elif index == 2:  # J3
            applied = max(
                1,
                int(round(base * J3_PROFILE_VELOCITY_SCALE)),
            )

        write4(
            port,
            packet,
            dxl_id,
            ADDR_PROFILE_VELOCITY,
            applied,
            f"ID {dxl_id} Profile Velocity={applied}",
        )


def wait_arm_reached(
    port: PortHandler,
    packet: PacketHandler,
    goal_ticks: np.ndarray,
    timeout_sec: float = ARM_WAYPOINT_TIMEOUT_SEC,
    thresholds: np.ndarray = ARM_POSITION_THRESHOLDS,
) -> np.ndarray:
    goal_ticks = np.asarray(goal_ticks, dtype=np.int64).reshape(5)
    thresholds = np.asarray(thresholds, dtype=np.int64).reshape(5)
    deadline = time.monotonic() + float(timeout_sec)
    last_ticks = read_arm_ticks(port, packet)

    while time.monotonic() < deadline:
        last_ticks = read_arm_ticks(port, packet)
        errors = goal_ticks - last_ticks
        if np.all(np.abs(errors) <= thresholds):
            return last_ticks
        time.sleep(0.05)

    errors = goal_ticks - last_ticks
    raise RuntimeError(
        "Arm waypoint 도달 시간 초과 | "
        f"goal={goal_ticks.tolist()} | present={last_ticks.tolist()} | "
        f"error={errors.tolist()} | limit={thresholds.tolist()}"
    )


def wait_waypoint_tracking(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    goal_ticks: np.ndarray,
    phase_name: str,
    thresholds: np.ndarray = ARM_POSITION_THRESHOLDS,
    timeout_sec: float = ARM_WAYPOINT_TIMEOUT_SEC,
) -> np.ndarray:
    """Task 1 collision v5와 동일한 phase 최종 waypoint 추종 감시."""
    goal_ticks = np.asarray(goal_ticks, dtype=np.int64).reshape(5)
    thresholds = np.asarray(thresholds, dtype=np.int64).reshape(5)

    start_time = time.monotonic()
    deadline = start_time + float(timeout_sec)
    last_ticks = read_arm_ticks(port, packet)
    last_errors = goal_ticks - last_ticks

    if np.all(np.abs(last_errors) <= thresholds):
        return last_ticks

    safe_thresholds = np.maximum(thresholds.astype(float), 1.0)
    best_error_ratio = float(np.max(np.abs(last_errors) / safe_thresholds))
    last_progress_time = start_time

    while time.monotonic() < deadline:
        time.sleep(float(WAYPOINT_TRACK_SAMPLE_DT_SEC))

        now = time.monotonic()
        last_ticks = read_arm_ticks(port, packet)
        last_errors = goal_ticks - last_ticks

        if np.all(np.abs(last_errors) <= thresholds):
            return last_ticks

        current_error_ratio = float(np.max(np.abs(last_errors) / safe_thresholds))

        if (
            current_error_ratio
            <= best_error_ratio - float(WAYPOINT_TRACK_PROGRESS_EPS_RATIO)
        ):
            best_error_ratio = current_error_ratio
            last_progress_time = now

        stalled_sec = now - last_progress_time
        if stalled_sec >= float(WAYPOINT_TRACK_STALL_TIMEOUT_SEC):
            # Task 1과 동일: Torque OFF 대신 현재 위치를 Goal로 보내 HOLD.
            arm_writer.send(last_ticks)
            raise CollisionDetected(
                phase_name=phase_name,
                goal_ticks=goal_ticks,
                present_ticks=last_ticks,
                error_ticks=last_errors,
                thresholds=thresholds,
                stalled_sec=stalled_sec,
            )

    arm_writer.send(last_ticks)
    raise CollisionDetected(
        phase_name=phase_name,
        goal_ticks=goal_ticks,
        present_ticks=last_ticks,
        error_ticks=last_errors,
        thresholds=thresholds,
        stalled_sec=float(timeout_sec),
    )


def move_arm_direct(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    current_ticks: np.ndarray,
    goal_ticks: np.ndarray,
    *,
    profile_velocity: int,
    timeout_sec: float,
    min_commands: int = 1,
    skip_if_reached: bool = False,
    reset_schedule_when_late: bool = False,
) -> np.ndarray:
    """현재 tick에서 목표 tick까지 quintic direct move를 수행한다."""
    current = np.asarray(current_ticks, dtype=np.int64).reshape(5)
    goal = np.asarray(goal_ticks, dtype=np.int64).reshape(5)
    delta = goal - current
    max_delta = int(np.max(np.abs(delta)))

    if skip_if_reached and max_delta == 0:
        return current.copy()

    set_arm_profile_velocity(port, packet, profile_velocity)
    command_count = max(
        int(min_commands), int(np.ceil(max_delta / DIRECT_MAX_TICK_STEP))
    )

    next_send_time = time.monotonic()

    for index in range(1, command_count + 1):
        blend = quintic_smoothstep(index / command_count)
        command = np.rint(current + blend * delta).astype(np.int64)

        arm_writer.send(command)

        next_send_time += DIRECT_COMMAND_DT_SEC
        remaining = next_send_time - time.monotonic()

        if remaining > 0.0:
            time.sleep(remaining)
        elif reset_schedule_when_late:
            next_send_time = time.monotonic()

    return wait_arm_reached(
        port, packet, goal,
        timeout_sec=timeout_sec,
        thresholds=ARM_POSITION_THRESHOLDS,
    )

def setup_arm(port: PortHandler, packet: PacketHandler, arm_writer: ArmSyncWriter) -> np.ndarray:
    # Torque OFF is required to change Operating Mode.
    set_torque(port, packet, ARM_IDS, False)
    for dxl_id in ARM_IDS:
        write1(
            port,
            packet,
            dxl_id,
            ADDR_OPERATING_MODE,
            OP_EXTENDED_POSITION,
            f"ID {dxl_id} Extended Position Mode",
        )

    set_arm_profile_velocity(port, packet, START_PROFILE_VELOCITY)
    present = read_arm_ticks(port, packet)

    # Prevent a jump when torque is enabled: current position becomes the goal.
    arm_writer.send(present)
    set_torque(port, packet, ARM_IDS, True)
    time.sleep(0.2)
    return present


def move_to_initial_start(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    current_ticks: np.ndarray,
) -> np.ndarray:
    """
    Motor Control Node 시작 시 현재 실제 자세에서 ASSUMED_START_TICKS까지
    기존 standalone Dynamixel demo와 동일한 방식으로 부드럽게 이동한다.

    순서:
      1. setup_arm()에서 현재 위치를 Goal Position으로 기록한 뒤 Torque ON
      2. 현재 tick과 START tick의 차이를 검사
      3. quintic time scaling으로 START까지 이동
      4. wait_arm_reached()로 실제 도착 확인
    """
    current_ticks = np.asarray(
        current_ticks,
        dtype=np.int64,
    ).reshape(5)

    goal_ticks = np.asarray(
        ASSUMED_START_TICKS,
        dtype=np.int64,
    ).reshape(5)

    delta = goal_ticks - current_ticks

    print("\n" + "=" * 88)
    print("MOTOR CONTROL INITIAL START MOVE")
    print("=" * 88)

    for index, (now, goal, difference) in enumerate(
        zip(current_ticks, goal_ticks, delta),
        start=1,
    ):
        print(
            f"J{index}: "
            f"current={int(now)}, "
            f"start={int(goal)}, "
            f"delta={int(difference):+d}"
        )

    # 기존 standalone demo의 초기 이동 안전 검사 유지.
    if np.any(
        np.abs(delta)
        > START_MAX_ABS_DELTA_TICKS
    ):
        raise RuntimeError(
            "현재 위치와 START 사이 tick 차이가 안전 한도를 초과했습니다. "
            f"delta={delta.tolist()}, "
            f"limit={START_MAX_ABS_DELTA_TICKS.tolist()}"
        )

    max_delta = int(np.max(np.abs(delta)))

    if max_delta == 0:
        print(
            "이미 초기 START 위치입니다: "
            f"{goal_ticks.tolist()}"
        )
        return current_ticks.copy()

    count = max(
        2,
        int(np.ceil(max_delta / DIRECT_MAX_TICK_STEP)),
    )

    print(
        "초기 START 이동 시작 | "
        f"goal={goal_ticks.tolist()} | "
        f"commands={count} | "
        f"profile_velocity={START_PROFILE_VELOCITY}"
    )

    reached = move_arm_direct(
        port,
        packet,
        arm_writer,
        current_ticks,
        goal_ticks,
        profile_velocity=START_PROFILE_VELOCITY,
        timeout_sec=DIRECT_TIMEOUT_SEC,
        min_commands=2,
        skip_if_reached=True,
        reset_schedule_when_late=True,
    )

    print(
        "초기 START 위치 도착: "
        f"{reached.tolist()}"
    )

    return reached


def read_gripper_position(
    port: PortHandler,
    packet: PacketHandler,
) -> int:
    raw_position = read4_retry(
        port,
        packet,
        GRIPPER_ID,
        ADDR_PRESENT_POSITION,
        "Gripper Present Position",
    )
    return signed_int32(raw_position)


def setup_gripper(
    port: PortHandler,
    packet: PacketHandler,
) -> None:
    current_position = read_gripper_position(
        port,
        packet,
    )

    set_torque(
        port,
        packet,
        (GRIPPER_ID,),
        False,
    )

    write1(
        port,
        packet,
        GRIPPER_ID,
        ADDR_OPERATING_MODE,
        OP_POSITION_CONTROL,
        "Gripper Position Control Mode",
    )

    write4(
        port,
        packet,
        GRIPPER_ID,
        ADDR_PROFILE_VELOCITY,
        GRIPPER_PROFILE_VELOCITY,
        "Gripper Profile Velocity",
    )

    safe_current_position = int(
        np.clip(
            current_position,
            GRIPPER_MIN_TICK,
            GRIPPER_MAX_TICK,
        )
    )

    write4(
        port,
        packet,
        GRIPPER_ID,
        ADDR_GOAL_POSITION,
        safe_current_position,
        "Gripper current position as goal",
    )

    set_torque(
        port,
        packet,
        (GRIPPER_ID,),
        True,
    )
    time.sleep(0.2)


def send_gripper_position(
    port: PortHandler,
    packet: PacketHandler,
    goal_tick: int,
) -> None:
    goal_tick = int(
        np.clip(
            goal_tick,
            GRIPPER_MIN_TICK,
            GRIPPER_MAX_TICK,
        )
    )

    write4(
        port,
        packet,
        GRIPPER_ID,
        ADDR_GOAL_POSITION,
        goal_tick,
        f"Gripper Goal Position={goal_tick}",
    )


def wait_gripper_reached(
    port: PortHandler,
    packet: PacketHandler,
    goal_tick: int,
    threshold_tick: int = GRIPPER_POSITION_THRESHOLD_TICK,
    timeout_sec: float = GRIPPER_MOVE_TIMEOUT_SEC,
) -> int:
    goal_tick = int(goal_tick)
    deadline = time.monotonic() + float(timeout_sec)
    last_position = read_gripper_position(
        port,
        packet,
    )

    while time.monotonic() < deadline:
        last_position = read_gripper_position(
            port,
            packet,
        )
        error = goal_tick - last_position


        if abs(error) <= int(threshold_tick):
            time.sleep(GRIPPER_SETTLE_SEC)
            return last_position

        time.sleep(GRIPPER_SAMPLE_DT_SEC)

    raise RuntimeError(
        "Gripper position 도달 시간 초과 | "
        f"goal={goal_tick}, "
        f"last={last_position}, "
        f"threshold={threshold_tick}"
    )


def run_tick_phase(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    ticks_path: list[np.ndarray],
    start_index: int,
    goal_index: int,
    actual_ticks: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    held_j5_tick: int | None = None,
    min_commands: int | None = None,
    max_commands: int | None = None,
    phase_name: str = "ARM_PHASE",
) -> np.ndarray:
    """한 arm phase의 path 생성 -> streaming -> 목표 도달 확인을 공통 수행한다."""
    kwargs = {}
    if held_j5_tick is not None:
        kwargs["held_j5_tick"] = int(held_j5_tick)
    if min_commands is not None:
        kwargs["min_commands"] = int(min_commands)
    if max_commands is not None:
        kwargs["max_commands"] = int(max_commands)

    phase_path = [
        np.asarray(ticks_path[index], dtype=np.int64).copy()
        for index in range(start_index, goal_index + 1)
    ]
    phase_path[0] = np.asarray(
        actual_ticks,
        dtype=np.int64,
    ).reshape(5).copy()

    commands = build_phase_commands(
        phase_path,
        **kwargs,
    )
    stream_commands(
        arm_writer,
        commands,
    )

    goal = np.asarray(
        ticks_path[goal_index],
        dtype=np.int64,
    ).copy()

    if held_j5_tick is not None:
        goal[4] = int(held_j5_tick)

    return wait_waypoint_tracking(
        port,
        packet,
        arm_writer,
        goal,
        phase_name=phase_name,
        thresholds=(
            ARM_POSITION_THRESHOLDS
            if thresholds is None
            else thresholds
        ),
    )


def recover_to_start_after_collision(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    start_ticks: np.ndarray | None = None,
) -> np.ndarray:
    """Task 1 collision v5와 동일하게 HOLD 후 저속 START 복귀."""
    time.sleep(float(COLLISION_RECOVERY_WAIT_SEC))

    current = read_arm_ticks(port, packet)
    goal = np.asarray(
        ASSUMED_START_TICKS if start_ticks is None else start_ticks,
        dtype=np.int64,
    ).reshape(5)

    set_arm_profile_velocity(
        port,
        packet,
        int(COLLISION_RECOVERY_PROFILE_VELOCITY),
    )

    delta = goal - current
    command_count = max(
        2,
        int(
            np.ceil(
                np.max(np.abs(delta))
                / float(COLLISION_RECOVERY_MAX_TICK_STEP)
            )
        ),
    )

    next_send_time = time.monotonic()
    for index in range(1, command_count + 1):
        blend = quintic_smoothstep(index / command_count)
        command = np.rint(current + blend * delta).astype(np.int64)
        arm_writer.send(command)

        next_send_time += float(COLLISION_RECOVERY_COMMAND_DT_SEC)
        remaining = next_send_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_send_time = time.monotonic()

    return wait_arm_reached(
        port,
        packet,
        goal,
        timeout_sec=float(COLLISION_RECOVERY_TIMEOUT_SEC),
        thresholds=ARM_ALIGN_THRESHOLDS,
    )


def execute_tick_path(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    ticks_path: list[np.ndarray],
    waypoint_names: list[str],
    gripper_open_tick: int,
    gripper_close_tick: int,
) -> np.ndarray:
    if len(ticks_path) != len(waypoint_names):
        raise ValueError(
            "ticks_path and waypoint_names length mismatch"
        )

    index_map = {
        name: index
        for index, name in enumerate(waypoint_names)
        if name
    }

    required = (
        "START",
        "PICK_ABOVE",
        "PICK",
        "PICK_RETURN",
        "PLACE_ABOVE",
        "PLACE",
        "PLACE_RETURN",
    )

    missing = [
        name
        for name in required
        if name not in index_map
    ]

    if missing:
        raise ValueError(
            f"required waypoint missing: {missing}"
        )

    set_arm_profile_velocity(
        port,
        packet,
        ARM_PROFILE_VELOCITY,
    )

    actual_ticks = read_arm_ticks(
        port,
        packet,
    )

    start_i = index_map["START"]
    pick_above_i = index_map["PICK_ABOVE"]
    pick_i = index_map["PICK"]
    pick_return_i = index_map["PICK_RETURN"]
    place_above_i = index_map["PLACE_ABOVE"]
    place_i = index_map["PLACE"]
    place_return_i = index_map["PLACE_RETURN"]

    # PICK_ABOVE
    send_gripper_position(
        port,
        packet,
        gripper_open_tick,
    )
    wait_gripper_reached(
        port,
        packet,
        gripper_open_tick,
    )

    start_j5 = int(actual_ticks[4])
    j5_free_thresholds = ARM_ALIGN_THRESHOLDS.copy()
    j5_free_thresholds[4] = 1_000_000

    actual_ticks = run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        start_i,
        pick_above_i,
        actual_ticks,
        thresholds=j5_free_thresholds,
        held_j5_tick=start_j5,
    )

    actual_ticks = rotate_j5_only(
        port,
        packet,
        arm_writer,
        actual_ticks,
        int(ticks_path[pick_above_i][4]),
        phase_name="PICK_ABOVE_J5",
    )

    # PICK
    actual_ticks = run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        pick_above_i,
        pick_i,
        actual_ticks,
        phase_name="PICK",
    )

    send_gripper_position(
        port,
        packet,
        gripper_close_tick,
    )
    wait_gripper_reached(
        port,
        packet,
        gripper_close_tick,
    )

    # PICK_RETURN
    actual_ticks = run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        pick_i,
        pick_return_i,
        actual_ticks,
        thresholds=ARM_ALIGN_THRESHOLDS,
        phase_name="PICK_RETURN",
    )

    # PICK_RETURN -> PLACE_ABOVE
    held_j5 = int(actual_ticks[4])

    actual_ticks = run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        pick_return_i,
        place_above_i,
        actual_ticks,
        thresholds=j5_free_thresholds,
        held_j5_tick=held_j5,
        min_commands=TRANSPORT_MIN_COMMANDS,
        max_commands=TRANSPORT_MAX_COMMANDS,
        phase_name="PLACE_ABOVE_POSITION",
    )

    actual_ticks = rotate_j5_only(
        port,
        packet,
        arm_writer,
        actual_ticks,
        int(ticks_path[place_above_i][4]),
    )

    # PLACE
    actual_ticks = run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        place_above_i,
        place_i,
        actual_ticks,
        phase_name="PLACE",
    )

    send_gripper_position(
        port,
        packet,
        gripper_open_tick,
    )
    wait_gripper_reached(
        port,
        packet,
        gripper_open_tick,
    )

    # PLACE_RETURN
    return run_tick_phase(
        port,
        packet,
        arm_writer,
        ticks_path,
        place_i,
        place_return_i,
        actual_ticks,
        thresholds=ARM_ALIGN_THRESHOLDS,
        phase_name="PLACE_RETURN",
    )


def safe_read_gripper_tick(
    port: PortHandler,
    packet: PacketHandler,
    default: int = 0,
) -> int:
    try:
        return int(read_gripper_position(port, packet))
    except Exception:
        return int(default)


def recover_after_collision(
    node,
    error: Exception,
    log_success: bool = False,
) -> tuple[str, list[int]]:
    node.get_logger().error(str(error))
    try:
        with node.motion_lock:
            reached = recover_to_start_after_collision(
                node.port, node.packet, node.arm_writer
            )
        if log_success:
            node.get_logger().warn(
                f"[EMERGENCY RECOVERY] START 위치 복귀 완료 | ticks={reached.tolist()}"
            )
        return f"{error} | START recovery completed", [int(v) for v in reached]
    except Exception as recovery_error:
        return f"{error} | recovery failed: {recovery_error}", []


class MotorControlNode(Node):
    def __init__(self) -> None:
        super().__init__("motor_control_node")

        self.motion_lock = threading.RLock()
        self.callback_group = MutuallyExclusiveCallbackGroup()

        self.port = PortHandler(DEVICENAME)
        self.packet = PacketHandler(PROTOCOL_VERSION)

        self.arm_writer: ArmSyncWriter | None = None
        self.hardware_ready = False

        self.joint_pub = self.create_publisher(
            Int32MultiArray,
            "/motor/joint_state",
            10,
        )

        self.gripper_pub = self.create_publisher(
            Int32,
            "/motor/gripper_state",
            10,
        )

        self.move_srv = self.create_service(
            MoveToTicks,
            "/motor/move_to_ticks",
            self.handle_move_to_ticks,
            callback_group=self.callback_group,
        )

        self.execute_srv = self.create_service(
            ExecutePickPlace,
            "/arm/execute_pick_place",
            self.handle_execute,
            callback_group=self.callback_group,
        )

        self.delivery_srv = self.create_service(
            DeliverMike,
            "/motor/deliver_mike",
            self.handle_deliver_mike,
            callback_group=self.callback_group,
        )

        self.gripper_srv = self.create_service(
            MikeGripper,
            "/motor/mike_gripper",
            self.handle_mike_gripper,
            callback_group=self.callback_group,
        )

        self.off_srv = self.create_service(
            Trigger,
            "/motor/torque_off",
            self.handle_torque_off,
            callback_group=self.callback_group,
        )

        self.state_timer = self.create_timer(
            0.2,
            self.publish_state,
        )

        self.initialize_hardware()

        self.get_logger().info("Mission2 Motor ready")

    def initialize_hardware(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(
                f"cannot open {DEVICENAME}"
            )

        if not self.port.setBaudRate(BAUDRATE):
            raise RuntimeError(
                f"baudrate failed {BAUDRATE}"
            )

        self.arm_writer = ArmSyncWriter(
            self.port,
            self.packet,
        )

        current_ticks = setup_arm(
            self.port,
            self.packet,
            self.arm_writer,
        )

        self.get_logger().info(
            "Move to START | "
            f"current={current_ticks.tolist()} | "
            f"goal={ASSUMED_START_TICKS.tolist()}"
        )

        move_to_initial_start(
            self.port,
            self.packet,
            self.arm_writer,
            current_ticks,
        )

        setup_gripper(
            self.port,
            self.packet,
        )

        send_gripper_position(
            self.port,
            self.packet,
            DEFAULT_GRIPPER_OPEN_TICK,
        )

        wait_gripper_reached(
            self.port,
            self.packet,
            DEFAULT_GRIPPER_OPEN_TICK,
        )

        self.hardware_ready = True

    def publish_state(self) -> None:
        if not self.hardware_ready:
            return

        if not self.motion_lock.acquire(blocking=False):
            return

        try:
            joints = read_arm_ticks(
                self.port,
                self.packet,
            )

            gripper = read_gripper_position(
                self.port,
                self.packet,
            )

            joint_msg = Int32MultiArray()
            joint_msg.data = [int(value) for value in joints]
            self.joint_pub.publish(joint_msg)

            gripper_msg = Int32()
            gripper_msg.data = int(gripper)
            self.gripper_pub.publish(gripper_msg)

        except Exception:
            pass

        finally:
            self.motion_lock.release()

    def handle_move_to_ticks(
        self,
        request: MoveToTicks.Request,
        response: MoveToTicks.Response,
    ) -> MoveToTicks.Response:
        try:
            with self.motion_lock:
                goal = np.asarray(request.goal_ticks, dtype=np.int64).reshape(5)

                current = read_arm_ticks(self.port, self.packet)

                velocity = int(request.profile_velocity) if int(request.profile_velocity) > 0 else DIRECT_PROFILE_VELOCITY
                timeout = float(request.timeout_sec) if float(request.timeout_sec) > 0.0 else DIRECT_TIMEOUT_SEC

                reached = move_arm_direct(
                    self.port,
                    self.packet,
                    self.arm_writer,
                    current,
                    goal,
                    profile_velocity=velocity,
                    timeout_sec=timeout,
                )

                response.success = True
                response.message = f"{request.label} reached"
                response.reached_ticks = [int(value) for value in reached]

        except Exception as error:
            response.success = False
            response.message = str(error)
            response.reached_ticks = []

        return response


    def handle_execute(
        self,
        request: ExecutePickPlace.Request,
        response: ExecutePickPlace.Response,
    ) -> ExecutePickPlace.Response:
        try:
            with self.motion_lock:
                point_count = int(request.point_count)
                ticks_path = parse_tick_path(request.joint_ticks, point_count)

                if len(request.waypoint_names) != point_count:
                    raise ValueError("waypoint_names length mismatch")

                final_ticks = execute_tick_path(
                    self.port,
                    self.packet,
                    self.arm_writer,
                    ticks_path,
                    list(request.waypoint_names),
                    int(request.gripper_open_tick),
                    int(request.gripper_close_tick),
                )

                response.success = True
                response.message = "pick-place execution completed"
                response.final_joint_ticks = [int(value) for value in final_ticks]
                response.final_gripper_tick = int(read_gripper_position(self.port, self.packet))

        except CollisionDetected as error:
            response.success = False
            response.message, response.final_joint_ticks = recover_after_collision(
                self, error, log_success=True
            )
            response.final_gripper_tick = safe_read_gripper_tick(self.port, self.packet)

        except Exception as error:
            response.success = False
            response.message = str(error)
            response.final_joint_ticks = []

            response.final_gripper_tick = safe_read_gripper_tick(self.port, self.packet)

        return response

    def handle_deliver_mike(
        self,
        request: DeliverMike.Request,
        response: DeliverMike.Response,
    ) -> DeliverMike.Response:
        """Task 2 전달 tick 경로 실행."""
        try:
            with self.motion_lock:
                point_count = int(request.point_count)
                path = parse_tick_path(request.joint_ticks, point_count)

                current = read_arm_ticks(self.port, self.packet)
                path[0] = current.copy()

                velocity = int(request.profile_velocity) if int(request.profile_velocity) > 0 else ARM_PROFILE_VELOCITY
                timeout = float(request.timeout_sec) if float(request.timeout_sec) > 0.0 else ARM_WAYPOINT_TIMEOUT_SEC

                set_arm_profile_velocity(self.port, self.packet, velocity)

                commands = build_phase_commands(
                    path,
                    min_commands=max(PHASE_MIN_COMMANDS, len(path) - 1),
                    max_commands=PHASE_MAX_COMMANDS,
                )
                stream_commands(self.arm_writer, commands)

                reached = wait_waypoint_tracking(
                    self.port,
                    self.packet,
                    self.arm_writer,
                    np.asarray(path[-1], dtype=np.int64),
                    phase_name=str(request.label) or "TASK2_DELIVERY",
                    timeout_sec=timeout,
                    thresholds=ARM_POSITION_THRESHOLDS,
                )

                response.success = True
                response.message = f"{request.label} reached"
                response.final_joint_ticks = [int(v) for v in reached]

        except CollisionDetected as error:
            response.success = False
            response.message, response.final_joint_ticks = recover_after_collision(
                self, error
            )

        except Exception as error:
            response.success = False
            response.message = str(error)
            response.final_joint_ticks = []

        return response

    def handle_mike_gripper(
        self,
        request: MikeGripper.Request,
        response: MikeGripper.Response,
    ) -> MikeGripper.Response:
        """Gripper tick 이동."""
        try:
            with self.motion_lock:
                goal = int(request.goal_tick)
                timeout = float(request.timeout_sec) if float(request.timeout_sec) > 0.0 else GRIPPER_MOVE_TIMEOUT_SEC

                send_gripper_position(self.port, self.packet, goal)
                reached = wait_gripper_reached(
                    self.port, self.packet, goal, timeout_sec=timeout
                )

                response.success = True
                response.message = f"gripper reached {goal}"
                response.reached_tick = int(reached)

        except Exception as error:
            response.success = False
            response.message = str(error)
            response.reached_tick = safe_read_gripper_tick(self.port, self.packet)

        return response

    def torque_off(self) -> None:
        if not self.hardware_ready:
            return

        set_torque(self.port, self.packet, (*ARM_IDS, GRIPPER_ID), False)

        self.hardware_ready = False

        self.get_logger().warn("J1~J5 + Gripper Torque OFF")

    def handle_torque_off(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        try:
            with self.motion_lock:
                self.torque_off()

            response.success = True
            response.message = "all motor torque off"

        except Exception as error:
            response.success = False
            response.message = str(error)

        return response

    def destroy_node(self):
        try:
            self.torque_off()
        except Exception:
            pass

        try:
            self.port.closePort()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = MultiThreadedExecutor(num_threads=2)

    try:
        node = MotorControlNode()
        executor.add_node(node)
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()

        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
