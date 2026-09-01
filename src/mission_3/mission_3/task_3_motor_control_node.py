#!/usr/bin/env python3
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
from soomac_interfaces.srv import ExecutePickPlace, MoveToTicks
from control_config.robot_config import *
from control_config.task_3_config import (
    COLLISION_ACTIVE_ERROR_TICKS,
    COLLISION_MIN_PROGRESS_TICK,
    COLLISION_STALL_SEC,
    COLLISION_CHECK_PERIOD_SEC,
    TASK3_PHASE_MAX_TICK_STEP,
    TASK3_ARM_ALIGN_THRESHOLDS,
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


def quintic_smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _cumulative_tick_path_length(
    ticks: list[np.ndarray],
) -> np.ndarray:
    values = [
        np.asarray(item, dtype=float).reshape(5)
        for item in ticks
    ]
    cumulative = np.zeros(len(values), dtype=float)
    for index in range(1, len(values)):
        cumulative[index] = (
            cumulative[index - 1]
            + np.linalg.norm(
                values[index] - values[index - 1]
            )
        )
    return cumulative


def _sample_tick_path(
    ticks: list[np.ndarray],
    cumulative: np.ndarray,
    target_distance: float,
) -> np.ndarray:
    values = [
        np.asarray(item, dtype=float).reshape(5)
        for item in ticks
    ]

    if target_distance <= 0.0:
        sampled = values[0].copy()

    elif target_distance >= cumulative[-1]:
        sampled = values[-1].copy()

    else:
        upper = int(
            np.searchsorted(
                cumulative,
                target_distance,
                side="right",
            )
        )
        lower = upper - 1
        segment_length = cumulative[upper] - cumulative[lower]

        if segment_length <= 1.0e-12:
            sampled = values[upper].copy()
        else:
            alpha = (
                target_distance - cumulative[lower]
            ) / segment_length
            sampled = (
                (1.0 - alpha) * values[lower]
                + alpha * values[upper]
            )

    return sampled


def build_phase_commands(
    ticks: list[np.ndarray],
    held_j5_tick: int | None = None,
    min_commands: int = PHASE_MIN_COMMANDS,
    max_commands: int = PHASE_MAX_COMMANDS,
) -> list[np.ndarray]:
    commands: list[np.ndarray] = []

    if ticks:
        path = [
            np.asarray(item, dtype=np.int64).reshape(5).copy()
            for item in ticks
        ]

        if held_j5_tick is not None:
            for item in path:
                item[4] = int(held_j5_tick)

        if len(path) == 1:
            commands = [path[0].copy()]

        else:
            cumulative = _cumulative_tick_path_length(path)
            total_length = float(cumulative[-1])

            if total_length <= 1.0e-12:
                commands = [path[-1].copy()]

            else:
                joint_travel = np.sum(
                    np.abs(
                        np.diff(
                            np.asarray(path, dtype=np.int64),
                            axis=0,
                        )
                    ),
                    axis=0,
                )
                required = int(
                    np.max(
                        np.ceil(
                            joint_travel
                            / np.asarray(
                                TASK3_PHASE_MAX_TICK_STEP,
                                dtype=np.int64,
                            )
                        )
                    )
                )

                count = max(
                    int(min_commands),
                    int(required),
                    len(path) - 1,
                )
                count = min(
                    count,
                    int(max_commands),
                )

                for index in range(1, count + 1):
                    u = index / count
                    distance = (
                        quintic_smoothstep(u)
                        * total_length
                    )
                    command = np.rint(
                        _sample_tick_path(
                            path,
                            cumulative,
                            distance,
                        )
                    ).astype(np.int64)

                    if held_j5_tick is not None:
                        command[4] = int(held_j5_tick)

                    commands.append(command)

                final_command = path[-1].copy()

                if held_j5_tick is not None:
                    final_command[4] = int(held_j5_tick)

                commands[-1] = final_command

    return commands


def stream_commands(
    arm_writer: ArmSyncWriter,
    commands: list[np.ndarray],
) -> None:
    next_send_time = time.monotonic()
    for command in commands:
        arm_writer.send(command)
        next_send_time += ARM_STREAM_DT_SEC
        remaining = next_send_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_send_time = time.monotonic()


def phase_ticks(
    ticks_path: list[np.ndarray],
    start_index: int,
    end_index: int,
    actual_start_ticks: np.ndarray | None = None,
) -> list[np.ndarray]:
    result = [
        np.asarray(
            ticks_path[index],
            dtype=np.int64,
        ).copy()
        for index in range(start_index, end_index + 1)
    ]
    if actual_start_ticks is not None:
        result[0] = np.asarray(
            actual_start_ticks,
            dtype=np.int64,
        ).reshape(5).copy()
    return result


def align_thresholds_ignoring_j5() -> np.ndarray:
    thresholds = np.asarray(
        TASK3_ARM_ALIGN_THRESHOLDS,
        dtype=np.int64,
    ).reshape(5).copy()
    thresholds[4] = 1_000_000
    return thresholds


def rotate_j5_only(
    port: PortHandler,
    packet: PacketHandler,
    arm_writer: ArmSyncWriter,
    current_ticks: np.ndarray,
    target_j5_tick: int,
) -> np.ndarray:
    start = np.asarray(
        current_ticks,
        dtype=np.int64,
    ).reshape(5).copy()

    delta = int(target_j5_tick) - int(start[4])
    reached = start

    if abs(delta) > int(TASK3_ARM_ALIGN_THRESHOLDS[4]):
        next_send_time = time.monotonic()

        for index in range(1, TRANSPORT_YAW_COMMANDS + 1):
            u = index / TRANSPORT_YAW_COMMANDS
            blend = quintic_smoothstep(u)

            command = start.copy()
            command[4] = int(
                round(
                    start[4]
                    + blend * delta
                )
            )

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
            [
                1000000,
                1000000,
                1000000,
                1000000,
                TASK3_ARM_ALIGN_THRESHOLDS[4],
            ],
            dtype=np.int64,
        )

        reached = wait_arm_reached(
            port,
            packet,
            goal,
            timeout_sec=ARM_WAYPOINT_TIMEOUT_SEC,
            thresholds=yaw_thresholds,
        )

    return reached


def _int32_le_bytes(value: int) -> list[int]:
    unsigned = int(value) & 0xFFFFFFFF
    return [
        unsigned & 0xFF,
        (unsigned >> 8) & 0xFF,
        (unsigned >> 16) & 0xFF,
        (unsigned >> 24) & 0xFF,
    ]


class ArmSyncWriter:

    def __init__(self, port: PortHandler, packet: PacketHandler) -> None:
        self.packet = packet
        self.group = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)

    def send(self, ticks: np.ndarray) -> None:
        ticks = np.asarray(ticks, dtype=np.int64).reshape(5)
        try:
            for dxl_id, tick in zip(ARM_IDS, ticks):
                if not self.group.addParam(
                    int(dxl_id),
                    _int32_le_bytes(int(tick)),
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


def hold_arm_position(
    port: PortHandler,
    packet: PacketHandler,
    ticks: np.ndarray | None = None,
) -> np.ndarray:
    present = (
        read_arm_ticks(port, packet)
        if ticks is None
        else np.asarray(ticks, dtype=np.int64).reshape(5).copy()
    )
    writer = ArmSyncWriter(port, packet)
    writer.send(present)
    return present


def wait_arm_reached(
    port: PortHandler,
    packet: PacketHandler,
    goal_ticks: np.ndarray,
    timeout_sec: float = ARM_WAYPOINT_TIMEOUT_SEC,
    thresholds: np.ndarray = ARM_POSITION_THRESHOLDS,
) -> np.ndarray:
    goal_ticks = np.asarray(goal_ticks, dtype=np.int64).reshape(5)
    thresholds = np.asarray(thresholds, dtype=np.int64).reshape(5)
    collision_limits = np.asarray(
        COLLISION_ACTIVE_ERROR_TICKS,
        dtype=np.int64,
    ).reshape(5)

    deadline = time.monotonic() + float(timeout_sec)
    last_ticks = read_arm_ticks(port, packet)
    best_errors = np.abs(goal_ticks - last_ticks).astype(float)
    now0 = time.monotonic()
    last_progress_times = np.full(5, now0, dtype=float)

    while time.monotonic() < deadline:
        now = time.monotonic()
        last_ticks = read_arm_ticks(port, packet)
        abs_errors = np.abs(goal_ticks - last_ticks).astype(float)

        if np.all(abs_errors <= thresholds):
            return last_ticks

        for joint_index in range(5):
            improvement = best_errors[joint_index] - abs_errors[joint_index]
            if improvement >= float(COLLISION_MIN_PROGRESS_TICK):
                best_errors[joint_index] = abs_errors[joint_index]
                last_progress_times[joint_index] = now
            elif abs_errors[joint_index] < best_errors[joint_index]:
                best_errors[joint_index] = abs_errors[joint_index]

        active_mask = abs_errors > collision_limits.astype(float)
        stalled_mask = (now - last_progress_times) >= float(COLLISION_STALL_SEC)
        stalled_active = active_mask & stalled_mask

        if bool(np.any(stalled_active)):
            held = hold_arm_position(port, packet, last_ticks)
            stalled_joints = (np.where(stalled_active)[0] + 1).tolist()
            raise RuntimeError(
                'COLLISION_DETECTED | arm tracking stalled | '
                f'joints={stalled_joints} | '
                f'goal={goal_ticks.tolist()} | present={held.tolist()} | '
                f'error={(goal_ticks - held).tolist()} | '
                f'active_limit={collision_limits.tolist()} | '
                f'stall_sec={float(COLLISION_STALL_SEC):.2f}'
            )

        time.sleep(float(COLLISION_CHECK_PERIOD_SEC))

    errors = goal_ticks - last_ticks
    raise RuntimeError(
        'Arm waypoint 도달 시간 초과 | '
        f'goal={goal_ticks.tolist()} | present={last_ticks.tolist()} | '
        f'error={errors.tolist()} | limit={thresholds.tolist()}'
    )


def setup_arm(port: PortHandler, packet: PacketHandler, arm_writer: ArmSyncWriter) -> np.ndarray:
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
    current_ticks = np.asarray(
        current_ticks,
        dtype=np.int64,
    ).reshape(5)

    goal_ticks = np.asarray(
        ASSUMED_START_TICKS,
        dtype=np.int64,
    ).reshape(5)

    delta = goal_ticks - current_ticks

    if np.any(
        np.abs(delta)
        > START_MAX_ABS_DELTA_TICKS
    ):
        raise RuntimeError(
            "현재 위치와 START 사이 tick 차이가 안전 한도를 초과했습니다. "
            f"delta={delta.tolist()}, "
            f"limit={START_MAX_ABS_DELTA_TICKS.tolist()}"
        )

    max_delta = int(
        np.max(
            np.abs(delta)
        )
    )

    reached = current_ticks.copy()

    if max_delta != 0:
        set_arm_profile_velocity(
            port,
            packet,
            START_PROFILE_VELOCITY,
        )

        count = max(
            2,
            int(
                np.ceil(
                    max_delta
                    / DIRECT_MAX_TICK_STEP
                )
            ),
        )

        next_send_time = time.monotonic()

        for index in range(1, count + 1):
            u = index / count
            blend = quintic_smoothstep(u)

            command = np.rint(
                current_ticks
                + blend * delta
            ).astype(np.int64)

            arm_writer.send(command)

            next_send_time += DIRECT_COMMAND_DT_SEC
            remaining = (
                next_send_time
                - time.monotonic()
            )

            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_send_time = time.monotonic()

        reached = wait_arm_reached(
            port,
            packet,
            goal_ticks,
            timeout_sec=DIRECT_TIMEOUT_SEC,
            thresholds=ARM_POSITION_THRESHOLDS,
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


def run_phase(
    arm_writer: ArmSyncWriter,
    ticks_path: list[np.ndarray],
    start_index: int,
    end_index: int,
    actual_ticks: np.ndarray,
    *,
    held_j5_tick: int | None = None,
    min_commands: int = PHASE_MIN_COMMANDS,
    max_commands: int = PHASE_MAX_COMMANDS,
) -> None:
    commands = build_phase_commands(
        phase_ticks(
            ticks_path,
            start_index,
            end_index,
            actual_ticks,
        ),
        held_j5_tick=held_j5_tick,
        min_commands=min_commands,
        max_commands=max_commands,
    )
    stream_commands(arm_writer, commands)


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

    run_phase(
        arm_writer,
        ticks_path,
        start_i,
        pick_above_i,
        actual_ticks,
        held_j5_tick=start_j5,
    )

    goal = np.asarray(
        ticks_path[pick_above_i],
        dtype=np.int64,
    ).copy()
    goal[4] = start_j5

    thresholds = align_thresholds_ignoring_j5()

    actual_ticks = wait_arm_reached(
        port,
        packet,
        goal,
        thresholds=thresholds,
    )

    actual_ticks = rotate_j5_only(
        port,
        packet,
        arm_writer,
        actual_ticks,
        int(ticks_path[pick_above_i][4]),
    )

    run_phase(
        arm_writer,
        ticks_path,
        pick_above_i,
        pick_i,
        actual_ticks,
    )

    actual_ticks = wait_arm_reached(
        port,
        packet,
        np.asarray(
            ticks_path[pick_i],
            dtype=np.int64,
        ),
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

    run_phase(
        arm_writer,
        ticks_path,
        pick_i,
        pick_return_i,
        actual_ticks,
    )

    actual_ticks = wait_arm_reached(
        port,
        packet,
        np.asarray(
            ticks_path[pick_return_i],
            dtype=np.int64,
        ),
        thresholds=TASK3_ARM_ALIGN_THRESHOLDS,
    )

    held_j5 = int(actual_ticks[4])

    run_phase(
        arm_writer,
        ticks_path,
        pick_return_i,
        place_above_i,
        actual_ticks,
        held_j5_tick=held_j5,
        min_commands=TRANSPORT_MIN_COMMANDS,
        max_commands=TRANSPORT_MAX_COMMANDS,
    )

    place_above_goal = np.asarray(
        ticks_path[place_above_i],
        dtype=np.int64,
    ).copy()

    position_goal = place_above_goal.copy()
    position_goal[4] = held_j5

    thresholds = align_thresholds_ignoring_j5()

    actual_ticks = wait_arm_reached(
        port,
        packet,
        position_goal,
        thresholds=thresholds,
    )

    actual_ticks = rotate_j5_only(
        port,
        packet,
        arm_writer,
        actual_ticks,
        int(place_above_goal[4]),
    )

    run_phase(
        arm_writer,
        ticks_path,
        place_above_i,
        place_i,
        actual_ticks,
    )

    actual_ticks = wait_arm_reached(
        port,
        packet,
        np.asarray(
            ticks_path[place_i],
            dtype=np.int64,
        ),
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

    run_phase(
        arm_writer,
        ticks_path,
        place_i,
        place_return_i,
        actual_ticks,
    )

    return wait_arm_reached(
        port,
        packet,
        np.asarray(
            ticks_path[place_return_i],
            dtype=np.int64,
        ),
        thresholds=TASK3_ARM_ALIGN_THRESHOLDS,
    )


class MotorControlNode(Node):
    def __init__(self) -> None:
        super().__init__("task3_motor_control_node")

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

        self.get_logger().info(
            "Motor Control Node ready"
        )

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

    def publish_state(self):
        if not self.motion_lock.acquire(blocking=False):
            return

        try:
            present_ticks = read_arm_positions(
                self.port,
                self.packet,
            )

            message = Int32MultiArray()
            message.data = [
                int(value)
                for value in present_ticks
            ]
            self.state_pub.publish(message)

        except Exception as error:
            self.get_logger().warn(
                f"state publish failed: {error}"
            )

        finally:
            self.motion_lock.release()

    def handle_move_to_ticks(
        self,
        request: MoveToTicks.Request,
        response: MoveToTicks.Response,
    ) -> MoveToTicks.Response:
        try:
            with self.motion_lock:
                goal = np.asarray(
                    request.goal_ticks,
                    dtype=np.int64,
                ).reshape(5)

                current = read_arm_ticks(
                    self.port,
                    self.packet,
                )

                velocity = (
                    int(request.profile_velocity)
                    if int(request.profile_velocity) > 0
                    else DIRECT_PROFILE_VELOCITY
                )

                timeout = (
                    float(request.timeout_sec)
                    if float(request.timeout_sec) > 0.0
                    else DIRECT_TIMEOUT_SEC
                )

                set_arm_profile_velocity(
                    self.port,
                    self.packet,
                    velocity,
                )

                delta = goal - current

                command_count = max(
                    1,
                    int(
                        np.ceil(
                            np.max(np.abs(delta))
                            / DIRECT_MAX_TICK_STEP
                        )
                    ),
                )

                next_send_time = time.monotonic()

                for index in range(1, command_count + 1):
                    alpha = index / command_count
                    blend = quintic_smoothstep(alpha)

                    command = np.rint(
                        current + blend * delta
                    ).astype(np.int64)

                    self.arm_writer.send(command)

                    next_send_time += DIRECT_COMMAND_DT_SEC
                    remaining = (
                        next_send_time
                        - time.monotonic()
                    )

                    if remaining > 0.0:
                        time.sleep(remaining)

                reached = wait_arm_reached(
                    self.port,
                    self.packet,
                    goal,
                    timeout_sec=timeout,
                    thresholds=ARM_POSITION_THRESHOLDS,
                )

                response.success = True
                response.message = (
                    f"{request.label} reached"
                )
                response.reached_ticks = [
                    int(value)
                    for value in reached
                ]

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
                flat_ticks = np.asarray(
                    request.joint_ticks,
                    dtype=np.int64,
                )
                expected_size = point_count * 5

                if (
                    point_count <= 0
                    or flat_ticks.size != expected_size
                ):
                    raise ValueError(
                        "invalid joint_ticks: "
                        f"point_count={point_count}, "
                        f"size={flat_ticks.size}"
                    )

                if len(request.waypoint_names) != point_count:
                    raise ValueError(
                        "waypoint_names length mismatch"
                    )

                ticks_path = [
                    row.copy()
                    for row in flat_ticks.reshape(
                        point_count,
                        5,
                    )
                ]
                waypoint_names = list(request.waypoint_names)

                final_ticks = execute_tick_path(
                    self.port,
                    self.packet,
                    self.arm_writer,
                    ticks_path,
                    waypoint_names,
                    int(request.gripper_open_tick),
                    int(request.gripper_close_tick),
                )

                response.success = True
                response.message = "pick & place execution completed"
                response.final_joint_ticks = [
                    int(value)
                    for value in final_ticks
                ]
                response.final_gripper_tick = int(
                    read_gripper_position(
                        self.port,
                        self.packet,
                    )
                )

        except Exception as error:
            response.success = False
            response.message = str(error)
            response.final_joint_ticks = []
            response.final_gripper_tick = 0

        return response

    def torque_off(self) -> None:
        if not self.hardware_ready:
            return

        set_torque(
            self.port,
            self.packet,
            (*ARM_IDS, GRIPPER_ID),
            False,
        )

        self.hardware_ready = False

        self.get_logger().warn(
            "J1~J5 + Gripper Torque OFF"
        )

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
            with self.motion_lock:
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
    executor = MultiThreadedExecutor(
        num_threads=2
    )

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
