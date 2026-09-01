#!/usr/bin/env python3
"""Mission 2 arm control: detection, IK/planning, and motor-service requests."""
from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.optimize import least_squares
from std_msgs.msg import Int32, Int32MultiArray
from std_srvs.srv import Trigger

from soomac_interfaces.msg import Detection, DetectionArray
from soomac_interfaces.srv import DeliverMike, ExecutePickPlace, MikeGripper, MoveToTicks

from control_config.robot_config import *
from control_config.task_2_config import *

@dataclass
class SolverResult:
    success: bool
    q: np.ndarray
    position_error: float
    tool_error: float | None = None
    yaw_error: float | None = None
    minimum_margin_deg: float = float("-inf")


def run_ik_solver(
    residual_fn,
    seed_q_deg: np.ndarray,
    *,
    args: tuple = (),
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
    x_scale=None,
):
    """공통 IK least_squares 실행부. 각 solver의 residual/옵션 값은 그대로 받는다."""
    x0 = np.deg2rad(
        np.clip(
            np.asarray(seed_q_deg, dtype=float).reshape(5),
            JOINT_LIMIT_LOWER_DEG,
            JOINT_LIMIT_UPPER_DEG,
        )
    )

    options = {
        "fun": residual_fn,
        "x0": x0,
        "bounds": (JOINT_LIMIT_LOWER_RAD, JOINT_LIMIT_UPPER_RAD),
        "args": args,
        "method": "trf",
        "max_nfev": max_nfev,
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
        "verbose": 0,
    }
    if x_scale is not None:
        options["x_scale"] = x_scale

    return least_squares(**options)

def xyz_tool_down_yaw_residual(
    q_model_rad: np.ndarray,
    robot_chain,
    target_position_m: np.ndarray,
    target_yaw_axis: np.ndarray,
) -> np.ndarray:
    """
    SciPy least_squares가 최소화할 residual.

    구성:
        0~2: XYZ 위치 오차
        3~5: Tool-down 방향 오차
        6~8: Yaw 방향 오차

    포함하지 않는 항목:
        - 현재 자세 비용
        - seed 거리 비용
        - 특이점 비용
        - 충돌 비용
        - 관절 제한 soft cost
    """

    state = fk_state(robot_chain, q_model_rad)


    position_error_mm = (
        state["ee_position_m"]
        - target_position_m
    ) * 1000.0

    position_residual = (
        position_error_mm
        / POSITION_RESIDUAL_SCALE_MM
    )


    tool_axis_error = (
        state["tool_axis"]
        - WORLD_DOWN
    )

    tool_residual = (
        TOOL_RESIDUAL_WEIGHT
        * tool_axis_error
    )


    # 현재 physical gripper heading(local -X)이 목표 yaw 방향을 향하도록 한다.
    yaw_axis_error = (
        state["heading_axis"]
        - target_yaw_axis
    )

    yaw_residual = (
        YAW_RESIDUAL_WEIGHT
        * yaw_axis_error
    )

    return np.concatenate([
        position_residual,
        tool_residual,
        yaw_residual,
    ])


def create_seed_list_deg(
    target_position_mm: np.ndarray,
    target_yaw_deg: float,
) -> list[np.ndarray]:
    """Radial/opposite branch seed 생성."""
    target_position_mm = np.asarray(target_position_mm, dtype=float).reshape(3)
    radial_q1_deg = wrap_to_180_deg(
        np.rad2deg(np.arctan2(target_position_mm[1], target_position_mm[0]))
    )
    opposite_q1_deg = wrap_to_180_deg(radial_q1_deg + 180.0)

    previous_seed = PREVIOUS_POSITION_SOLUTION_DEG.copy()
    previous_seed[0] = radial_q1_deg
    previous_seed[4] = yaw_consistent_q5_deg(radial_q1_deg, target_yaw_deg)

    seeds = [previous_seed]
    branch_templates = (
        (10.0, -105.0, -85.0),
        (30.0, -120.0, -90.0),
        (-10.0, 95.0, 95.0),
        (-30.0, 80.0, 130.0),
    )

    for q1_deg in (radial_q1_deg, opposite_q1_deg):
        for q2_deg, q3_deg, q4_deg in branch_templates:
            seeds.append(np.array([
                q1_deg, q2_deg, q3_deg, q4_deg,
                yaw_consistent_q5_deg(q1_deg, target_yaw_deg),
            ], dtype=float))

    return seeds


def evaluate_solver_state(
    chain,
    q_rad: np.ndarray,
) -> tuple[dict, np.ndarray, dict]:
    """Solver FK/joint-limit 상태 계산."""
    q_rad = np.asarray(q_rad, dtype=float).reshape(5)
    state = fk_state(chain, q_rad)
    q_deg = np.asarray(state["q_model_deg"], dtype=float).reshape(5)
    limit_info = joint_limit_info(q_deg)
    return state, q_deg, limit_info

def make_pose_solver_result(
    q_deg: np.ndarray,
    limit_info: dict,
    position_error: float,
    position_tolerance: float,
    *,
    tool_error: float | None = None,
    tool_tolerance: float | None = None,
    yaw_error: float | None = None,
    yaw_tolerance: float | None = None,
) -> SolverResult:
    success = position_error <= position_tolerance and limit_info["inside"]
    if tool_tolerance is not None:
        success = success and tool_error <= tool_tolerance
    if yaw_tolerance is not None:
        success = success and yaw_error <= yaw_tolerance

    return SolverResult(
        success=bool(success),
        q=np.asarray(q_deg, dtype=float).reshape(5),
        position_error=float(position_error),
        tool_error=tool_error,
        yaw_error=yaw_error,
        minimum_margin_deg=limit_info["minimum_margin_deg"],
    )


def solve_single_seed(
    robot_chain,
    target_position_m: np.ndarray,
    target_yaw_deg: float,
    target_yaw_axis: np.ndarray,
    seed_q_deg: np.ndarray,
) -> SolverResult:
    seed_q_deg = np.asarray(seed_q_deg, dtype=float).reshape(5)

    try:
        result = run_ik_solver(
            xyz_tool_down_yaw_residual,
            seed_q_deg,
            args=(robot_chain, target_position_m, target_yaw_axis),
            max_nfev=MAX_FUNCTION_EVALUATIONS,
            ftol=FTOL,
            xtol=XTOL,
            gtol=GTOL,
        )

        q_rad = np.asarray(result.x, dtype=float).reshape(5)
        state, q_deg, limit_info = evaluate_solver_state(robot_chain, q_rad)
        position_error = float(
            np.linalg.norm((state["ee_position_m"] - target_position_m) * 1000.0)
        )
        tool_error = float(state["tool_error"])
        yaw_error = abs(calculate_yaw_error_deg(state["yaw"], target_yaw_deg))

        return make_pose_solver_result(
            q_deg,
            limit_info,
            position_error,
            POSITION_TOLERANCE_MM,
            tool_error=tool_error,
            tool_tolerance=TOOL_DOWN_TOLERANCE_DEG,
            yaw_error=yaw_error,
            yaw_tolerance=YAW_TOLERANCE_DEG,
        )

    except Exception:
        return SolverResult(
            False,
            np.full(5, np.nan),
            float("inf"),
            float("inf"),
            float("inf"),
        )


def solve_first_waypoint(
    robot_chain,
    waypoint: dict,
    start_q_deg: np.ndarray | None = None,
) -> list[SolverResult]:
    """첫 waypoint를 여러 seed로 풀어 candidate 목록을 반환한다."""
    if start_q_deg is None:
        start_q_deg = get_start_q_model_deg()

    start_q_deg = np.asarray(start_q_deg, dtype=float).reshape(5)
    target_xyz_mm = np.asarray(waypoint["xyz_mm"], dtype=float).reshape(3)
    target_yaw_deg = float(waypoint["yaw_deg"])

    actual_start_seed = start_q_deg.copy()
    actual_start_seed[4] = yaw_consistent_q5_deg(actual_start_seed[0], target_yaw_deg)
    actual_start_seed = np.clip(
        actual_start_seed, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG
    )

    seed_list = [actual_start_seed, *create_seed_list_deg(target_xyz_mm, target_yaw_deg)]
    target_yaw_axis = create_target_yaw_axis(target_yaw_deg)
    target_position_m = target_xyz_mm / 1000.0

    return [
        solve_single_seed(
            robot_chain, target_position_m, target_yaw_deg,
            target_yaw_axis, seed_q_deg
        )
        for seed_q_deg in seed_list
    ]


def fk_state(
    chain,
    q_rad: np.ndarray,
) -> dict:
    """FK 위치와 tool/yaw 상태를 계산한다."""
    q_rad = np.asarray(q_rad, dtype=float).reshape(5)

    ikpy_vector = model_q_to_ikpy_vector(q_rad)

    transform = np.asarray(chain.forward_kinematics(ikpy_vector), dtype=float)

    rotation = transform[:3, :3]
    xyz_m = transform[:3, 3]
    xyz_mm = xyz_m * 1000.0

    tool_state = calculate_tool_yaw_state(rotation)

    tool_axis = tool_state["tool_axis"]
    heading_axis = tool_state["heading_axis"]
    tool_error = float(tool_state["tool_down_error_deg"])
    yaw_deg = float(tool_state["yaw_deg"])

    return {
        "q_model_deg": np.rad2deg(q_rad),
        "ee_position_m": xyz_m,
        "xyz": xyz_mm,
        "tool_axis": tool_axis,
        "heading_axis": heading_axis,
        "yaw": yaw_deg,
        "tool_error": tool_error,
    }


def solve_position(
    chain,
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> SolverResult:
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    previous_q_rad = np.deg2rad(previous_q_deg)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float).reshape(3)

    def residual(q_rad: np.ndarray) -> np.ndarray:
        position_error = fk_state(chain, q_rad)["xyz"] - target_xyz
        posture_error = POSITION_POSTURE_WEIGHT * np.rad2deg(q_rad - previous_q_rad)
        return np.r_[position_error, posture_error]

    def solve_from_seed(seed_q_deg: np.ndarray) -> SolverResult:
        result = run_ik_solver(
            residual,
            seed_q_deg,
            max_nfev=POSITION_MAX_NFEV,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            x_scale="jac",
        )

        state, q_deg, limit_info = evaluate_solver_state(chain, result.x)
        error = float(np.linalg.norm(state["xyz"] - target_xyz))
        return SolverResult(
            success=error <= POSITION_TOLERANCE_MM,
            q=q_deg,
            position_error=error,
            minimum_margin_deg=limit_info["minimum_margin_deg"],
        )

    direct = solve_from_seed(previous_q_deg)
    if direct.success or np.linalg.norm(target_xyz[:2]) <= 1.0e-9:
        return direct

    backup_seed_q_deg = previous_q_deg.copy()
    backup_seed_q_deg[0] = wrap_to_180_deg(
        np.rad2deg(np.arctan2(target_xyz[1], target_xyz[0]))
    )
    backup = solve_from_seed(backup_seed_q_deg)

    if backup.success:
        return backup

    return min((direct, backup), key=lambda item: item.position_error)


def valid_j5_equivalents(angle_deg: float) -> np.ndarray:
    """J5 ±360° 동치각 중 physical limit 안의 값만 반환한다."""
    candidates = float(angle_deg) + np.array([-360.0, 0.0, 360.0])
    return candidates[
        (candidates >= JOINT_LIMIT_LOWER_DEG[4])
        & (candidates <= JOINT_LIMIT_UPPER_DEG[4])
    ]

def nearest_equivalent_j5_deg(
    angle_deg: float,
    reference_deg: float,
) -> float | None:
    """Return the closest equivalent J5 angle that remains inside ±200°."""
    valid = valid_j5_equivalents(angle_deg)
    if valid.size == 0:
        return None
    return float(valid[np.argmin(np.abs(valid - reference_deg))])


def joint_limit_info(q_deg: np.ndarray) -> dict:
    """전체 관절 제한 여부와 최소 여유각을 반환한다."""
    q_deg = np.asarray(q_deg, dtype=float).reshape(5)
    nearest_margin = np.minimum(
        q_deg - JOINT_LIMIT_LOWER_DEG,
        JOINT_LIMIT_UPPER_DEG - q_deg,
    )
    return {
        "inside": bool(np.all(nearest_margin >= -1.0e-6)),
        "minimum_margin_deg": float(np.min(nearest_margin)),
    }


def _mic_up_residual(
    q_rad: np.ndarray,
    chain,
    target_xyz_mm: np.ndarray,
    reference_q_rad: np.ndarray,
) -> np.ndarray:
    """Task 2 delivery residual: XYZ + microphone-up, yaw/tool-down은 강제하지 않는다."""
    state = fk_state(chain, q_rad)

    position_error = (
        state["xyz"] - np.asarray(target_xyz_mm, dtype=float).reshape(3)
    ) / 10.0

    # 현재 설계에서 microphone 길이 방향은 gripper heading(local -X)으로 본다.
    # 전달 시 이 축을 world +Z로 세운다.
    target_up_model = normalize_vector(
        world_xyz_to_model_xyz(np.array([0.0, 0.0, 1.0], dtype=float))
    )
    axis_error = MIC_UP_AXIS_WEIGHT * (
        state["heading_axis"] - target_up_model
    )

    posture_error = MIC_UP_POSTURE_WEIGHT * np.rad2deg(q_rad - reference_q_rad)

    return np.r_[position_error, axis_error, posture_error]


def solve_mic_up_single_seed(
    chain,
    waypoint: dict,
    seed_q_deg: np.ndarray,
    reference_q_deg: np.ndarray,
) -> SolverResult:
    seed_q_deg = np.asarray(seed_q_deg, dtype=float).reshape(5)
    reference_q_deg = np.asarray(reference_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float).reshape(3)

    result = run_ik_solver(
        _mic_up_residual,
        seed_q_deg,
        args=(chain, target_xyz, np.deg2rad(reference_q_deg)),
        max_nfev=MIC_UP_MAX_NFEV,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
    )

    state, q_deg, limit_info = evaluate_solver_state(chain, result.x)
    target_up_model = normalize_vector(
        world_xyz_to_model_xyz(np.array([0.0, 0.0, 1.0], dtype=float))
    )
    position_error = float(np.linalg.norm(state["xyz"] - target_xyz))
    mic_up_error = calculate_vector_angle_deg(state["heading_axis"], target_up_model)

    return make_pose_solver_result(
        q_deg,
        limit_info,
        position_error,
        MIC_UP_POSITION_TOLERANCE_MM,
        tool_error=mic_up_error,
        tool_tolerance=MIC_UP_AXIS_TOLERANCE_DEG,
    )


def solve_mic_up_pose(
    chain,
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> SolverResult:
    """MIC_UP deterministic multi-seed solver."""
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float).reshape(3)

    radial_q1 = (
        wrap_to_180_deg(np.rad2deg(np.arctan2(target_xyz[1], target_xyz[0])))
        if np.linalg.norm(target_xyz[:2]) > 1.0e-9
        else previous_q_deg[0]
    )

    seeds = [previous_q_deg.copy()]
    for q2, q3, q4 in (
        (0.0, 60.0, 30.0),
        (-20.0, 80.0, 30.0),
        (20.0, 60.0, 10.0),
        (-30.0, 100.0, 20.0),
        (30.0, 40.0, 20.0),
    ):
        seed = previous_q_deg.copy()
        seed[0] = float(np.clip(
            radial_q1, JOINT_LIMIT_LOWER_DEG[0], JOINT_LIMIT_UPPER_DEG[0]
        ))
        seed[1] = q2
        seed[2] = q3
        seed[3] = q4
        seeds.append(seed)

    results = [
        solve_mic_up_single_seed(chain, waypoint, seed, previous_q_deg)
        for seed in seeds
    ]

    successful = [item for item in results if item.success]
    if successful:
        return min(
            successful,
            key=lambda item: (
                item.position_error, item.tool_error, -item.minimum_margin_deg
            ),
        )

    return min(results, key=lambda item: (item.position_error, item.tool_error))


def solve_waypoint(
    chain,
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> SolverResult:
    mode = waypoint["mode"]

    if mode == POSITION_ONLY:
        return solve_position(chain, waypoint, previous_q_deg)
    if mode == MIC_UP_POSE:
        return solve_mic_up_pose(chain, waypoint, previous_q_deg)
    if mode != FULL_POSE:
        raise ValueError(f"지원하지 않는 constraint mode: {mode}")

    direct = solve_full_pose_single_seed(
        chain, waypoint, previous_q_deg, previous_q_deg
    )
    if direct.success:
        return direct

    if waypoint.get("name", "") == "PLACE_ABOVE":
        candidates = []
        for seed_q_deg in make_place_above_fast_seeds(waypoint, previous_q_deg):
            result = solve_full_pose_single_seed(
                chain, waypoint, seed_q_deg, previous_q_deg
            )
            candidates.append(result)
            if result.success:
                return result

        if candidates:
            return min(
                candidates,
                key=lambda item: (
                    item.position_error, item.tool_error, item.yaw_error
                ),
            )

    return direct


def append_j5_continuous_result(
    solved: list[dict],
    waypoint: dict,
    previous_q_deg: np.ndarray,
    solver_q_deg: np.ndarray,
) -> np.ndarray | None:
    """J5 branch 정렬 및 큰 이동 분할."""
    goal_q_deg = np.asarray(solver_q_deg, dtype=float).copy()
    continuous_j5 = nearest_equivalent_j5_deg(goal_q_deg[4], previous_q_deg[4])

    if continuous_j5 is None:
        return None

    goal_q_deg[4] = continuous_j5
    if not joint_limit_info(goal_q_deg)["inside"]:
        return None

    segment_count = max(
        int(np.ceil(abs(continuous_j5 - previous_q_deg[4]) / MAX_J5_WAYPOINT_STEP_DEG)),
        1,
    )

    for segment_index in range(1, segment_count + 1):
        alpha = segment_index / segment_count
        intermediate_q_deg = previous_q_deg + alpha * (goal_q_deg - previous_q_deg)
        solved.append({
            "waypoint": {
                **waypoint,
                "name": waypoint.get("name", "") if segment_index == segment_count else "",
            },
            "q_deg": intermediate_q_deg.copy(),
        })

    return goal_q_deg


def _fast_full_pose_residual(
    q_rad: np.ndarray,
    chain,
    target_xyz_mm: np.ndarray,
    target_yaw_deg: float,
    reference_q_rad: np.ndarray,
) -> np.ndarray:
    state = fk_state(chain, q_rad)

    position_error = (state["xyz"] - np.asarray(target_xyz_mm, dtype=float)) / 10.0

    tool_error = 10.0 * (state["tool_axis"] - WORLD_DOWN)

    target_yaw_rad = np.deg2rad(target_yaw_deg)
    target_heading = np.array([
        np.cos(target_yaw_rad),
        np.sin(target_yaw_rad),
        0.0,
    ], dtype=float)

    yaw_error = 10.0 * (state["heading_axis"] - target_heading)

    posture_error = 0.01 * np.rad2deg(q_rad - reference_q_rad)

    return np.r_[position_error, tool_error, yaw_error, posture_error]


def solve_full_pose_single_seed(
    chain,
    waypoint: dict,
    seed_q_deg: np.ndarray,
    reference_q_deg: np.ndarray | None = None,
) -> SolverResult:
    seed_q_deg = np.asarray(seed_q_deg, dtype=float).reshape(5)
    if reference_q_deg is None:
        reference_q_deg = seed_q_deg

    reference_q_deg = np.asarray(reference_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float).reshape(3)
    target_yaw = float(waypoint["yaw"])

    result = run_ik_solver(
        _fast_full_pose_residual,
        seed_q_deg,
        args=(chain, target_xyz, target_yaw, np.deg2rad(reference_q_deg)),
        max_nfev=FAST_FULL_POSE_MAX_NFEV,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        x_scale="jac",
    )

    state, q_deg, limit_info = evaluate_solver_state(chain, result.x)
    position_error = float(np.linalg.norm(state["xyz"] - target_xyz))
    yaw_error = abs(calculate_yaw_error_deg(state["yaw"], target_yaw))
    tool_error = float(state["tool_error"])

    return make_pose_solver_result(
        q_deg,
        limit_info,
        position_error,
        FAST_FULL_POSE_POSITION_TOLERANCE_MM,
        tool_error=tool_error,
        tool_tolerance=FAST_FULL_POSE_TOOL_TOLERANCE_DEG,
        yaw_error=yaw_error,
        yaw_tolerance=FAST_FULL_POSE_YAW_TOLERANCE_DEG,
    )


def make_place_above_fast_seeds(
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> list[np.ndarray]:
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float)
    target_yaw = float(waypoint["yaw"])

    radial_q1 = wrap_to_180_deg(np.rad2deg(np.arctan2(target_xyz[1], target_xyz[0])))
    q1_candidates = (
        float(previous_q_deg[0]),
        float(radial_q1),
        float(wrap_to_180_deg(radial_q1 + 180.0)),
    )

    seeds = []
    seen = set()

    for base_q1 in q1_candidates:
        for q1_offset in PLACE_ABOVE_BACKUP_Q1_OFFSETS_DEG:
            q1 = float(np.clip(
                wrap_to_180_deg(base_q1 + q1_offset),
                JOINT_LIMIT_LOWER_DEG[0], JOINT_LIMIT_UPPER_DEG[0],
            ))
            q5 = nearest_equivalent_j5_deg(
                yaw_consistent_q5_deg(q1, target_yaw),
                previous_q_deg[4],
            )
            if q5 is None:
                continue

            for j2_offset, j3_offset, j4_offset in PLACE_ABOVE_BACKUP_ARM_OFFSETS_DEG:
                seed = previous_q_deg.copy()
                seed[0] = q1
                seed[1:4] += (j2_offset, j3_offset, j4_offset)
                seed[4] = q5
                seed = np.clip(seed, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG)

                key = tuple(np.round(seed, 5))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(seed)

    return seeds


def solve_sequence(
    chain,
    waypoints: list[dict],
    initial_q_deg: np.ndarray,
) -> tuple[list[dict] | None, str | None]:
    q_deg = np.asarray(initial_q_deg, dtype=float)
    solved: list[dict] = []

    for index, waypoint in enumerate(waypoints[1:], start=1):
        result = solve_waypoint(chain, waypoint, q_deg)
        name = waypoint.get("name", "") or f"INTERMEDIATE_{index}"

        if not result.success:
            target_xyz = np.asarray(waypoint["xyz"], dtype=float).reshape(3)
            solved_q = np.asarray(result.q, dtype=float).reshape(5)

            if np.all(np.isfinite(solved_q)):
                solved_xyz = np.asarray(
                    fk_state(chain, np.deg2rad(solved_q))["xyz"], dtype=float
                ).reshape(3)
                error_xyz = solved_xyz - target_xyz
            else:
                solved_xyz = error_xyz = np.full(3, np.nan)
                solved_q = np.full(5, np.nan)

            detail = (
                f"{name} {waypoint['mode']} | "
                f"pos={result.position_error:.3f}mm"
            )
            if result.tool_error is not None:
                detail += f", tool={result.tool_error:.3f}deg"
            if result.yaw_error is not None:
                detail += f", yaw={result.yaw_error:.3f}deg"

            return None, (
                detail
                + f" | target_xyz={np.round(target_xyz, 3).tolist()}mm"
                + f" | solved_xyz={np.round(solved_xyz, 3).tolist()}mm"
                + f" | error_xyz={np.round(error_xyz, 3).tolist()}mm"
                + f" | q={np.round(solved_q, 3).tolist()}deg"
            )

        next_q_deg = append_j5_continuous_result(
            solved=solved,
            waypoint=waypoint,
            previous_q_deg=q_deg,
            solver_q_deg=result.q,
        )
        if next_q_deg is None:
            return None, f"{name} J5 continuity/limit reject"

        q_deg = next_q_deg

    return solved, None


def make_keypoints(
    start_xyz: np.ndarray,
    pick_xyz: np.ndarray,
    place_xyz: np.ndarray,
    start_yaw: float,
    pick_yaw: float,
    place_yaw: float,
    target_class_name: str | None = None,
) -> list[dict]:
    if target_class_name == "Bottle":
        place_above_z = place_xyz[2] + BOTTLE_PLACE_APPROACH_HEIGHT_MM
        pick_above_z = max(
            pick_xyz[2] + BOTTLE_PICK_CLEARANCE_MM,
            BOTTLE_SAFE_TRANSPORT_Z_MM,
        )
    else:
        place_above_z = place_xyz[2] + PLACE_APPROACH_HEIGHT_MM
        pick_above_z = max(
            pick_xyz[2] + 100.0,
            SAFE_ABOVE_Z_MM,
        )

    pick_above = np.array([pick_xyz[0], pick_xyz[1], pick_above_z])
    place_above = np.array([place_xyz[0], place_xyz[1], place_above_z])
    return [
        {"name": "START", "xyz": start_xyz, "yaw": start_yaw,
         "mode": POSITION_ONLY, "target_class_name": target_class_name},
        {"name": "PICK_ABOVE", "xyz": pick_above, "yaw": pick_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},
        {"name": "PICK", "xyz": pick_xyz, "yaw": pick_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},
        {"name": "PICK_RETURN", "xyz": pick_above, "yaw": pick_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},

        {"name": "PLACE_ABOVE", "xyz": place_above, "yaw": place_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},
        {"name": "PLACE", "xyz": place_xyz, "yaw": place_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},
        {"name": "PLACE_RETURN", "xyz": place_above, "yaw": place_yaw,
         "mode": FULL_POSE, "target_class_name": target_class_name},
    ]

def bezier_control(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    a0 = np.arctan2(start[1], start[0])
    a1 = np.arctan2(goal[1], goal[0])
    middle = a0 + 0.5 * np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    radius = max(np.linalg.norm(start[:2]), np.linalg.norm(goal[:2]), BEZIER_MIN_RADIUS_MM)
    z = max(start[2], goal[2], BEZIER_CONTROL_Z_MM)
    return np.array([radius * np.cos(middle), radius * np.sin(middle), z])

def interpolate(keypoints: list[dict]) -> list[dict]:
    waypoints = [dict(keypoints[0])]

    for start, goal in zip(keypoints[:-1], keypoints[1:]):
        transport = (
            start["name"] == "PICK_RETURN"
            and goal["name"] == "PLACE_ABOVE"
        )
        is_bottle_transport = (
            transport
            and start.get("target_class_name") == "Bottle"
        )

        yaw_delta = wrap_to_180_deg(
            goal["yaw"] - start["yaw"]
        )

        control = None
        p0 = p1 = p2 = p3 = None

        if is_bottle_transport:
            # Bottle 전용:
            # PICK_RETURN에서 높은 Z를 확보한 뒤
            # PLACE 쪽 XY까지 높은 상태를 유지하고,
            # 마지막 구간에서만 PLACE_ABOVE로 내려간다.
            p0 = np.asarray(start["xyz"], dtype=float)
            safe_z = max(
                float(p0[2]),
                float(BOTTLE_SAFE_TRANSPORT_Z_MM),
            )

            p1 = np.array([
                p0[0],
                p0[1],
                safe_z,
            ], dtype=float)

            p2 = np.array([
                goal["xyz"][0],
                goal["xyz"][1],
                safe_z,
            ], dtype=float)

            p3 = np.asarray(goal["xyz"], dtype=float)

            length = (
                np.linalg.norm(p1 - p0)
                + np.linalg.norm(p2 - p1)
                + np.linalg.norm(p3 - p2)
            )

        elif transport:
            control = bezier_control(
                start["xyz"],
                goal["xyz"],
            )
            length = (
                np.linalg.norm(control - start["xyz"])
                + np.linalg.norm(goal["xyz"] - control)
            )

        else:
            length = np.linalg.norm(
                goal["xyz"] - start["xyz"]
            )

        if transport:
            count = max(
                int(np.ceil(length / CARTESIAN_STEP_MM)),
                1,
            )
        else:
            count = max(
                int(np.ceil(length / CARTESIAN_STEP_MM)),
                int(np.ceil(abs(yaw_delta) / YAW_STEP_DEG)),
                1,
            )

        for index in range(1, count + 1):
            alpha = index / count

            if is_bottle_transport:
                xyz = (
                    (1.0 - alpha) ** 3 * p0
                    + 3.0 * (1.0 - alpha) ** 2 * alpha * p1
                    + 3.0 * (1.0 - alpha) * alpha**2 * p2
                    + alpha**3 * p3
                )

            elif control is None:
                xyz = (
                    (1.0 - alpha) * start["xyz"]
                    + alpha * goal["xyz"]
                )

            else:
                xyz = (
                    (1.0 - alpha) ** 2 * start["xyz"]
                    + 2.0 * (1.0 - alpha) * alpha * control
                    + alpha ** 2 * goal["xyz"]
                )

            if transport:
                is_final = index == count

                # PICK_RETURN -> PLACE_ABOVE 동안 yaw는 바꾸지 않는다.
                # 최종 PLACE_ABOVE에서만 place yaw를 FULL_POSE로 계산한다.
                if is_final:
                    yaw = goal["yaw"]
                    mode = goal["mode"]
                else:
                    yaw = start["yaw"]
                    mode = POSITION_ONLY
            else:
                yaw = wrap_to_180_deg(
                    start["yaw"] + alpha * yaw_delta
                )
                mode = goal["mode"]

            waypoints.append({
                "name": (
                    goal["name"]
                    if index == count
                    else ""
                ),
                "xyz": xyz,
                "yaw": yaw,
                "mode": mode,
                "target_class_name": start.get(
                    "target_class_name"
                ),
            })

    return waypoints

def joint_transition(
    start_q_deg: np.ndarray,
    goal_q_deg: np.ndarray,
) -> list[dict]:
    difference = goal_q_deg - start_q_deg
    count = max(
        int(np.ceil(np.max(np.abs(difference)) / MAX_START_JOINT_STEP_DEG)),
        1,
    )

    return [
        {
            "waypoint": {"name": "START" if index == 0 else "PICK_ABOVE" if index == count else ""},
            "q_deg": start_q_deg + (index / count) * difference,
        }
        for index in range(count + 1)
    ]

def choose_first_waypoint_j5_branch(
    candidate_q_deg: np.ndarray,
    start_q_deg: np.ndarray,
    place_xyz: np.ndarray,
    place_yaw_deg: float,
) -> np.ndarray | None:
    """PICK_ABOVE와 PLACE_ABOVE를 모두 고려해 연속적인 J5 branch를 고른다."""
    candidate_q_deg = np.asarray(candidate_q_deg, dtype=float).copy()
    start_q_deg = np.asarray(start_q_deg, dtype=float)
    place_xyz = np.asarray(place_xyz, dtype=float)

    pick_equivalents = valid_j5_equivalents(candidate_q_deg[4])
    if pick_equivalents.size == 0:
        return None

    place_q1_deg = wrap_to_180_deg(
        np.rad2deg(np.arctan2(place_xyz[1], place_xyz[0]))
    )
    place_nominal_j5_deg = yaw_consistent_q5_deg(
        place_q1_deg,
        place_yaw_deg,
    )

    place_equivalents = valid_j5_equivalents(place_nominal_j5_deg)
    if place_equivalents.size == 0:
        return None

    best_pick_j5 = None
    best_cost = np.inf

    for pick_j5 in pick_equivalents:
        start_delta = abs(pick_j5 - start_q_deg[4])
        place_delta = float(np.min(np.abs(place_equivalents - pick_j5)))

        # PLACE까지 이어질 수 없는 branch는 제외한다.
        if place_delta > 180.0:
            continue

        # 시작 이동보다 이후 Pick->Place 연속성을 더 중요하게 본다.
        cost = 0.25 * start_delta + place_delta

        if cost < best_cost:
            best_cost = cost
            best_pick_j5 = float(pick_j5)

    if best_pick_j5 is None:
        return None

    candidate_q_deg[4] = best_pick_j5

    return candidate_q_deg


def make_j5_path_continuous(path: list[dict]) -> list[dict] | None:
    """J5를 physical ±200° 안에서 연속적인 branch로 정리한다."""
    if not path:
        return path

    continuous_path: list[dict] = []
    previous_j5 = float(np.asarray(path[0]["q_deg"], dtype=float)[4])

    if not JOINT_LIMIT_LOWER_DEG[4] <= previous_j5 <= JOINT_LIMIT_UPPER_DEG[4]:
        return None

    for index, item in enumerate(path):
        copied = {
            "waypoint": dict(item["waypoint"]),
            "q_deg": np.asarray(item["q_deg"], dtype=float).copy(),
        }

        if index > 0:
            raw_j5 = float(copied["q_deg"][4])
            if (
                JOINT_LIMIT_LOWER_DEG[4] <= raw_j5 <= JOINT_LIMIT_UPPER_DEG[4]
                and abs(raw_j5 - previous_j5) <= MAX_J5_WAYPOINT_STEP_DEG + 1.0e-9
            ):
                continuous_j5 = raw_j5
            else:
                continuous_j5 = nearest_equivalent_j5_deg(raw_j5, previous_j5)

            if (
                continuous_j5 is None
                or abs(continuous_j5 - previous_j5) > MAX_J5_WAYPOINT_STEP_DEG
            ):
                return None

            copied["q_deg"][4] = continuous_j5

        previous_j5 = float(copied["q_deg"][4])
        continuous_path.append(copied)

    return continuous_path


def plan(
    chain,
    pick_xyz: np.ndarray,
    place_xyz: np.ndarray,
    pick_yaw: float,
    place_yaw: float,
    start_ticks: np.ndarray | None = None,
    target_class_name: str | None = None,
) -> list[dict]:
    start_ticks = np.asarray(
        ASSUMED_START_TICKS if start_ticks is None else start_ticks, dtype=np.int64
    ).reshape(5)
    start_q_rad = ticks_to_model_rad(start_ticks)
    start_q_deg = np.rad2deg(start_q_rad)
    start = fk_state(chain, start_q_rad)

    pick_xyz_model = world_xyz_to_model_xyz(
        np.asarray(pick_xyz, dtype=float).reshape(3)
    )
    place_xyz_model = world_xyz_to_model_xyz(
        np.asarray(place_xyz, dtype=float).reshape(3)
    )
    place_yaw_model = world_yaw_to_model_yaw(float(place_yaw))

    keypoints = make_keypoints(
        start["xyz"],
        pick_xyz_model,
        place_xyz_model,
        start["yaw"],
        world_yaw_to_model_yaw(float(pick_yaw)),
        place_yaw_model,
        target_class_name=target_class_name,
    )
    candidates = solve_first_waypoint(
        robot_chain=chain,
        waypoint={
            "xyz_mm": keypoints[1]["xyz"],
            "yaw_deg": keypoints[1]["yaw"],
        },
        start_q_deg=start_q_deg,
    )
    candidates = sorted(
        (item for item in candidates if item.success),
        key=lambda item: (
            float(np.linalg.norm(item.q - start_q_deg)),
            -float(item.minimum_margin_deg),
        ),
    )[:4]

    if not candidates:
        raise RuntimeError("FAST planner 실패 | PICK_ABOVE 성공 candidate 없음")

    path_waypoints = interpolate(keypoints[1:])
    failures = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_q = choose_first_waypoint_j5_branch(
            candidate_q_deg=candidate.q.copy(),
            start_q_deg=start_q_deg,
            place_xyz=place_xyz_model,
            place_yaw_deg=place_yaw_model,
        )
        if candidate_q is None:
            failures.append(
                f"candidate {candidate_index}: PICK_ABOVE J5 branch reject"
            )
            continue

        solved, failure = solve_sequence(chain, path_waypoints, candidate_q)
        if solved is None:
            failures.append(f"candidate {candidate_index}: {failure}")
            continue

        continuous_path = make_j5_path_continuous([
            *joint_transition(start_q_deg, candidate_q),
            *solved,
        ])
        if continuous_path is not None:
            return continuous_path

        failures.append(
            f"candidate {candidate_index}: final J5 path continuity reject"
        )

    raise RuntimeError("FAST planner 실패 | " + " | ".join(failures))


def path_to_ticks(
    path: list[dict],
    expected_start_ticks: np.ndarray,
) -> list[np.ndarray]:
    expected = np.asarray(expected_start_ticks, dtype=np.int64).reshape(5)

    ticks_path = []

    for index, item in enumerate(path):
        q_deg = np.asarray(item["q_deg"], dtype=float).reshape(5)

        ticks = model_deg_to_ticks(q_deg)

        if index == 0:
            error = ticks - expected

            if np.max(np.abs(error)) > 2:
                raise RuntimeError(
                    "planned START mismatch: "
                    f"converted={ticks.tolist()}, "
                    f"expected={expected.tolist()}, "
                    f"error={error.tolist()}"
                )

        ticks_path.append(ticks)

    return ticks_path


def point_to_numpy(point: Point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


def transform_point(transform: np.ndarray, point_xyz_m: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=float)
    point_h[:3] = np.asarray(point_xyz_m, dtype=float).reshape(3)
    return (np.asarray(transform, dtype=float).reshape(4, 4) @ point_h)[:3]


def load_t_flange_camera(path: str) -> np.ndarray:
    handeye_path = Path(path)

    if not handeye_path.exists():
        raise FileNotFoundError(
            f"Hand-Eye JSON이 없습니다: {handeye_path}"
        )

    with handeye_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    rotation = np.asarray(
        data["R_cam2gripper"],
        dtype=float,
    ).reshape(3, 3)

    translation = np.asarray(
        data["t_cam2gripper"],
        dtype=float,
    ).reshape(3)

    if not np.allclose(
        rotation @ rotation.T,
        np.eye(3),
        atol=1.0e-3,
    ):
        raise ValueError(
            "R_cam2gripper가 직교 회전행렬이 아닙니다."
        )

    if abs(
        float(np.linalg.det(rotation)) - 1.0
    ) > 1.0e-3:
        raise ValueError(
            "R_cam2gripper determinant가 1이 아닙니다."
        )

    transform = np.eye(
        4,
        dtype=float,
    )

    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def make_pick_only_keypoints(
    start_xyz: np.ndarray,
    pick_xyz: np.ndarray,
    start_yaw: float,
    pick_yaw: float,
) -> list[dict]:
    pick_above = np.array(
        [pick_xyz[0], pick_xyz[1], max(pick_xyz[2] + 100.0, SAFE_ABOVE_Z_MM)],
        dtype=float,
    )

    return [
        {"name": "START", "xyz": start_xyz, "yaw": start_yaw, "mode": POSITION_ONLY},
        {"name": "PICK_ABOVE", "xyz": pick_above, "yaw": pick_yaw, "mode": FULL_POSE},
        {"name": "PICK", "xyz": pick_xyz, "yaw": pick_yaw, "mode": FULL_POSE},
        {"name": "PICK_RETURN", "xyz": pick_above, "yaw": pick_yaw, "mode": FULL_POSE},
    ]


def plan_pick_only(
    chain,
    pick_xyz: np.ndarray,
    pick_yaw: float,
    start_ticks: np.ndarray,
) -> list[dict]:
    start_ticks = np.asarray(start_ticks, dtype=np.int64).reshape(5)
    start_q_rad = ticks_to_model_rad(start_ticks)
    start_q_deg = np.rad2deg(start_q_rad)
    start_state = fk_state(chain, start_q_rad)

    pick_xyz_model = world_xyz_to_model_xyz(
        np.asarray(pick_xyz, dtype=float).reshape(3)
    )
    pick_yaw_model = world_yaw_to_model_yaw(float(pick_yaw))

    keypoints = make_pick_only_keypoints(
        start_state["xyz"],
        pick_xyz_model,
        start_state["yaw"],
        pick_yaw_model,
    )

    candidates = solve_first_waypoint(
        robot_chain=chain,
        waypoint={
            "xyz_mm": keypoints[1]["xyz"],
            "yaw_deg": keypoints[1]["yaw"],
        },
        start_q_deg=start_q_deg,
    )
    candidates = sorted(
        (item for item in candidates if item.success),
        key=lambda item: (
            float(np.linalg.norm(item.q - start_q_deg)),
            -float(item.minimum_margin_deg),
        ),
    )[:4]

    if not candidates:
        raise RuntimeError(
            "FAST planner 실패 | microphone PICK_ABOVE 성공 candidate 없음"
        )

    waypoints = interpolate(keypoints[1:])
    failures = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_q = candidate.q.copy()
        continuous_j5 = nearest_equivalent_j5_deg(candidate_q[4], start_q_deg[4])
        if continuous_j5 is None:
            failures.append(f"candidate {candidate_index}: J5 branch reject")
            continue
        candidate_q[4] = continuous_j5

        solved, failure = solve_sequence(chain, waypoints, candidate_q)
        if solved is None:
            failures.append(f"candidate {candidate_index}: {failure}")
            continue

        continuous = make_j5_path_continuous([
            *joint_transition(start_q_deg, candidate_q),
            *solved,
        ])
        if continuous is not None:
            return continuous

        failures.append(f"candidate {candidate_index}: final J5 continuity reject")

    raise RuntimeError(
        "FAST planner가 microphone pick 경로를 찾지 못했습니다. | "
        + " | ".join(failures)
    )


def _joint_orientation_transition(
    start_q_deg: np.ndarray,
    goal_q_deg: np.ndarray,
    final_name: str,
) -> list[dict]:
    """같은 안전 위치에서 tool-down -> mic-up 자세를 joint space로 부드럽게 전환."""
    start_q_deg = np.asarray(start_q_deg, dtype=float).reshape(5)
    goal_q_deg = np.asarray(goal_q_deg, dtype=float).reshape(5)
    difference = goal_q_deg - start_q_deg
    count = max(
        int(np.ceil(np.max(np.abs(difference)) / MAX_START_JOINT_STEP_DEG)),
        1,
    )

    return [
        {
            "waypoint": {"name": final_name if index == count else ""},
            "q_deg": start_q_deg + (index / count) * difference,
        }
        for index in range(1, count + 1)
    ]


def _make_mic_up_cartesian_waypoints(
    start_xyz: np.ndarray,
    hand_xyz: np.ndarray,
) -> list[dict]:
    """MIC_UP_READY -> HAND_APPROACH -> HAND_PLACE. place-above/tool-down은 사용하지 않는다."""
    start_xyz = np.asarray(start_xyz, dtype=float).reshape(3)
    hand_xyz = np.asarray(hand_xyz, dtype=float).reshape(3)

    direction = hand_xyz - start_xyz
    distance = float(np.linalg.norm(direction))

    if distance > 1.0e-9:
        retreat = min(float(HAND_APPROACH_DISTANCE_MM), 0.5 * distance)
        hand_approach = hand_xyz - (direction / distance) * retreat
    else:
        hand_approach = hand_xyz.copy()

    keypoints = [
        {
            "name": "MIC_UP_READY",
            "xyz": start_xyz,
            "yaw": 0.0,
            "mode": MIC_UP_POSE,
        },
        {
            "name": "HAND_APPROACH",
            "xyz": hand_approach,
            "yaw": 0.0,
            "mode": MIC_UP_POSE,
        },
        {
            "name": "HAND_PLACE",
            "xyz": hand_xyz,
            "yaw": 0.0,
            "mode": MIC_UP_POSE,
        },
    ]

    waypoints = [dict(keypoints[0])]

    for start, goal in zip(keypoints[:-1], keypoints[1:]):
        length = float(np.linalg.norm(goal["xyz"] - start["xyz"]))
        count = max(int(np.ceil(length / CARTESIAN_STEP_MM)), 1)

        for index in range(1, count + 1):
            alpha = index / count
            xyz = (1.0 - alpha) * start["xyz"] + alpha * goal["xyz"]
            waypoints.append({
                "name": goal["name"] if index == count else "",
                "xyz": xyz,
                "yaw": 0.0,
                "mode": MIC_UP_POSE,
            })

    return waypoints


def plan_mic_delivery(
    chain,
    pick_xyz: np.ndarray,
    hand_xyz: np.ndarray,
    pick_yaw: float,
    start_ticks: np.ndarray,
) -> list[dict]:
    """Task 2 microphone delivery planner."""
    pick_path = plan_pick_only(
        chain=chain,
        pick_xyz=pick_xyz,
        pick_yaw=pick_yaw,
        start_ticks=start_ticks,
    )

    pick_return_i = named_index(pick_path, "PICK_RETURN")
    pick_return_q = np.asarray(pick_path[pick_return_i]["q_deg"], dtype=float).reshape(5)

    initial_ready_q = ticks_to_model_deg(
        np.asarray(TASK2_START_TICKS, dtype=np.int64).reshape(5)
    )
    initial_transition = _joint_orientation_transition(
        pick_return_q, initial_ready_q, "INITIAL_READY"
    )
    initial_ready_state = fk_state(chain, np.deg2rad(initial_ready_q))

    mic_up_ready_wp = {
        "name": "MIC_UP_READY",
        "xyz": initial_ready_state["xyz"].copy(),
        "yaw": 0.0,
        "mode": MIC_UP_POSE,
    }
    mic_up_result = solve_mic_up_pose(chain, mic_up_ready_wp, initial_ready_q)
    if not mic_up_result.success:
        raise RuntimeError(
            "초기 자세에서 microphone-up 전환 IK를 찾지 못했습니다. | "
            f"pos_err={mic_up_result.position_error:.3f}mm | "
            f"axis_err={mic_up_result.tool_error:.3f}deg"
        )

    mic_up_q = np.asarray(mic_up_result.q, dtype=float).reshape(5)
    mic_up_transition = _joint_orientation_transition(
        initial_ready_q, mic_up_q, "MIC_UP_READY"
    )

    hand_xyz_model = world_xyz_to_model_xyz(
        np.asarray(hand_xyz, dtype=float).reshape(3)
    )
    mic_up_state = fk_state(chain, np.deg2rad(mic_up_q))
    delivery_waypoints = _make_mic_up_cartesian_waypoints(
        mic_up_state["xyz"], hand_xyz_model
    )

    solved_delivery, failure = solve_sequence(chain, delivery_waypoints, mic_up_q)
    if solved_delivery is None:
        raise RuntimeError(
            "microphone-up 상태로 HAND까지의 전달 경로를 찾지 못했습니다. | "
            + str(failure)
        )

    continuous = make_j5_path_continuous([
        *pick_path, *initial_transition, *mic_up_transition, *solved_delivery,
    ])
    if continuous is None:
        raise RuntimeError("microphone delivery 경로의 J5 연속성을 만들지 못했습니다.")

    return continuous


def named_index(path: list[dict], name: str) -> int:
    indices = [
        index
        for index, item in enumerate(path)
        if item.get("waypoint", {}).get("name", "") == name
    ]
    if not indices:
        raise RuntimeError(f"planner path에 {name} waypoint가 없습니다.")
    return indices[-1]


class Mission2ArmControlNode(Node):
    def __init__(self) -> None:
        super().__init__("mission2_arm_control_node")

        self.callback_group = ReentrantCallbackGroup()
        self.motion_lock = threading.RLock()
        self.detection_condition = threading.Condition(threading.RLock())
        self.state_condition = threading.Condition(threading.RLock())


        self.chain = create_robot_chain()
        self.t_flange_camera = load_t_flange_camera(DEFAULT_HANDEYE_PATH)

        self.latest_detection_msg: DetectionArray | None = None
        self.latest_detection_stamp_ns = 0
        self.latest_joint_ticks: np.ndarray | None = None
        self.latest_gripper_tick: int | None = None

        # 초기 자세에서 검출한 hand 위치를 base/floor 좌표로 저장한다.
        # flange 좌표 자체는 팔이 움직이면 무효가 되므로 저장하지 않는다.
        self.saved_hand_floor_xyz_m: np.ndarray | None = None

        # RIGHT basket의 mic를 처음 검출했을 때 Task 1 좌표 변환으로 얻은 pick XYZ를 저장한다.
        # 질문 종료 후 table mic를 회수할 때 basket place 목표로 그대로 재사용한다.
        self.saved_right_mic_pick_xyz_mm: np.ndarray | None = None
        self.saved_right_mic_pick_yaw_deg: float | None = None

        self.create_subscription(
            DetectionArray,
            DETECTION_TOPIC,
            self.on_detections,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Int32MultiArray,
            MOTOR_JOINT_STATE_TOPIC,
            self.on_joint_state,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Int32,
            MOTOR_GRIPPER_STATE_TOPIC,
            self.on_gripper_state,
            10,
            callback_group=self.callback_group,
        )

        self.move_client = self.create_client(
            MoveToTicks,
            MOTOR_MOVE_SERVICE,
            callback_group=self.callback_group,
        )
        self.delivery_client = self.create_client(
            DeliverMike,
            MOTOR_DELIVER_MIKE_SERVICE,
            callback_group=self.callback_group,
        )
        self.pick_place_client = self.create_client(
            ExecutePickPlace,
            MOTOR_PICK_PLACE_SERVICE,
            callback_group=self.callback_group,
        )
        self.gripper_client = self.create_client(
            MikeGripper,
            MOTOR_MIKE_GRIPPER_SERVICE,
            callback_group=self.callback_group,
        )
        self.torque_off_client = self.create_client(
            Trigger,
            MOTOR_TORQUE_OFF_SERVICE,
            callback_group=self.callback_group,
        )

        # Scout 정지 후 HAND detection 시작 신호.
        self.create_service(
            Trigger,
            ARM_BEGIN_SEAT_SERVICE,
            self.handle_begin_seat,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            ARM_DELIVER_SERVICE,
            self.handle_deliver_mic,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            ARM_RETURN_SERVICE,
            self.handle_return_mic,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            ARM_HOME_SERVICE,
            self.handle_home,
            callback_group=self.callback_group,
        )


        self.get_logger().info(
            "Mission2 Arm ready: begin_seat -> deliver_mic -> return_mic -> home"
        )

    # ------------------------------------------------------------------
    # Detection / motor state cache
    # ------------------------------------------------------------------

    def on_detections(self, msg: DetectionArray) -> None:
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.detection_condition:
            if stamp_ns >= self.latest_detection_stamp_ns:
                self.latest_detection_msg = deepcopy(msg)
                self.latest_detection_stamp_ns = stamp_ns
                self.detection_condition.notify_all()


    def on_joint_state(self, msg: Int32MultiArray) -> None:
        if len(msg.data) != 5:
            return

        with self.state_condition:
            self.latest_joint_ticks = np.asarray(
                msg.data,
                dtype=np.int64,
            ).reshape(5)
            self.state_condition.notify_all()

    def on_gripper_state(self, msg: Int32) -> None:
        with self.state_condition:
            self.latest_gripper_tick = int(msg.data)
            self.state_condition.notify_all()

    def wait_joint_state(
        self,
        timeout_sec: float = 2.0,
    ) -> np.ndarray:
        deadline = time.monotonic() + float(timeout_sec)

        with self.state_condition:
            while self.latest_joint_ticks is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "/motor/joint_state를 받지 못했습니다."
                    )
                self.state_condition.wait(
                    timeout=min(remaining, 0.1)
                )

            return self.latest_joint_ticks.copy()

    def wait_for_category_after(
        self,
        category: str,
        gate_stamp_ns: int,
        timeout_sec: float,
    ) -> tuple[Detection | None, DetectionArray | None]:
        deadline = time.monotonic() + float(timeout_sec)

        with self.detection_condition:
            while time.monotonic() < deadline:
                msg = self.latest_detection_msg
                stamp_ns = self.latest_detection_stamp_ns

                if msg is not None and stamp_ns > gate_stamp_ns:
                    if REQUIRE_FLANGE_FRAME and msg.header.frame_id != "flange":
                        raise RuntimeError(
                            f"Detection frame='{msg.header.frame_id}'. "
                            "flange frame이 아닙니다."
                        )

                    candidates = [
                        detection
                        for detection in msg.detections
                        if detection.category == category
                        and float(detection.confidence) >= MINIMUM_CONFIDENCE
                    ]

                    if candidates:
                        selected = min(
                            candidates,
                            key=lambda item: float(item.distance_m),
                        )
                        return deepcopy(selected), deepcopy(msg)

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break

                self.detection_condition.wait(
                    timeout=min(remaining, 0.1)
                )

        return None, None

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def base_transform_from_ticks(
        self,
        ticks: np.ndarray,
    ) -> np.ndarray:
        q_model_rad = ticks_to_model_rad(
            np.asarray(ticks, dtype=float).reshape(5)
        )

        return np.asarray(
            self.chain.forward_kinematics(
                model_q_to_ikpy_vector(q_model_rad)
            ),
            dtype=float,
        )

    def detection_to_floor_xyz_m(
        self,
        detection: Detection,
        stationary_ticks: np.ndarray,
    ) -> np.ndarray:
        t_base_flange = self.base_transform_from_ticks(
            stationary_ticks
        )
        p_flange_m = point_to_numpy(detection.pose)
        p_base_m = transform_point(
            t_base_flange,
            p_flange_m,
        )

        return p_base_m + np.array(
            [0.0, 0.0, FLOOR_Z_OFFSET_M],
            dtype=float,
        )

    def camera_yaw_to_floor_yaw_deg(
        self,
        detection: Detection,
        stationary_ticks: np.ndarray,
    ) -> float:
        if not bool(getattr(detection, "yaw_valid", False)):
            return DEFAULT_MIC_PICK_YAW_DEG

        camera_yaw_deg = float(
            getattr(detection, "yaw_deg", 0.0)
        )

        if not np.isfinite(camera_yaw_deg):
            return DEFAULT_MIC_PICK_YAW_DEG

        t_base_flange = self.base_transform_from_ticks(
            stationary_ticks
        )
        r_base_camera = (
            t_base_flange[:3, :3]
            @ self.t_flange_camera[:3, :3]
        )

        yaw_rad = np.deg2rad(camera_yaw_deg)
        heading_camera = np.array(
            [
                np.cos(yaw_rad),
                np.sin(yaw_rad),
                0.0,
            ],
            dtype=float,
        )
        heading_base = r_base_camera @ heading_camera

        if np.linalg.norm(heading_base[:2]) < 1.0e-9:
            return DEFAULT_MIC_PICK_YAW_DEG

        return wrap_to_180_deg(
            np.rad2deg(
                np.arctan2(
                    heading_base[1],
                    heading_base[0],
                )
            )
        )

    def mic_pick_xyz_mm(
        self,
        mic_detection: Detection,
        stationary_ticks: np.ndarray,
        fine_offset_m: np.ndarray,
    ) -> np.ndarray:
        """Task 1 좌표 변환 후 실기 파지점 fine offset만 적용한다."""
        pick_floor_m = self.detection_to_floor_xyz_m(
            mic_detection,
            stationary_ticks,
        )

        pick_floor_m += np.asarray(
            fine_offset_m,
            dtype=float,
        ).reshape(3)

        return pick_floor_m * 1000.0

    def saved_hand_place_xyz_mm(self) -> np.ndarray:
        if self.saved_hand_floor_xyz_m is None:
            raise RuntimeError(
                "저장된 hand base 좌표가 없습니다. "
                "먼저 HAND detection을 수행하세요."
            )

        # hand detection을 Task 1과 동일하게 base/floor로 변환한 좌표를 그대로 사용한다.
        # Task 2 전달에서는 tool horn offset이나 place-above Z 보정을 추가하지 않는다.
        place_floor_m = (
            self.saved_hand_floor_xyz_m.copy()
            + np.asarray(
                HAND_PLACE_FINE_OFFSET_M,
                dtype=float,
            ).reshape(3)
        )

        return place_floor_m * 1000.0

    def saved_right_basket_place_xyz_mm(self) -> np.ndarray:
        if self.saved_right_mic_pick_xyz_mm is None:
            raise RuntimeError(
                "저장된 RIGHT basket mic pick 위치가 없습니다. "
                "먼저 deliver 과정에서 RIGHT mic를 검출/집어야 합니다."
            )

        offset_mm = (
            np.asarray(
                RIGHT_BASKET_PLACE_FINE_OFFSET_M,
                dtype=float,
            ).reshape(3)
            * 1000.0
        )
        return (
            self.saved_right_mic_pick_xyz_mm.copy()
            + offset_mm
        )

    def saved_right_basket_place_yaw_deg(self) -> float:
        if self.saved_right_mic_pick_yaw_deg is None:
            raise RuntimeError(
                "저장된 RIGHT basket mic pick yaw가 없습니다."
            )

        return float(
            self.saved_right_mic_pick_yaw_deg
            + RIGHT_BASKET_PLACE_YAW_OFFSET_DEG
        )

    # ------------------------------------------------------------------
    # Motor service helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_future(
        future,
        timeout_sec: float,
        name: str,
    ):
        deadline = time.monotonic() + float(timeout_sec)

        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{name} timeout ({timeout_sec:.1f}s)"
                )
            time.sleep(0.02)

        return future.result()

    def call_motor_service(
        self,
        client,
        request,
        service_name: str,
        label: str,
    ):
        if not client.wait_for_service(
            timeout_sec=MOTOR_SERVICE_WAIT_SEC
        ):
            raise RuntimeError(
                f"{service_name} 서비스가 없습니다."
            )

        response = self._wait_future(
            client.call_async(request),
            MOTOR_SERVICE_TIMEOUT_SEC,
            label,
        )

        if response is None or not response.success:
            raise RuntimeError(
                response.message
                if response is not None
                else f"{label} 응답 없음"
            )

        return response

    def move_to_ticks(
        self,
        goal_ticks: np.ndarray,
        label: str,
    ) -> np.ndarray:
        request = MoveToTicks.Request()
        request.goal_ticks = [
            int(value) for value in np.asarray(goal_ticks, dtype=np.int64).reshape(5)
        ]
        request.profile_velocity = int(DIRECT_MOVE_PROFILE_VELOCITY)
        request.timeout_sec = float(DIRECT_MOVE_TIMEOUT_SEC)
        request.label = str(label)

        response = self.call_motor_service(
            self.move_client,
            request,
            MOTOR_MOVE_SERVICE,
            label,
        )

        return np.asarray(response.reached_ticks, dtype=np.int64).reshape(5)

    @staticmethod
    def flatten_tick_path(
        ticks_path: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[int]]:
        values = [np.asarray(item, dtype=np.int64).reshape(5) for item in ticks_path]
        flat_ticks = [int(value) for row in values for value in row]
        return values, flat_ticks

    def execute_delivery_trajectory(
        self,
        ticks_path: list[np.ndarray],
        label: str,
    ) -> np.ndarray:
        if not ticks_path:
            raise ValueError("빈 trajectory 입니다.")

        values, flat_ticks = self.flatten_tick_path(
            ticks_path
        )

        request = DeliverMike.Request()
        request.point_count = len(values)
        request.joint_ticks = flat_ticks
        request.profile_velocity = int(ARM_PROFILE_VELOCITY)
        request.timeout_sec = float(MOTOR_SERVICE_TIMEOUT_SEC)
        request.label = str(label)

        response = self.call_motor_service(
            self.delivery_client,
            request,
            MOTOR_DELIVER_MIKE_SERVICE,
            label,
        )

        return np.asarray(response.final_joint_ticks, dtype=np.int64).reshape(5)

    def execute_pick_place_trajectory(
        self,
        path: list[dict],
        ticks_path: list[np.ndarray],
        label: str,
    ) -> np.ndarray:
        """일반 Pick & Place 경로를 기존 ExecutePickPlace.srv로 실행한다."""
        if not ticks_path:
            raise ValueError("빈 Pick & Place trajectory 입니다.")

        if len(path) != len(ticks_path):
            raise ValueError("path와 ticks_path 길이가 다릅니다.")

        values, flat_ticks = self.flatten_tick_path(
            ticks_path
        )
        waypoint_names = [
            str(item["waypoint"].get("name", ""))
            for item in path
        ]

        request = ExecutePickPlace.Request()
        request.point_count = len(values)
        request.joint_ticks = flat_ticks
        request.waypoint_names = waypoint_names
        request.gripper_open_tick = int(MIC_GRIPPER_OPEN_TICK)
        request.gripper_close_tick = int(MIC_GRIPPER_CLOSE_TICK)

        response = self.call_motor_service(
            self.pick_place_client,
            request,
            MOTOR_PICK_PLACE_SERVICE,
            label,
        )

        return np.asarray(response.final_joint_ticks, dtype=np.int64).reshape(5)

    def set_gripper(self, goal_tick: int) -> int:
        request = MikeGripper.Request()
        request.goal_tick = int(goal_tick)
        request.timeout_sec = float(
            GRIPPER_MOVE_TIMEOUT_SEC
        )

        response = self.call_motor_service(
            self.gripper_client,
            request,
            MOTOR_MIKE_GRIPPER_SERVICE,
            "set_gripper",
        )

        return int(response.reached_tick)

    def move_home(self) -> np.ndarray:
        return self.move_to_ticks(
            TASK2_START_TICKS,
            "TASK2_HOME",
        )

    def validate_dynamic_start(
        self,
        expected_ticks: np.ndarray,
    ) -> None:
        actual = self.wait_joint_state()
        expected = np.asarray(
            expected_ticks,
            dtype=np.int64,
        ).reshape(5)
        error = actual - expected

        if np.max(np.abs(error)) > DYNAMIC_START_MAX_ERROR_TICK:
            raise RuntimeError(
                "계획 시작 자세와 실제 자세 차이가 큽니다. "
                f"expected={expected.tolist()}, "
                f"actual={actual.tolist()}, "
                f"error={error.tolist()}"
            )

    # ------------------------------------------------------------------
    # Task 2 startup / perception helpers
    # ------------------------------------------------------------------

    def move_to_initial_detection_view(self) -> np.ndarray:
        """Task 2 시작 시 사람 손을 인식하는 초기 자세로 이동한다."""
        return self.move_to_ticks(
            TASK2_START_TICKS,
            "TASK2_INITIAL_VIEW",
        )

    def wait_fresh_detection(
        self,
        category: str,
        view_label: str,
    ) -> Detection:
        gate_ns = self.get_clock().now().nanoseconds
        time.sleep(DETECTION_SETTLE_SEC)

        detection, _ = self.wait_for_category_after(
            category,
            gate_stamp_ns=gate_ns,
            timeout_sec=DETECTION_WAIT_SEC,
        )
        if detection is None:
            raise RuntimeError(
                f"{view_label}에서 새 {category} detection을 받지 못했습니다."
            )
        return detection

    def detect_and_save_hand(
        self,
        reached_ticks: np.ndarray,
    ) -> Detection:
        hand = self.wait_fresh_detection(
            Detection.CATEGORY_HAND,
            "초기 자세",
        )

        self.saved_hand_floor_xyz_m = self.detection_to_floor_xyz_m(
            hand,
            reached_ticks,
        )

        self.get_logger().info(
            "HAND 저장 완료 | floor/base xyz[m]="
            f"{np.round(self.saved_hand_floor_xyz_m, 4).tolist()}"
        )

        return hand

    def detect_and_save_right_microphone(
        self,
    ) -> tuple[Detection, np.ndarray]:
        """RIGHT basket view로 이동한 뒤 mic를 검출하고 실제 pick 목표를 저장한다."""
        right_view_ticks = self.require_pose_configured(
            RIGHT_MIC_DETECTION_TICKS,
            "RIGHT_MIC_DETECTION_TICKS",
        )

        reached = self.move_to_ticks(
            right_view_ticks,
            "RIGHT_MIC_DETECTION_VIEW",
        )

        mic = self.wait_fresh_detection(
            Detection.CATEGORY_MIC,
            "RIGHT_MIC_DETECTION_VIEW",
        )

        reached = np.asarray(reached, dtype=np.int64).reshape(5)

        # 회수 place 목표 저장.
        self.saved_right_mic_pick_xyz_mm = self.mic_pick_xyz_mm(
            mic,
            reached,
            RIGHT_MIC_PICK_FINE_OFFSET_M,
        )
        self.saved_right_mic_pick_yaw_deg = self.camera_yaw_to_floor_yaw_deg(
            mic,
            reached,
        )

        self.get_logger().info(
            "RIGHT MIC pick 위치 저장 완료 | xyz[mm]="
            f"{np.round(self.saved_right_mic_pick_xyz_mm, 2).tolist()} "
            f"yaw={self.saved_right_mic_pick_yaw_deg:.2f} deg"
        )

        return mic, reached

    def detect_table_microphone(
        self,
    ) -> tuple[Detection, np.ndarray]:
        table_view_ticks = self.require_pose_configured(
            TABLE_MIC_DETECTION_TICKS,
            "TABLE_MIC_DETECTION_TICKS",
        )

        reached = self.move_to_ticks(
            table_view_ticks,
            "TABLE_MIC_DETECTION_VIEW",
        )

        mic = self.wait_fresh_detection(
            Detection.CATEGORY_MIC,
            "TABLE_MIC_DETECTION_VIEW",
        )

        return mic, reached

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def plan_delivery_cycle(
        self,
        start_ticks: np.ndarray,
    ) -> tuple[list[dict], list[np.ndarray]]:
        if self.saved_right_mic_pick_xyz_mm is None:
            raise RuntimeError("저장된 RIGHT mic pick 위치가 없습니다.")
        if self.saved_right_mic_pick_yaw_deg is None:
            raise RuntimeError("저장된 RIGHT mic pick yaw가 없습니다.")

        pick_xyz_mm = self.saved_right_mic_pick_xyz_mm.copy()
        pick_yaw_deg = float(self.saved_right_mic_pick_yaw_deg)
        hand_xyz_mm = self.saved_hand_place_xyz_mm()

        path = plan_mic_delivery(
            self.chain,
            pick_xyz_mm,
            hand_xyz_mm,
            pick_yaw_deg,
            start_ticks=start_ticks,
        )

        return path, path_to_ticks(
            path,
            start_ticks,
        )

    def plan_return_cycle(
        self,
        mic: Detection,
        start_ticks: np.ndarray,
    ) -> tuple[list[dict], list[np.ndarray]]:
        pick_xyz_mm = self.mic_pick_xyz_mm(
            mic,
            start_ticks,
            TABLE_MIC_PICK_FINE_OFFSET_M,
        )
        pick_yaw_deg = self.camera_yaw_to_floor_yaw_deg(
            mic,
            start_ticks,
        )

        # 처음 RIGHT basket에서 mic를 검출할 때 저장한 pick 위치를
        # 회수 후 place 목표로 그대로 재사용한다.
        place_xyz_mm = self.saved_right_basket_place_xyz_mm()
        place_yaw_deg = self.saved_right_basket_place_yaw_deg()

        path = plan(
            self.chain,
            pick_xyz_mm,
            place_xyz_mm,
            pick_yaw_deg,
            place_yaw_deg,
            start_ticks=start_ticks,
            target_class_name=None,
        )

        return path, path_to_ticks(
            path,
            start_ticks,
        )

    # ------------------------------------------------------------------
    # Task 2 motion sequence
    # ------------------------------------------------------------------

    def deliver_microphone(self) -> None:
        """hand 저장 -> RIGHT mic pick -> mic-up 전달 -> release -> 초기 복귀 -> table search view."""
        self.saved_hand_floor_xyz_m = None
        self.saved_right_mic_pick_xyz_mm = None
        self.saved_right_mic_pick_yaw_deg = None

        self.set_gripper(MIC_GRIPPER_OPEN_TICK)

        # 1) ManualScout2Node의 /arm/begin_seat에서 HAND view가 준비된다.
        #    Manager의 0.3m 접근이 끝난 뒤 현재 실제 arm tick 기준으로
        #    새 hand detection을 받아 최종 전달 위치를 저장한다.
        hand_view_ticks = self.wait_joint_state()
        self.detect_and_save_hand(hand_view_ticks)

        # 2) RIGHT basket 전용 view로 이동한 뒤 mic를 검출하고
        #    실제 pick 위치/yaw를 저장한다.
        _, right_mic_view_ticks = self.detect_and_save_right_microphone()

        # 3) RIGHT mic view의 실제 도달 자세를 시작점으로
        #    검출된 mic는 tool-down으로 pick하고, 이후 mic-up 자세로 hand까지 전달한다.
        path, ticks_path = self.plan_delivery_cycle(
            right_mic_view_ticks,
        )

        pick_i = named_index(path, "PICK")
        initial_ready_i = named_index(path, "INITIAL_READY")
        hand_approach_i = named_index(path, "HAND_APPROACH")
        hand_place_i = named_index(path, "HAND_PLACE")

        self.validate_dynamic_start(right_mic_view_ticks)

        # PICK: tool-down + detected mic yaw.
        reached = self.execute_delivery_trajectory(
            ticks_path[: pick_i + 1],
            "DETECTED_RIGHT_MIC_PICK",
        )

        self.set_gripper(MIC_GRIPPER_CLOSE_TICK)

        # PICK -> PICK_RETURN -> INITIAL_READY.
        reached = self.execute_delivery_trajectory(
            [
                reached,
                *ticks_path[pick_i + 1 : initial_ready_i + 1],
            ],
            "PICK_RETURN_TO_INITIAL",
        )

        # INITIAL_READY -> MIC_UP -> HAND.
        reached = self.execute_delivery_trajectory(
            [
                reached,
                *ticks_path[initial_ready_i + 1 : hand_place_i + 1],
            ],
            "INITIAL_MIC_UP_TO_SAVED_HAND",
        )

        self.get_logger().info(
            f"저장된 HAND 위치 도착 -> {HAND_RELEASE_DELAY_SEC:.1f}초 뒤 gripper release"
        )
        time.sleep(max(0.0, HAND_RELEASE_DELAY_SEC))

        self.set_gripper(MIC_GRIPPER_OPEN_TICK)

        # release 후 HAND_APPROACH까지 후퇴.
        retreat_to_approach = list(
            reversed(
                ticks_path[hand_approach_i:hand_place_i]
            )
        )
        if retreat_to_approach:
            reached = self.execute_delivery_trajectory(
                [reached, *retreat_to_approach],
                "HAND_PLACE_TO_APPROACH",
            )

        # 전달 경로 역순으로 INITIAL_READY 복귀.
        return_to_initial = list(
            reversed(
                ticks_path[initial_ready_i:hand_approach_i]
            )
        )
        if return_to_initial:
            reached = self.execute_delivery_trajectory(
                [reached, *return_to_initial],
                "HAND_APPROACH_TO_INITIAL",
            )

        # TASK2_START_TICKS로 최종 정렬.
        reached = self.move_to_ticks(
            TASK2_START_TICKS,
            "TASK2_POST_DELIVERY_HOME",
        )

        # table mic search 자세 이동.
        table_search_ticks = self.require_pose_configured(
            TABLE_MIC_DETECTION_TICKS,
            "TABLE_MIC_DETECTION_TICKS",
        )
        self.move_to_ticks(
            table_search_ticks,
            "TABLE_MIC_SEARCH_READY",
        )

    def return_microphone(self) -> None:
        """TABLE view -> mic detection -> pick -> 처음 저장한 RIGHT mic 위치에 place."""
        if self.saved_right_mic_pick_xyz_mm is None:
            raise RuntimeError(
                "RIGHT basket 원래 mic 위치가 저장되어 있지 않습니다. "
                "먼저 /arm/deliver_mic을 성공시켜야 합니다."
            )

        # 책상 mic fresh detection.
        mic, start_ticks = self.detect_table_microphone()

        # detected mic pick -> saved RIGHT basket place.
        path, ticks_path = self.plan_return_cycle(
            mic,
            start_ticks,
        )

        self.validate_dynamic_start(start_ticks)
        self.execute_pick_place_trajectory(
            path,
            ticks_path,
            "TABLE_MIC_TO_SAVED_RIGHT_POSITION",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def require_pose_configured(
        value,
        name: str,
    ) -> np.ndarray:
        if value is None:
            raise RuntimeError(
                f"{name}=None 입니다. 실측 J1~J5 tick을 설정하세요."
            )

        return np.asarray(
            value,
            dtype=np.int64,
        ).reshape(5)

    # ------------------------------------------------------------------
    # Mission services
    # ------------------------------------------------------------------

    def handle_begin_seat(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """HAND 자세 이동 및 fresh detection."""
        del request

        try:
            with self.motion_lock:
                reached = self.move_to_initial_detection_view()

                self.get_logger().info(
                    "/arm/begin_seat -> HAND detection 시작"
                )

                hand = self.detect_and_save_hand(reached)

            response.success = True
            response.message = (
                "HAND detection ready | "
                f"distance={float(hand.distance_m):.3f}m | "
                f"ticks={np.asarray(reached, dtype=np.int64).tolist()}"
            )

            self.get_logger().info(
                "/arm/begin_seat 완료 -> HAND 검출 성공 | "
                f"distance={float(hand.distance_m):.3f}m | "
                "Task Manager 0.3m 접근 대기"
            )

        except Exception as error:
            self.get_logger().error(
                f"{ARM_BEGIN_SEAT_SERVICE} 실패: {error}"
            )
            response.success = False
            response.message = str(error)

        return response

    def handle_deliver_mic(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        try:
            # Manager가 SCOUT 이동/접근을 완료한 뒤 현재 hand 위치를 저장한다.
            # 완료한 뒤 호출한 시점의 hand를 저장하고 RIGHT basket mic를 전달한다.
            with self.motion_lock:
                self.deliver_microphone()

            response.success = True
            response.message = (
                "hand saved -> right basket mic delivered -> 3 sec wait -> "
                "released -> initial pose -> table mic search pose"
            )

        except Exception as error:
            self.get_logger().error(
                f"{ARM_DELIVER_SERVICE} 실패: {error}"
            )
            response.success = False
            response.message = str(error)

        return response

    def handle_return_mic(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        try:
            # TABLE_MIC_DETECTION 자세에서 새 mic detection을 사용해 회수한다.
            with self.motion_lock:
                self.return_microphone()

            response.success = True
            response.message = (
                "table mic detected -> picked -> right basket placed"
            )

        except Exception as error:
            self.get_logger().error(
                f"{ARM_RETURN_SERVICE} 실패: {error}"
            )
            response.success = False
            response.message = str(error)

        return response

    def handle_home(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        with self.motion_lock:
            try:
                reached = self.move_home()
                response.success = True
                response.message = f"home reached: {reached.tolist()}"

            except Exception as error:
                response.success = False
                response.message = str(error)

        return response


def main(args=None) -> None:
    rclpy.init(args=args)

    node = Mission2ArmControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
