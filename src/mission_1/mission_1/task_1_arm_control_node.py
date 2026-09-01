#!/usr/bin/env python3
"""Mission1 Arm Control: perception -> IK/trajectory -> motor ticks -> motor service.

No Dynamixel SDK and no direct serial I/O are allowed in this node.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.optimize import least_squares
from std_srvs.srv import Trigger
from soomac_interfaces.msg import Detection, DetectionArray
from soomac_interfaces.srv import PickPlace, ExecutePickPlace, MoveToTicks

from control_config.robot_config import *
from control_config.task_1_config import *


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
    """목표 방향 기준 radial/opposite branch seed를 만든다."""
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
            seeds.append(
                np.array([
                    q1_deg,
                    q2_deg,
                    q3_deg,
                    q4_deg,
                    yaw_consistent_q5_deg(q1_deg, target_yaw_deg),
                ], dtype=float)
            )

    return seeds

def evaluate_solver_state(
    chain,
    q_rad: np.ndarray,
) -> tuple[dict, np.ndarray, dict]:
    """Solver 해의 FK 상태, model joint(deg), joint-limit 정보를 한 번에 계산한다."""
    q_rad = np.asarray(q_rad, dtype=float).reshape(5)
    state = fk_state(chain, q_rad)
    q_deg = np.asarray(state["q_model_deg"], dtype=float).reshape(5)
    limit_info = joint_limit_info(q_deg)
    return state, q_deg, limit_info


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
        state, q_deg, limit_info = evaluate_solver_state(
            robot_chain,
            q_rad,
        )
        position_error = float(
            np.linalg.norm((state["ee_position_m"] - target_position_m) * 1000.0)
        )
        tool_error = float(state["tool_error"])
        yaw_error = abs(
            calculate_yaw_error_deg(state["yaw"], target_yaw_deg)
        )

        return SolverResult(
            success=bool(
                position_error <= POSITION_TOLERANCE_MM
                and tool_error <= TOOL_DOWN_TOLERANCE_DEG
                and yaw_error <= YAW_TOLERANCE_DEG
                and limit_info["inside"]
            ),
            q=q_deg,
            position_error=position_error,
            tool_error=tool_error,
            yaw_error=yaw_error,
            minimum_margin_deg=limit_info["minimum_margin_deg"],
        )

    except Exception:
        return SolverResult(
            success=False,
            q=np.full(5, np.nan),
            position_error=float("inf"),
            tool_error=float("inf"),
            yaw_error=float("inf"),
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
    actual_start_seed[4] = yaw_consistent_q5_deg(
        actual_start_seed[0], target_yaw_deg
    )
    actual_start_seed = np.clip(
        actual_start_seed, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG
    )

    seed_list = [
        actual_start_seed,
        *create_seed_list_deg(target_xyz_mm, target_yaw_deg),
    ]
    target_yaw_axis = create_target_yaw_axis(target_yaw_deg)
    target_position_m = target_xyz_mm / 1000.0

    results = [
        solve_single_seed(
            robot_chain=robot_chain,
            target_position_m=target_position_m,
            target_yaw_deg=target_yaw_deg,
            target_yaw_axis=target_yaw_axis,
            seed_q_deg=seed_q_deg,
        )
        for seed_q_deg in seed_list
    ]

    return results

def fk_state(
    chain,
    q_rad: np.ndarray,
) -> dict:
    """FK 위치와 tool/yaw 상태를 계산한다."""
    q_rad = np.asarray(
        q_rad,
        dtype=float,
    ).reshape(5)

    ikpy_vector = model_q_to_ikpy_vector(q_rad)

    transform = np.asarray(
        chain.forward_kinematics(ikpy_vector),
        dtype=float,
    )

    rotation = transform[:3, :3]
    xyz_m = transform[:3, 3]
    xyz_mm = xyz_m * 1000.0

    tool_state = calculate_tool_yaw_state(rotation)

    tool_axis = tool_state["tool_axis"]
    heading_axis = tool_state["heading_axis"]
    tool_error = float(
        tool_state["tool_down_error_deg"]
    )
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
        posture_error = (
            POSITION_POSTURE_WEIGHT * np.rad2deg(q_rad - previous_q_rad)
        )
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

        state, q_deg, limit_info = evaluate_solver_state(
            chain,
            result.x,
        )
        error = float(
            np.linalg.norm(state["xyz"] - target_xyz)
        )
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

def solve_waypoint(
    chain,
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> SolverResult:
    mode = waypoint["mode"]

    if mode == POSITION_ONLY:
        return solve_position(chain, waypoint, previous_q_deg)
    if mode != FULL_POSE:
        raise ValueError(f"지원하지 않는 constraint mode: {mode}")

    direct = solve_full_pose_single_seed(
        chain=chain,
        waypoint=waypoint,
        seed_q_deg=previous_q_deg,
        reference_q_deg=previous_q_deg,
    )

    # PLACE_ABOVE는 local minimum 하나에 바로 고정하지 않고,
    # position-only 해와 여러 elbow seed를 같이 비교한다.
    if waypoint.get("name", "") == "PLACE_ABOVE":
        candidates = [direct]

        # XYZ만 먼저 맞춘 해를 FULL_POSE seed로 다시 사용한다.
        # 하드웨어적으로 가능한데 FULL_POSE가 관절 한계 branch로 몰리는 경우를 줄인다.
        position_seed = solve_position(
            chain=chain,
            waypoint=waypoint,
            previous_q_deg=previous_q_deg,
        )
        if np.all(np.isfinite(position_seed.q)):
            candidates.append(
                solve_full_pose_single_seed(
                    chain=chain,
                    waypoint=waypoint,
                    seed_q_deg=position_seed.q,
                    reference_q_deg=previous_q_deg,
                )
            )

        for seed_q_deg in make_place_above_fast_seeds(waypoint, previous_q_deg):
            candidates.append(
                solve_full_pose_single_seed(
                    chain=chain,
                    waypoint=waypoint,
                    seed_q_deg=seed_q_deg,
                    reference_q_deg=previous_q_deg,
                )
            )

        successful = [item for item in candidates if item.success]
        if successful:
            # 허용오차를 만족하는 해 중 관절 한계에서 가장 여유 있는 branch를 우선한다.
            return max(
                successful,
                key=lambda item: (
                    item.minimum_margin_deg,
                    -item.position_error,
                    -item.tool_error,
                    -item.yaw_error,
                ),
            )

        return min(
            candidates,
            key=lambda item: (
                item.position_error,
                item.tool_error,
                item.yaw_error,
                -item.minimum_margin_deg,
            ),
        )

    return direct

def append_j5_continuous_result(
    solved: list[dict],
    waypoint: dict,
    previous_q_deg: np.ndarray,
    solver_q_deg: np.ndarray,
) -> np.ndarray | None:
    """J5 동치 branch를 맞추고 큰 J5 이동은 waypoint로 분할한다."""
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

    position_error = (
        state["xyz"] - np.asarray(target_xyz_mm, dtype=float)
    ) / 10.0

    tool_error = 10.0 * (
        state["tool_axis"] - WORLD_DOWN
    )

    target_yaw_rad = np.deg2rad(target_yaw_deg)
    target_heading = np.array([
        np.cos(target_yaw_rad),
        np.sin(target_yaw_rad),
        0.0,
    ], dtype=float)

    yaw_error = 10.0 * (
        state["heading_axis"] - target_heading
    )

    posture_error = FAST_FULL_POSE_POSTURE_WEIGHT * np.rad2deg(
        q_rad - reference_q_rad
    )

    return np.r_[
        position_error,
        tool_error,
        yaw_error,
        posture_error,
    ]

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

    state, q_deg, limit_info = evaluate_solver_state(
        chain,
        result.x,
    )
    position_error = float(
        np.linalg.norm(state["xyz"] - target_xyz)
    )
    yaw_error = abs(
        calculate_yaw_error_deg(state["yaw"], target_yaw)
    )
    tool_error = float(state["tool_error"])

    return SolverResult(
        success=bool(
            position_error <= FAST_FULL_POSE_POSITION_TOLERANCE_MM
            and tool_error <= FAST_FULL_POSE_TOOL_TOLERANCE_DEG
            and yaw_error <= FAST_FULL_POSE_YAW_TOLERANCE_DEG
            and limit_info["inside"]
        ),
        q=q_deg,
        position_error=position_error,
        tool_error=tool_error,
        yaw_error=yaw_error,
        minimum_margin_deg=limit_info["minimum_margin_deg"],
    )

def make_place_above_fast_seeds(
    waypoint: dict,
    previous_q_deg: np.ndarray,
) -> list[np.ndarray]:
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float)
    target_yaw = float(waypoint["yaw"])

    radial_q1 = wrap_to_180_deg(
        np.rad2deg(np.arctan2(target_xyz[1], target_xyz[0]))
    )
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
                JOINT_LIMIT_LOWER_DEG[0],
                JOINT_LIMIT_UPPER_DEG[0],
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
                seed = np.clip(
                    seed,
                    JOINT_LIMIT_LOWER_DEG,
                    JOINT_LIMIT_UPPER_DEG,
                )

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
                    fk_state(chain, np.deg2rad(solved_q))["xyz"],
                    dtype=float,
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
        pick_above_z = max(
            pick_xyz[2] + BOTTLE_PICK_CLEARANCE_MM,
            BOTTLE_PICK_ABOVE_MIN_Z_MM,
        )
        place_above_z = (
            place_xyz[2]
            + PLACE_APPROACH_HEIGHT_MM
            + BOTTLE_PLACE_APPROACH_OFFSET_MM
        )
    else:
        pick_above_z = max(pick_xyz[2] + 100.0, SAFE_ABOVE_Z_MM)
        place_above_z = place_xyz[2] + PLACE_APPROACH_HEIGHT_MM

    pick_above = np.array([pick_xyz[0], pick_xyz[1], pick_above_z])
    place_above = np.array([place_xyz[0], place_xyz[1], place_above_z])

    specs = (
        ("START", start_xyz, start_yaw, POSITION_ONLY),
        ("PICK_ABOVE", pick_above, pick_yaw, FULL_POSE),
        ("PICK", pick_xyz, pick_yaw, FULL_POSE),
        ("PICK_RETURN", pick_above, pick_yaw, FULL_POSE),
        ("PLACE_ABOVE", place_above, place_yaw, FULL_POSE),
        ("PLACE", place_xyz, place_yaw, FULL_POSE),
        ("PLACE_RETURN", place_above, place_yaw, FULL_POSE),
    )

    return [
        {
            "name": name,
            "xyz": xyz,
            "yaw": yaw,
            "mode": mode,
            "target_class_name": target_class_name,
        }
        for name, xyz, yaw, mode in specs
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
        bottle_transport = (
            transport and start.get("target_class_name") == "Bottle"
        )
        yaw_delta = wrap_to_180_deg(goal["yaw"] - start["yaw"])

        if bottle_transport:
            # Bottle도 일반 물체 transport 높이 계산을 기준으로 사용하고,
            # 운반 구간에만 Bottle 전용 offset을 추가한다.
            p0 = np.asarray(start["xyz"], dtype=float)
            p_goal = np.asarray(goal["xyz"], dtype=float)

            base_transport_z = max(
                float(p0[2]),
                float(p_goal[2]),
                float(BEZIER_CONTROL_Z_MM),
            )
            safe_z = base_transport_z + float(BOTTLE_TRANSPORT_OFFSET_MM)

            points = (
                p0,
                np.array([p0[0], p0[1], safe_z], dtype=float),
                np.array([p_goal[0], p_goal[1], safe_z], dtype=float),
                p_goal,
            )

            for segment_index, (segment_start, segment_goal) in enumerate(
                zip(points[:-1], points[1:])
            ):
                distance = float(np.linalg.norm(segment_goal - segment_start))
                if distance <= 1.0e-9:
                    continue

                count = max(int(np.ceil(distance / CARTESIAN_STEP_MM)), 1)
                last_segment = segment_index == len(points) - 2

                for index in range(1, count + 1):
                    alpha = index / count
                    is_final = last_segment and index == count
                    waypoints.append({
                        "name": goal["name"] if is_final else "",
                        "xyz": (
                            (1.0 - alpha) * segment_start
                            + alpha * segment_goal
                        ),
                        "yaw": goal["yaw"] if is_final else start["yaw"],
                        "mode": goal["mode"] if is_final else POSITION_ONLY,
                        "target_class_name": start.get("target_class_name"),
                    })
            continue

        control = (
            bezier_control(start["xyz"], goal["xyz"])
            if transport else None
        )
        length = (
            np.linalg.norm(control - start["xyz"])
            + np.linalg.norm(goal["xyz"] - control)
            if control is not None
            else np.linalg.norm(goal["xyz"] - start["xyz"])
        )
        count = max(
            int(np.ceil(length / CARTESIAN_STEP_MM)),
            0 if transport else int(np.ceil(abs(yaw_delta) / YAW_STEP_DEG)),
            1,
        )

        for index in range(1, count + 1):
            alpha = index / count
            xyz = (
                (1.0 - alpha) ** 2 * start["xyz"]
                + 2.0 * (1.0 - alpha) * alpha * control
                + alpha**2 * goal["xyz"]
                if control is not None
                else (1.0 - alpha) * start["xyz"] + alpha * goal["xyz"]
            )

            if transport and index < count:
                yaw, mode = start["yaw"], POSITION_ONLY
            else:
                yaw = (
                    goal["yaw"]
                    if transport
                    else wrap_to_180_deg(start["yaw"] + alpha * yaw_delta)
                )
                mode = goal["mode"]

            waypoints.append({
                "name": goal["name"] if index == count else "",
                "xyz": xyz,
                "yaw": yaw,
                "mode": mode,
                "target_class_name": start.get("target_class_name"),
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
        ASSUMED_START_TICKS if start_ticks is None else start_ticks,
        dtype=np.int64,
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
        raise RuntimeError(
            "FAST planner 실패 | PICK_ABOVE 성공 candidate 없음"
        )

    path_waypoints = interpolate(keypoints[1:])
    failures = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        # 전체 경로를 풀기 전에 PICK_ABOVE의 J5 동치 branch를 확정한다.
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

        solved, failure = solve_sequence(
            chain,
            path_waypoints,
            candidate_q,
        )
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

    raise RuntimeError(
        "FAST planner 실패 | " + " | ".join(failures)
    )


def path_to_ticks(
    path: list[dict],
    expected_start_ticks: np.ndarray,
) -> list[np.ndarray]:
    expected = np.asarray(
        expected_start_ticks,
        dtype=np.int64,
    ).reshape(5)

    ticks_path = []

    for index, item in enumerate(path):
        q_deg = np.asarray(
            item["q_deg"],
            dtype=float,
        ).reshape(5)

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


_ACTIVE_NODE = None


def nearest_gripper_symmetric_yaw_deg(
    target_yaw_deg: float,
    reference_yaw_deg: float,
) -> float:
    """
    평행 그리퍼는 yaw와 yaw+180 deg가 동일한 파지 방향이다.
    현재 gripper yaw(reference)에서 가장 적게 회전하는
    180 deg 대칭 yaw를 선택한다.
    """
    delta = (
        (
            float(target_yaw_deg)
            - float(reference_yaw_deg)
            + 90.0
        )
        % 180.0
        - 90.0
    )

    return wrap_to_180_deg(
        float(reference_yaw_deg)
        + delta
    )


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

def point_to_numpy(point):
    return np.array(
        [
            float(point.x),
            float(point.y),
            float(point.z),
        ],
        dtype=float,
    )

class ArmControlNode(Node):
    def __init__(self) -> None:
        global _ACTIVE_NODE
        super().__init__("arm_control_node")
        _ACTIVE_NODE = self

        self._emergency_torque_off_done = False
        self.callback_group = ReentrantCallbackGroup()
        self.motion_lock = threading.RLock()
        self.detection_condition = threading.Condition(threading.RLock())

        for name, default in (
            ("detections_topic", "/mission1/detections"),
            ("handeye_path", DEFAULT_HANDEYE_PATH),
            ("detection_wait_sec_per_pose", DETECTION_WAIT_SEC_PER_POSE),
            ("floor_z_offset_m", FLOOR_Z_OFFSET_M),
        ):
            self.declare_parameter(name, default)

        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.detection_wait_sec_per_pose = float(
            self.get_parameter("detection_wait_sec_per_pose").value
        )
        self.floor_z_offset_m = float(self.get_parameter("floor_z_offset_m").value)
        self.t_flange_camera = load_t_flange_camera(
            str(self.get_parameter("handeye_path").value)
        )
        self.chain = create_robot_chain()

        self.latest_detection_msg: DetectionArray | None = None
        self.latest_detection_stamp_ns = 0
        self.pick_sequence_index = 0
        self.aruco_floor_m: np.ndarray | None = None

        self.auto_sequence_lock = threading.Lock()
        self.auto_sequence_active = False

        self.move_ticks_client = self.create_client(
            MoveToTicks, "/motor/move_to_ticks", callback_group=self.callback_group
        )
        self.execute_client = self.create_client(
            ExecutePickPlace, "/arm/execute_pick_place", callback_group=self.callback_group
        )
        self.torque_off_client = self.create_client(
            Trigger, "/motor/torque_off", callback_group=self.callback_group
        )
        self.start_detect_client = self.create_client(
            Trigger, "/mission1/start_detect", callback_group=self.callback_group
        )
        self.lift_down_client = self.create_client(
            Trigger, "/lift/down", callback_group=self.callback_group
        )

        self.subscription = self.create_subscription(
            DetectionArray,
            self.detections_topic,
            self.on_detections,
            10,
            callback_group=self.callback_group,
        )
        self.begin_seat_service = self.create_service(
            Trigger, "/arm/begin_seat", self.handle_begin_seat,
            callback_group=self.callback_group,
        )
        self.pick_place_service = self.create_service(
            PickPlace, "/arm/pick_place", self.handle_pick_place,
            callback_group=self.callback_group,
        )
        self.overview_service = self.create_service(
            Trigger, "/arm/move_to_overview_pose", self.handle_overview,
            callback_group=self.callback_group,
        )
        self.home_service = self.create_service(
            Trigger, "/arm/home", self.handle_home,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Arm Control Node ready")

    # Lifecycle

    def emergency_torque_off(self) -> None:
        if getattr(self, "_emergency_torque_off_done", False):
            return
        self._emergency_torque_off_done = True
        try:
            if self.torque_off_client.wait_for_service(timeout_sec=0.5):
                future = self.torque_off_client.call_async(Trigger.Request())
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not future.done():
                    time.sleep(0.01)
        except Exception as error:
            self.get_logger().error(f"motor torque-off request failed: {error}")
    def destroy_node(self):
        self.emergency_torque_off()
        return super().destroy_node()
    # Detection

    def on_detections(self, msg: DetectionArray) -> None:
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.detection_condition:
            if stamp_ns >= self.latest_detection_stamp_ns:
                self.latest_detection_msg = deepcopy(msg)
                self.latest_detection_stamp_ns = stamp_ns
                self.detection_condition.notify_all()

    def wait_for_detection_message(
        self,
        gate_stamp_ns: int,
        timeout_sec: float,
        validator,
    ) -> DetectionArray | None:
        """gate 이후 새 DetectionArray 중 validator를 만족하는 메시지를 기다린다."""
        deadline = time.monotonic() + float(timeout_sec)

        with self.detection_condition:
            while time.monotonic() < deadline:
                msg = self.latest_detection_msg
                stamp_ns = self.latest_detection_stamp_ns

                if msg is not None and stamp_ns > gate_stamp_ns:
                    valid_frame = (
                        not REQUIRE_FLANGE_FRAME
                        or msg.header.frame_id == "flange"
                    )

                    if not valid_frame:
                        self.get_logger().error(
                            f"Detection frame='{msg.header.frame_id}'. "
                            "flange frame이 아니므로 사용하지 않습니다."
                        )
                    elif validator(msg):
                        return deepcopy(msg)

                remaining = deadline - time.monotonic()

                if remaining <= 0.0:
                    break

                self.detection_condition.wait(
                    timeout=min(remaining, 0.10)
                )

        return None

    def wait_for_target_after(
        self,
        target_class_name: str,
        gate_stamp_ns: int,
        timeout_sec: float,
    ) -> Detection | None:
        def has_target(msg: DetectionArray) -> bool:
            return any(
                detection.class_name == target_class_name
                for detection in msg.detections
            )

        msg = self.wait_for_detection_message(
            gate_stamp_ns=gate_stamp_ns,
            timeout_sec=timeout_sec,
            validator=has_target,
        )

        if msg is None:
            return None

        candidates = [
            detection
            for detection in msg.detections
            if detection.class_name == target_class_name
        ]

        if target_class_name == "name_tag":
            distance_candidates = [
                detection
                for detection in candidates
                if np.isfinite(float(detection.distance_m))
                and float(detection.distance_m) > 0.0
            ]

            if distance_candidates:
                selected = min(
                    distance_candidates,
                    key=lambda detection: float(detection.distance_m),
                )
            else:
                selected = max(
                    candidates,
                    key=lambda detection: float(detection.confidence),
                )
        else:
            selected = max(
                candidates,
                key=lambda detection: float(detection.confidence),
            )

        return deepcopy(selected)

    # Target sequence

    def resolve_target_class_name(self, category: str) -> str:
        category = str(category)

        if category == Detection.CATEGORY_SNACK:
            if self.pick_sequence_index not in (1, 2, 3):
                raise RuntimeError(
                    "현재 Pick 순서에서는 snack 요청을 처리할 수 없습니다. "
                    f"sequence_index={self.pick_sequence_index}"
                )

            return PICK_SEQUENCE[self.pick_sequence_index]

        category_map = {
            Detection.CATEGORY_NAME_TAG: "name_tag",
            Detection.CATEGORY_BOTTLE: "Bottle",
        }

        if category not in category_map:
            raise ValueError(
                f"지원하지 않는 category: {category!r}"
            )

        return category_map[category]

    def validate_pick_sequence(self, target_class_name: str) -> None:
        if self.pick_sequence_index >= len(PICK_SEQUENCE):
            raise RuntimeError(
                "현재 seat의 모든 Pick & Place 순서가 이미 완료되었습니다."
            )

        expected = PICK_SEQUENCE[self.pick_sequence_index]
        if target_class_name != expected:
            raise RuntimeError(
                "Pick & Place 요청 순서가 다릅니다. "
                f"expected={expected}, received={target_class_name}, "
                f"index={self.pick_sequence_index}"
            )

    def mark_request_completed(self) -> None:
        self.pick_sequence_index = min(
            self.pick_sequence_index + 1,
            len(PICK_SEQUENCE),
        )

    # Direct motion

    def wait_service_future(self, future, timeout_sec: float, label: str):
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"{label} timeout")
        response = future.result()
        if response is None:
            raise RuntimeError(f"{label}: no response")
        return response

    def move_to_ticks(self, goal_ticks: np.ndarray, label: str) -> np.ndarray:
        if not self.move_ticks_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/motor/move_to_ticks service is not available")

        request = MoveToTicks.Request()
        request.goal_ticks = [
            int(value)
            for value in np.asarray(goal_ticks, dtype=np.int64).reshape(5)
        ]
        request.label = str(label)
        request.profile_velocity = int(SCAN_PROFILE_VELOCITY)
        request.timeout_sec = float(SCAN_TIMEOUT_SEC)

        response = self.wait_service_future(
            self.move_ticks_client.call_async(request),
            float(SCAN_TIMEOUT_SEC) + 5.0,
            f"{label}: motor move service",
        )
        if not response.success:
            raise RuntimeError(response.message)

        return np.asarray(response.reached_ticks, dtype=np.int64).reshape(5)

    def move_to_scan_pose(self, pose_name: str) -> np.ndarray:
        if pose_name not in SCAN_POSES:
            raise ValueError(f"알 수 없는 scan pose: {pose_name}")
        return self.move_to_ticks(SCAN_POSES[pose_name], f"SCAN_{pose_name}")

    def move_home(self) -> np.ndarray:
        return self.move_to_ticks(
            np.asarray(ASSUMED_START_TICKS, dtype=np.int64),
            "PLANNER_HOME",
        )

    def acquire_aruco_at_center(
        self,
        timeout_sec: float,
        *,
        move_to_center: bool = True,
    ) -> bool:
        """CENTER 자세에서 새 ArUco를 받아 floor 좌표 cache를 갱신한다."""
        if move_to_center:
            center_ticks = self.move_to_scan_pose(
                ARUCO_INITIAL_SCAN_POSE
            )
        else:
            center_ticks = np.asarray(
                SCAN_POSES[ARUCO_INITIAL_SCAN_POSE], dtype=np.int64
            ).copy()

        time.sleep(DETECTION_SETTLE_SEC)

        # 반드시 CENTER 정지 이후에 발행된 새 메시지만 사용한다.
        gate_stamp_ns = self.get_clock().now().nanoseconds

        aruco_msg = self.wait_for_detection_message(
            gate_stamp_ns=gate_stamp_ns,
            timeout_sec=float(timeout_sec),
            validator=lambda msg: (
                bool(getattr(msg, "aruco_detected", False))
                and not bool(getattr(msg, "aruco_is_temp", False))
            ),
        )

        if aruco_msg is None:
            return False

        self.update_aruco_cache(
            aruco_msg,
            center_ticks,
        )

        return self.aruco_floor_m is not None

    # Coordinate conversion

    def base_transform_from_ticks(self, ticks: np.ndarray) -> np.ndarray:
        q_model_rad = ticks_to_model_rad(
            np.asarray(ticks, dtype=float).reshape(5)
        )
        return np.asarray(
            self.chain.forward_kinematics(
                model_q_to_ikpy_vector(q_model_rad)
            ),
            dtype=float,
        )

    def flange_point_to_floor_xyz_m(self, point, scan_ticks: np.ndarray) -> np.ndarray:
        """flange Point를 floor 기준 XYZ[m]로 변환한다."""
        point_h = np.ones(4, dtype=float)
        point_h[:3] = point_to_numpy(point)
        base_xyz_m = (
            self.base_transform_from_ticks(scan_ticks) @ point_h
        )[:3]
        return base_xyz_m + np.array(
            [0.0, 0.0, self.floor_z_offset_m],
            dtype=float,
        )

    def camera_yaw_to_floor_yaw_deg(
        self,
        detection: Detection,
        scan_ticks: np.ndarray,
    ) -> float:
        if not bool(getattr(detection, "yaw_valid", False)):
            return DEFAULT_PICK_YAW_DEG.get(detection.class_name, 0.0)

        camera_yaw_deg = float(getattr(detection, "yaw_deg", 0.0))
        if not np.isfinite(camera_yaw_deg):
            return DEFAULT_PICK_YAW_DEG.get(detection.class_name, 0.0)

        t_base_flange = self.base_transform_from_ticks(scan_ticks)
        r_base_camera = (
            t_base_flange[:3, :3] @ self.t_flange_camera[:3, :3]
        )

        yaw_rad = np.deg2rad(camera_yaw_deg)
        heading_camera = np.array(
            [np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=float
        )
        heading_base = r_base_camera @ heading_camera

        if np.linalg.norm(heading_base[:2]) < 1.0e-9:
            return DEFAULT_PICK_YAW_DEG.get(detection.class_name, 0.0)

        return wrap_to_180_deg(
            np.rad2deg(np.arctan2(heading_base[1], heading_base[0]))
        )

    def update_aruco_cache(
        self,
        msg: DetectionArray,
        scan_ticks: np.ndarray,
    ) -> None:
        if not bool(getattr(msg, "aruco_detected", False)):
            return
        if bool(getattr(msg, "aruco_is_temp", False)):
            return

        self.aruco_floor_m = self.flange_point_to_floor_xyz_m(
            msg.aruco_pose,
            scan_ticks,
        )


    # Search

    def search_target(
        self,
        target_class_name: str,
    ) -> tuple[Detection | None, np.ndarray | None]:
        """물체별 detection pose로 이동해 새 target과 실제 도달 tick을 반환한다."""
        if target_class_name not in OBJECT_DETECTION_POSES:
            raise ValueError(
                f"등록되지 않은 object detection pose: {target_class_name}"
            )

        scan_ticks = self.move_to_ticks(
            np.asarray(
                OBJECT_DETECTION_POSES[target_class_name],
                dtype=np.int64,
            ).reshape(5),
            f"DETECTION_{target_class_name}",
        )

        time.sleep(DETECTION_SETTLE_SEC)
        gate_stamp_ns = self.get_clock().now().nanoseconds

        detection = self.wait_for_target_after(
            target_class_name=target_class_name,
            gate_stamp_ns=gate_stamp_ns,
            timeout_sec=self.detection_wait_sec_per_pose,
        )

        if detection is None:
            self.get_logger().warn(
                f"DETECTION_{target_class_name}: {target_class_name} 미검출"
            )
            return None, None

        return detection, np.asarray(
            scan_ticks,
            dtype=np.int64,
        ).copy()

    def calculate_pick_target(
        self,
        detection: Detection,
        scan_ticks: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """검출 결과와 scan pose에서 최종 pick floor XYZ[m]/yaw[deg]를 계산한다."""
        target_class_name = detection.class_name

        pick_floor_m = self.flange_point_to_floor_xyz_m(
            detection.pose,
            scan_ticks,
        )
        pick_floor_m[2] += (
            TOOL_HORN_TO_GRASP_M
            - float(
                PICK_GRASP_DEPTH_FROM_TOP_M.get(
                    target_class_name,
                    0.0,
                )
            )
        )
        pick_floor_m += np.asarray(
            PICK_FINE_OFFSET_M.get(
                target_class_name,
                np.zeros(3, dtype=float),
            ),
            dtype=float,
        )

        raw_pick_yaw_deg = self.camera_yaw_to_floor_yaw_deg(
            detection,
            scan_ticks,
        )

        if target_class_name == "name_tag":
            return pick_floor_m, raw_pick_yaw_deg

        start_state = fk_state(
            self.chain,
            ticks_to_model_rad(
                np.asarray(scan_ticks, dtype=float).reshape(5)
            ),
        )
        pick_yaw_deg = nearest_gripper_symmetric_yaw_deg(
            raw_pick_yaw_deg,
            float(start_state["yaw"]),
        )

        return pick_floor_m, pick_yaw_deg


    # Place / execution

    def calculate_place_target(
        self,
        target_class_name: str,
        place_offset_m: np.ndarray,
        place_mode: int,
        place_yaw_deg: float,
    ) -> tuple[np.ndarray, float]:
        place_offset_m = np.asarray(place_offset_m, dtype=float).reshape(3)

        if int(place_mode) == PickPlace.Request.PLACE_ARUCO_OFFSET:
            if self.aruco_floor_m is None:
                raise RuntimeError(
                    "실제 ArUco world/floor 좌표가 아직 캐시되지 않았습니다."
                )
            place_floor_m = self.aruco_floor_m + place_offset_m
        else:
            place_floor_m = place_offset_m.copy()

        place_floor_m[2] += (
            float(
                PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M.get(
                    target_class_name,
                    0.0,
                )
            )
            + TOOL_HORN_TO_GRASP_M
        )
        place_floor_m += PLACE_FINE_OFFSET_M.get(
            target_class_name,
            np.zeros(3, dtype=float),
        )

        return (
            place_floor_m * 1000.0,
            NAME_TAG_PLACE_YAW_DEG
            if target_class_name == "name_tag"
            else float(place_yaw_deg),
        )

    def execute_pick_place(
        self,
        pick_xyz_mm: np.ndarray,
        place_xyz_mm: np.ndarray,
        pick_yaw_deg: float,
        place_yaw_deg: float,
        start_ticks: np.ndarray,
        target_class_name: str,
    ) -> str:
        start_ticks = np.asarray(start_ticks, dtype=np.int64).reshape(5)
        path = plan(
            self.chain,
            np.asarray(pick_xyz_mm, dtype=float),
            np.asarray(place_xyz_mm, dtype=float),
            float(pick_yaw_deg),
            float(place_yaw_deg),
            start_ticks=start_ticks,
            target_class_name=target_class_name,
        )
        ticks_path = path_to_ticks(path, start_ticks)
        profile = GRIPPER_POSITION_PROFILE.get(
            target_class_name,
            DEFAULT_GRIPPER_POSITION_PROFILE,
        )

        if not self.execute_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/arm/execute_pick_place service is not available")

        request = ExecutePickPlace.Request()
        request.joint_ticks = [
            int(value)
            for row in ticks_path
            for value in np.asarray(row, dtype=np.int64).reshape(5)
        ]
        request.waypoint_names = [
            str(item["waypoint"].get("name", ""))
            for item in path
        ]
        request.point_count = len(ticks_path)
        request.gripper_open_tick = int(profile["open_tick"])
        request.gripper_close_tick = int(profile["close_tick"])

        response = self.wait_service_future(
            self.execute_client.call_async(request),
            180.0,
            "motor execute service",
        )
        if not response.success:
            raise RuntimeError(str(response.message))

        return str(response.message)

    def notify_task_complete(self) -> bool:
        """
        Mission1 한 자리 작업 완료 신호.

        기존 PickPlace 응답의 task_complete=True는 내부 자동 호출에서는
        외부 노드가 받을 수 없으므로, Arm이 기존 Lift의 /lift/down 서비스를
        직접 호출해서 '팔 작업 완료 -> 리프트 하강' 흐름을 이어간다.
        """
        if not self.lift_down_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Mission1 완료 신호 실패: /lift/down service is not available"
            )
            return False

        response = self.wait_service_future(
            self.lift_down_client.call_async(Trigger.Request()),
            30.0,
            "lift down service",
        )
        if not response.success:
            self.get_logger().error(
                f"Mission1 완료 신호 실패: {response.message}"
            )
            return False

        self.get_logger().info(
            f"Mission1 완료 신호 전송 완료: /lift/down | {response.message}"
        )
        return True

    def auto_pick_place_sequence(self) -> None:
        """begin_seat 이후 task_1_config의 자동 배치 설정으로 Mission1을 실행한다."""
        try:
            self.get_logger().info(
                "AUTO SEQUENCE: Mission1 전체 Pick & Place 자동 시작"
            )

            for index, target_class_name in enumerate(PICK_SEQUENCE, start=1):
                config = AUTO_PLACE_CONFIG[target_class_name]
                offset = np.asarray(config["offset_m"], dtype=float).reshape(3)

                for attempt in range(1, AUTO_MAX_ARM_RETRY + 1):
                    self.get_logger().info(
                        f"AUTO SEQUENCE: {target_class_name} "
                        f"({index}/{len(PICK_SEQUENCE)}) "
                        f"시도 {attempt}/{AUTO_MAX_ARM_RETRY} | "
                        f"place_offset={offset.tolist()}"
                    )

                    success, _, message, detection = self.execute_target(
                        target_class_name=target_class_name,
                        place_offset_m=offset,
                        place_mode=PickPlace.Request.PLACE_ARUCO_OFFSET,
                        place_yaw_deg=float(config["place_yaw_deg"]),
                    )

                    if success:
                        self.get_logger().info(
                            f"AUTO SEQUENCE: {target_class_name} 완료"
                        )
                        break

                    if detection is None:
                        self.get_logger().error(
                            f"AUTO SEQUENCE: {target_class_name} 미검출 -> 중단"
                        )
                        return

                    self.get_logger().warn(
                        f"AUTO SEQUENCE: {target_class_name} 실패 | "
                        f"message={message}"
                    )
                else:
                    self.get_logger().error(
                        f"AUTO SEQUENCE: {target_class_name} "
                        f"{AUTO_MAX_ARM_RETRY}회 실패 -> 중단"
                    )
                    return

            self.get_logger().info(
                "AUTO SEQUENCE: Mission1 전체 Pick & Place 완료"
            )

            # 마지막 Bottle은 execute_target()에서 HOME을 건너뛰므로
            # 전체 sequence 종료 시 한 번만 HOME 복귀한다.
            self.get_logger().info(
                "AUTO SEQUENCE: 전체 작업 완료 -> PLANNER_HOME 복귀"
            )
            self.move_home()
            self.get_logger().info(
                "AUTO SEQUENCE: PLANNER_HOME 복귀 완료"
            )

            if not self.notify_task_complete():
                self.get_logger().error(
                    "AUTO SEQUENCE: 작업은 완료했지만 Lift 완료 신호 전달 실패"
                )
                return

            self.get_logger().info(
                "AUTO SEQUENCE: Mission1 종료 처리 완료"
            )

        except Exception as error:
            self.get_logger().error(
                f"AUTO SEQUENCE 예외: {error}"
            )
        finally:
            with self.auto_sequence_lock:
                self.auto_sequence_active = False

    def handle_begin_seat(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        with self.auto_sequence_lock:
            if self.auto_sequence_active:
                response.success = True
                response.message = "Mission1 auto sequence already running"
                return response
            self.auto_sequence_active = True

        try:
            with self.motion_lock:
                self.pick_sequence_index = 0
                self.aruco_floor_m = None

                # Lift UP 완료 후 /arm/begin_seat 수신:
                # CENTER 이동 -> ArUco 선행 확보 -> 전체 Task1 자동 실행.
                self.move_to_scan_pose(ARUCO_INITIAL_SCAN_POSE)
                acquired = self.acquire_aruco_at_center(
                    ARUCO_BEGIN_SEAT_WAIT_SEC,
                    move_to_center=False,
                )

                if acquired:
                    response.message = (
                        "CENTER ready; ArUco acquired; "
                        "Mission1 auto sequence started"
                    )
                else:
                    response.message = (
                        "CENTER ready; ArUco pending; "
                        "Mission1 auto sequence started"
                    )
                    self.get_logger().warn(
                        "begin_seat: 아직 ArUco를 못 잡았습니다. "
                        "첫 자동 Pick & Place에서 CENTER ArUco를 다시 획득합니다."
                    )

            # 기존 Task Manager의 /mission1/start_detect 역할도 Arm이 수행한다.
            if self.start_detect_client.wait_for_service(timeout_sec=5.0):
                detect_response = self.wait_service_future(
                    self.start_detect_client.call_async(Trigger.Request()),
                    10.0,
                    "mission1 start_detect service",
                )
                if not detect_response.success:
                    self.get_logger().warn(
                        f"/mission1/start_detect failed: {detect_response.message}"
                    )
            else:
                self.get_logger().warn(
                    "/mission1/start_detect service is not available"
                )

            threading.Thread(
                target=self.auto_pick_place_sequence,
                daemon=True,
            ).start()

            response.success = True
            return response

        except Exception as error:
            with self.auto_sequence_lock:
                self.auto_sequence_active = False
            response.success = False
            response.message = str(error)
            self.get_logger().error(
                f"begin_seat 초기화/CENTER 이동 실패: {error}"
            )
            return response

    def handle_overview(self, request, response):
        del request
        try:
            with self.motion_lock:
                self.move_to_scan_pose(OVERVIEW_POSE_NAME)
            response.success = True
            response.message = f"overview pose={OVERVIEW_POSE_NAME} reached"
        except Exception as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f"overview 이동 실패: {error}")
        return response

    def handle_home(self, request, response):
        del request
        try:
            with self.motion_lock:
                self.move_home()
            response.success = True
            response.message = "planner home reached"
        except Exception as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f"home 이동 실패: {error}")
        return response

    def execute_target(
        self,
        target_class_name: str,
        place_offset_m: np.ndarray,
        place_mode: int = PickPlace.Request.PLACE_ARUCO_OFFSET,
        place_yaw_deg: float = 0.0,
    ) -> tuple[bool, bool, str, Detection | None]:
        """한 물체의 search -> target 계산 -> Pick & Place를 공통 실행한다."""
        self.get_logger().info(
            f"[PICK & PLACE] {target_class_name} 시작 | "
            f"sequence={self.pick_sequence_index + 1}/{len(PICK_SEQUENCE)}"
        )

        # Detection 성공 이후 planner/motor에서 실패해도
        # "미검출"로 오판하지 않도록 마지막 검출 결과를 보존한다.
        detection = None

        try:
            with self.motion_lock:
                if (
                    self.aruco_floor_m is None
                    and not self.acquire_aruco_at_center(
                        ARUCO_PICK_GATE_WAIT_SEC,
                        move_to_center=True,
                    )
                ):
                    raise RuntimeError(
                        "CENTER(J1=1950)에서 ArUco marker를 "
                        "획득하지 못했습니다. Pick & Place를 시작하지 않습니다."
                    )

                self.validate_pick_sequence(target_class_name)

                detection, detection_start_ticks = self.search_target(
                    target_class_name
                )
                if detection is None or detection_start_ticks is None:
                    return (
                        False,
                        False,
                        f"{target_class_name} not detected in detection pose",
                        None,
                    )

                pick_floor_m, pick_yaw_deg = self.calculate_pick_target(
                    detection,
                    detection_start_ticks,
                )
                place_xyz_mm, resolved_place_yaw_deg = self.calculate_place_target(
                    target_class_name=target_class_name,
                    place_offset_m=place_offset_m,
                    place_mode=place_mode,
                    place_yaw_deg=place_yaw_deg,
                )

                motor_message = self.execute_pick_place(
                    pick_xyz_mm=pick_floor_m * 1000.0,
                    place_xyz_mm=place_xyz_mm,
                    pick_yaw_deg=float(pick_yaw_deg),
                    place_yaw_deg=resolved_place_yaw_deg,
                    start_ticks=detection_start_ticks,
                    target_class_name=target_class_name,
                )

                task_complete = (
                    self.pick_sequence_index == len(PICK_SEQUENCE) - 1
                    and target_class_name == PICK_SEQUENCE[-1]
                )

                self.get_logger().info(
                    f"[PICK & PLACE] {target_class_name} 완료"
                )

                # 마지막 Bottle 전까지는 각 cycle 후 HOME 복귀.
                if not task_complete:
                    self.move_home()

                self.mark_request_completed()

                return True, task_complete, motor_message, detection

        except Exception as error:
            self.get_logger().error(
                f"{target_class_name} pick_place 처리 실패: {error}"
            )
            return False, False, str(error), detection

    def handle_pick_place(
        self,
        request: PickPlace.Request,
        response: PickPlace.Response,
    ) -> PickPlace.Response:
        try:
            target_class_name = self.resolve_target_class_name(
                str(request.category)
            )
        except Exception as error:
            response.success, response.task_complete = False, False
            response.nothing_detected = False
            response.message = str(error)
            return response

        success, task_complete, message, detection = self.execute_target(
            target_class_name=target_class_name,
            place_offset_m=point_to_numpy(request.place_offset),
            place_mode=int(request.place_mode),
            place_yaw_deg=float(request.place_yaw_deg),
        )

        response.success = success
        response.task_complete = task_complete
        response.nothing_detected = detection is None and not success
        if detection is not None:
            response.picked = detection
        response.message = message
        return response


def main(args=None) -> None:
    global _ACTIVE_NODE
    rclpy.init(args=args)
    node=None
    executor=MultiThreadedExecutor(num_threads=4)

    def emergency_signal_handler(signum, frame):
        del signum, frame
        active=_ACTIVE_NODE
        if active is not None:
            active.emergency_torque_off()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, emergency_signal_handler)
    signal.signal(signal.SIGTERM, emergency_signal_handler)

    try:
        node=ArmControlNode(); executor.add_node(node); executor.spin()
    except KeyboardInterrupt:
        if node is not None: node.emergency_torque_off()
    finally:
        executor.shutdown()
        if node is not None: node.destroy_node()
        _ACTIVE_NODE=None
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
