#!/usr/bin/env python3
from __future__ import annotations

import json
import signal
import threading
import time
from copy import deepcopy
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
from control_config.task_3_config import *


def xyz_tool_down_yaw_residual( q_model_rad: np.ndarray, robot_chain, target_position_m: np.ndarray,
    target_yaw_axis: np.ndarray,
) -> np.ndarray:
    state = fk_state(robot_chain, q_model_rad)
    position_error_mm = ( state["ee_position_m"] - target_position_m ) * 1000.0
    position_residual = ( position_error_mm / POSITION_RESIDUAL_SCALE_MM )
    tool_axis_error = ( state["tool_axis"] - WORLD_DOWN )
    tool_residual = ( TOOL_RESIDUAL_WEIGHT * tool_axis_error )
    yaw_axis_error = ( state["yaw_axis"] - target_yaw_axis )
    yaw_residual = ( YAW_RESIDUAL_WEIGHT * yaw_axis_error )

    return np.concatenate([
        position_residual, tool_residual, yaw_residual,
    ])


def create_seed_list_deg( target_position_mm: np.ndarray, target_yaw_deg: float,
) -> list[tuple[str, np.ndarray]]:
    x_mm, y_mm = np.asarray( target_position_mm, dtype=float, ).reshape(3)[:2]

    radial_q1_deg = wrap_to_180_deg( np.rad2deg(np.arctan2(y_mm, x_mm)) )
    opposite_q1_deg = wrap_to_180_deg( radial_q1_deg + 180.0 )

    start_seed = get_start_q_model_deg().copy()
    start_seed[4] = yaw_consistent_q5_deg( start_seed[0], target_yaw_deg, )

    previous_seed = PREVIOUS_POSITION_SOLUTION_DEG.copy()
    previous_seed[0] = radial_q1_deg
    previous_seed[4] = yaw_consistent_q5_deg( previous_seed[0], target_yaw_deg, )

    arm_profiles = ( ("negative branch A", 10.0, -105.0, -85.0),
        ("negative branch B", 30.0, -120.0, -90.0), ("positive branch A", -10.0, 95.0, 95.0),
        ("positive branch B", -30.0, 80.0, 130.0),
    )
    q1_branches = ( ("Radial", radial_q1_deg), ("Opposite", opposite_q1_deg), )

    seeds = [
        ("START yaw adjusted", start_seed), ("Previous negative branch", previous_seed),
    ]

    for branch_name, q1_deg in q1_branches:
        for profile_name, q2_deg, q3_deg, q4_deg in arm_profiles:
            seeds.append( (
                    f"{branch_name} {profile_name}",
                    np.array( [
                            q1_deg, q2_deg, q3_deg, q4_deg,
                            yaw_consistent_q5_deg( q1_deg, target_yaw_deg, ),
                        ], dtype=float,
                    ),
                )
            )

    return seeds


def solve_single_seed( robot_chain, target_position_m: np.ndarray, target_yaw_deg: float,
    target_yaw_axis: np.ndarray, seed_name: str, seed_q_deg: np.ndarray, seed_index: int,
) -> dict:
    seed_q_deg = np.asarray( seed_q_deg, dtype=float, ).reshape(5)

    clipped_seed_q_deg = np.clip( seed_q_deg, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG, )
    initial_q_rad = np.deg2rad(clipped_seed_q_deg)

    result = {
        "seed_index": seed_index,
        "seed_name": seed_name,
        "seed_q_deg": seed_q_deg,
        "clipped_seed_q_deg": clipped_seed_q_deg,
        "solver_exception": None,
        "optimization_result": None,
        "state": None,
        "position_error_vector_mm": None,
        "position_error_mm": float("inf"),
        "tool_down_error_deg": float("inf"),
        "target_yaw_deg": target_yaw_deg,
        "current_yaw_deg": float("nan"),
        "yaw_error_deg": float("inf"),
        "absolute_yaw_error_deg": float("inf"),
        "limit_info": None,
        "residual_norm": float("inf"),
        "success": False,
    }

    try:
        optimization_result = least_squares( fun=xyz_tool_down_yaw_residual,
            x0=initial_q_rad, bounds=( JOINT_LIMIT_LOWER_RAD, JOINT_LIMIT_UPPER_RAD,
            ), args=( robot_chain, target_position_m, target_yaw_axis,
            ), method="trf", max_nfev=MAX_FUNCTION_EVALUATIONS, xtol=XTOL,
            ftol=FTOL, gtol=GTOL, verbose=0,
        )

        solution_q_rad = np.asarray( optimization_result.x, dtype=float, ).reshape(5)

        state = fk_state( robot_chain, solution_q_rad, )

        position_error_vector_mm = ( state["ee_position_m"] - target_position_m ) * 1000.0
        position_error_mm = float( np.linalg.norm(position_error_vector_mm) )

        yaw_error_deg = calculate_yaw_error_deg( current_yaw_deg=state["current_yaw_deg"],
            target_yaw_deg=target_yaw_deg,
        )
        absolute_yaw_error_deg = abs(yaw_error_deg)

        limit_info = joint_limit_info( state["q_model_deg"] )

        residual_norm = float( np.linalg.norm(
                xyz_tool_down_yaw_residual( q_model_rad=solution_q_rad,
                    robot_chain=robot_chain, target_position_m=target_position_m,
                    target_yaw_axis=target_yaw_axis,
                )
            )
        )

        success = bool( position_error_mm <= POSITION_TOLERANCE_MM
            and state["tool_down_error_deg"] <= TOOL_DOWN_TOLERANCE_DEG
            and absolute_yaw_error_deg <= YAW_TOLERANCE_DEG and limit_info["all_inside_limits"]
        )

        result.update({
            "optimization_result": optimization_result,
            "state": state,
            "position_error_vector_mm": position_error_vector_mm,
            "position_error_mm": position_error_mm,
            "tool_down_error_deg": state["tool_down_error_deg"],
            "current_yaw_deg": state["current_yaw_deg"],
            "yaw_error_deg": yaw_error_deg,
            "absolute_yaw_error_deg": absolute_yaw_error_deg,
            "limit_info": limit_info,
            "residual_norm": residual_norm,
            "success": success,
        })

    except Exception as error:
        result["solver_exception"] = repr(error)

    return result


def successful_results(results: list[dict]) -> list[dict]:
    return [result for result in results if result["success"]]


def solve_first_waypoint( robot_chain, waypoint: dict, start_q_deg: np.ndarray | None = None,
) -> tuple[dict | None, list[dict]]:
    if start_q_deg is None:
        start_q_deg = get_start_q_model_deg()

    start_q_deg = np.asarray( start_q_deg, dtype=float, ).reshape(5)
    target_xyz_mm = np.asarray( waypoint["xyz_mm"], dtype=float, ).reshape(3)
    target_yaw_deg = float( waypoint["yaw_deg"] )
    actual_start_seed = start_q_deg.copy()
    actual_start_seed[4] = yaw_consistent_q5_deg( actual_start_seed[0], target_yaw_deg, )
    actual_start_seed = np.clip( actual_start_seed, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG, )
    seed_list = create_seed_list_deg( target_position_mm=target_xyz_mm,
        target_yaw_deg=target_yaw_deg,
    )
    seed_list[0] = (
        "Actual START yaw adjusted",
        actual_start_seed,
    )

    target_yaw_axis = create_target_yaw_axis( target_yaw_deg )
    target_position_m = target_xyz_mm / 1000.0
    results: list[dict] = []

    for seed_index, (seed_name, seed_q_deg) in enumerate( seed_list, start=1, ):
        results.append( solve_single_seed(
                robot_chain=robot_chain, target_position_m=target_position_m,
                target_yaw_deg=target_yaw_deg, target_yaw_axis=target_yaw_axis,
                seed_name=seed_name, seed_q_deg=seed_q_deg, seed_index=seed_index,
            )
        )

    successful = successful_results(results)
    best_result = ( min( successful, key=lambda result: (
                -result["limit_info"]["minimum_margin_deg"], result["position_error_mm"],
                result["tool_down_error_deg"], result["absolute_yaw_error_deg"],
            ),
        )
        if successful else None
    )

    return best_result, results


def fk_state( chain, q_rad: np.ndarray, ) -> dict:
    q_rad = np.asarray( q_rad, dtype=float, ).reshape(5)
    ikpy_vector = model_q_to_ikpy_vector(q_rad)
    transform = np.asarray( chain.forward_kinematics(ikpy_vector), dtype=float, )
    rotation = transform[:3, :3]
    xyz_m = transform[:3, 3]
    xyz_mm = xyz_m * 1000.0
    tool_state = calculate_tool_yaw_state(rotation)
    tool_axis = tool_state["tool_axis"]
    heading_axis = tool_state["heading_axis"]
    tool_error = float( tool_state["tool_down_error_deg"] )
    yaw_deg = float(tool_state["yaw_deg"])

    return {
        "q_model_rad": q_rad,
        "q_model_deg": np.rad2deg(q_rad),
        "ikpy_vector": ikpy_vector,
        "transform": transform,
        "rotation": rotation,
        "ee_position_m": xyz_m,
        "ee_position_mm": xyz_mm,
        "yaw_axis": heading_axis,
        "tool_down_error_deg": tool_error,
        "current_yaw_deg": yaw_deg,
        "xyz": xyz_mm,
        "tool_axis": tool_axis,
        "heading_axis": heading_axis,
        "yaw": yaw_deg,
        "tool_error": tool_error,
    }


def planner_result( success: bool, q_deg: np.ndarray, position_error: float,
    tool_error: float = np.inf, yaw_error: float = np.inf, minimum_margin_deg: float = np.inf,
) -> dict:
    return {
        "success": bool(success),
        "q": np.asarray(q_deg, dtype=float),
        "position_error": float(position_error),
        "tool_error": float(tool_error),
        "yaw_error": float(yaw_error),
        "minimum_margin_deg": float(minimum_margin_deg),
    }


def solve_position(chain, waypoint: dict, previous_q_deg: np.ndarray) -> dict:
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    previous_q_rad = np.deg2rad(previous_q_deg)

    def residual(q_rad: np.ndarray) -> np.ndarray:
        position_error = fk_state(chain, q_rad)["xyz"] - waypoint["xyz"]
        posture_error = POSITION_POSTURE_WEIGHT * np.rad2deg(q_rad - previous_q_rad)
        return np.r_[position_error, posture_error]

    def solve_from_seed(seed_q_deg: np.ndarray) -> dict:
        seed_q_deg = np.asarray(seed_q_deg, dtype=float).reshape(5)
        result = least_squares( residual, np.deg2rad( np.clip(
                    seed_q_deg, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG,
                )
            ), bounds=( JOINT_LIMIT_LOWER_RAD, JOINT_LIMIT_UPPER_RAD,
            ), x_scale="jac", max_nfev=POSITION_MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )

        q_deg = np.rad2deg(result.x)
        state = fk_state(chain, result.x)
        error = float( np.linalg.norm( state["xyz"] - waypoint["xyz"] ) )
        return planner_result( error <= POSITION_TOLERANCE_MM, q_deg, error, )

    direct = solve_from_seed(previous_q_deg)
    selected = direct

    if not direct["success"]:
        target_xyz = np.asarray( waypoint["xyz"], dtype=float, ).reshape(3)

        if np.linalg.norm(target_xyz[:2]) > 1.0e-9:
            radial_q1_deg = wrap_to_180_deg( np.rad2deg( np.arctan2( target_xyz[1], target_xyz[0], )
                )
            )

            backup_seed_q_deg = previous_q_deg.copy()
            backup_seed_q_deg[0] = radial_q1_deg

            backup = solve_from_seed(backup_seed_q_deg)

            if backup["success"]:
                selected = backup
            else:
                selected = min( (direct, backup), key=lambda item: item["position_error"], )

    return selected


def nearest_equivalent_j5_deg( angle_deg: float, reference_deg: float, ) -> float | None:
    candidates = np.array( [
            angle_deg - 360.0, angle_deg, angle_deg + 360.0,
        ], dtype=float,
    )
    valid = candidates[ (candidates >= JOINT_LIMIT_LOWER_DEG[4])
        & (candidates <= JOINT_LIMIT_UPPER_DEG[4])
    ]

    if valid.size == 0:
        return None

    return float(valid[np.argmin(np.abs(valid - reference_deg))])


def joint_limit_info(q_deg: np.ndarray) -> dict:
    q_deg = np.asarray( q_deg, dtype=float, ).reshape(5)
    lower_margin = q_deg - JOINT_LIMIT_LOWER_DEG
    upper_margin = JOINT_LIMIT_UPPER_DEG - q_deg
    nearest_margin = np.minimum( lower_margin, upper_margin, )
    inside_limits = nearest_margin >= -1.0e-6
    violated = np.flatnonzero(~inside_limits)

    return {
        "inside_limits": inside_limits,
        "all_inside_limits": bool(np.all(inside_limits)),
        "nearest_margin_deg": nearest_margin,
        "inside": violated.size == 0,
        "lower_margin": lower_margin,
        "upper_margin": upper_margin,
        "violated": violated,
        "minimum_margin_deg": float(
            np.min(nearest_margin)
        ),
    }


def solve_waypoint( chain, waypoint: dict, previous_q_deg: np.ndarray, ) -> dict:
    mode = waypoint["mode"]

    if mode == POSITION_ONLY:
        result = solve_position( chain, waypoint, previous_q_deg, )

    elif mode == FULL_POSE:
        direct = solve_full_pose_single_seed( chain=chain,
            waypoint=waypoint, seed_q_deg=previous_q_deg, reference_q_deg=previous_q_deg,
        )
        result = direct

        if ( not direct["success"] and waypoint.get("name", "") == "PLACE_ABOVE" ):
            candidates = []

            for seed_q_deg in make_place_above_fast_seeds( waypoint, previous_q_deg, ):
                candidate = solve_full_pose_single_seed( chain=chain,
                    waypoint=waypoint, seed_q_deg=seed_q_deg, reference_q_deg=previous_q_deg,
                )
                candidates.append(candidate)

                if candidate["success"]:
                    result = candidate
                    break

            if not result["success"] and candidates:
                result = min( candidates, key=lambda item: ( item["position_error"],
                        item["tool_error"], item["yaw_error"],
                    ),
                )

    else:
        raise ValueError(
            f"지원하지 않는 constraint mode: {mode}"
        )

    return result


def append_j5_continuous_result( solved: list[dict], waypoint: dict, mode: str,
    previous_q_deg: np.ndarray, solver_q_deg: np.ndarray,
) -> np.ndarray | None:
    goal_q_deg = np.asarray(solver_q_deg, dtype=float).copy()
    continuous_j5 = nearest_equivalent_j5_deg(goal_q_deg[4], previous_q_deg[4])

    if continuous_j5 is None:
        print(
            f'[J5_CONTINUITY_FAIL] '
            f'waypoint={waypoint.get("name", "") or "INTERMEDIATE"} '
            f'mode={mode} '
            f'previous={previous_q_deg[4]:.3f} '
            f'solver={goal_q_deg[4]:.3f}'
        )

        return None

    goal_q_deg[4] = continuous_j5
    limit_info = joint_limit_info(goal_q_deg)

    if not limit_info["inside"]:
        print_joint_limit_failure( waypoint, mode, previous_q_deg, goal_q_deg, limit_info )

        return None

    j5_step = abs(continuous_j5 - previous_q_deg[4])
    segment_count = max( int(np.ceil(j5_step / MAX_J5_WAYPOINT_STEP_DEG)), 1, )

    if segment_count > 1:
        print(
            f'[J5_STEP_SPLIT] '
            f'waypoint={waypoint.get("name", "") or "INTERMEDIATE"} '
            f'mode={mode} '
            f'previous={previous_q_deg[4]:.3f} '
            f'goal={continuous_j5:.3f} '
            f'delta={j5_step:.3f} '
            f'segments={segment_count}'
        )
    for segment_index in range(1, segment_count + 1):
        alpha = segment_index / segment_count
        intermediate_q_deg = ( previous_q_deg + alpha * (goal_q_deg - previous_q_deg) )
        solved.append({
            "waypoint": {
                **waypoint,
                "name": (
                    waypoint.get("name", "")
                    if segment_index == segment_count else ""
                ),
            },
            "q_deg": intermediate_q_deg.copy(),
        })

    return goal_q_deg


def _fast_full_pose_residual( q_rad: np.ndarray, chain, target_xyz_mm: np.ndarray,
    target_yaw_deg: float, reference_q_rad: np.ndarray,
) -> np.ndarray:
    state = fk_state(chain, q_rad)
    position_error = ( state["xyz"] - np.asarray(target_xyz_mm, dtype=float) ) / 10.0
    tool_error = 10.0 * ( state["tool_axis"] - WORLD_DOWN )
    target_yaw_rad = np.deg2rad(target_yaw_deg)
    target_heading = np.array([
        np.cos(target_yaw_rad), np.sin(target_yaw_rad), 0.0,
    ], dtype=float)
    yaw_error = 10.0 * ( state["heading_axis"] - target_heading )
    posture_error = 0.01 * np.rad2deg( q_rad - reference_q_rad )

    return np.r_[
        position_error, tool_error, yaw_error, posture_error,
    ]


def solve_full_pose_single_seed( chain, waypoint: dict, seed_q_deg: np.ndarray,
    reference_q_deg: np.ndarray | None = None,
) -> dict:
    seed_q_deg = np.asarray(seed_q_deg, dtype=float).reshape(5)

    if reference_q_deg is None:
        reference_q_deg = seed_q_deg

    reference_q_deg = np.asarray( reference_q_deg, dtype=float, ).reshape(5)
    result = least_squares( _fast_full_pose_residual, np.deg2rad( np.clip(
                seed_q_deg, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG,
            )
        ), bounds=( JOINT_LIMIT_LOWER_RAD, JOINT_LIMIT_UPPER_RAD,
        ), args=( chain, np.asarray(waypoint["xyz"], dtype=float),
            float(waypoint["yaw"]), np.deg2rad(reference_q_deg),
        ), x_scale="jac", max_nfev=FAST_FULL_POSE_MAX_NFEV, ftol=1e-9, xtol=1e-9, gtol=1e-9,
    )
    q_deg = np.rad2deg(result.x)
    state = fk_state(chain, result.x)
    position_error = float( np.linalg.norm( state["xyz"] - np.asarray(waypoint["xyz"], dtype=float)
        )
    )
    yaw_error = abs( calculate_yaw_error_deg( state["yaw"], float(waypoint["yaw"]), ) )
    tool_error = float(state["tool_error"])
    limit_info = joint_limit_info(q_deg)

    return planner_result( position_error <= FAST_FULL_POSE_POSITION_TOLERANCE_MM
        and tool_error <= FAST_FULL_POSE_TOOL_TOLERANCE_DEG
        and yaw_error <= FAST_FULL_POSE_YAW_TOLERANCE_DEG and limit_info["inside"],
        q_deg, position_error, tool_error, yaw_error, limit_info["minimum_margin_deg"],
    )


def make_place_above_fast_seeds( waypoint: dict, previous_q_deg: np.ndarray, ) -> list[np.ndarray]:
    previous_q_deg = np.asarray(previous_q_deg, dtype=float).reshape(5)
    target_xyz = np.asarray(waypoint["xyz"], dtype=float)
    target_yaw = float(waypoint["yaw"])
    radial_q1 = wrap_to_180_deg( np.rad2deg(np.arctan2(target_xyz[1], target_xyz[0])) )
    q1_candidates = ( float(previous_q_deg[0]),
        float(radial_q1), float(wrap_to_180_deg(radial_q1 + 180.0)),
    )
    seeds = []
    seen = set()

    for base_q1 in q1_candidates:
        for q1_offset in PLACE_ABOVE_BACKUP_Q1_OFFSETS_DEG:
            q1 = float(np.clip( wrap_to_180_deg(base_q1 + q1_offset),
                JOINT_LIMIT_LOWER_DEG[0], JOINT_LIMIT_UPPER_DEG[0],
            ))
            q5 = nearest_equivalent_j5_deg( yaw_consistent_q5_deg(q1, target_yaw),
                previous_q_deg[4],
            )

            if q5 is None:
                continue
            for offsets in PLACE_ABOVE_BACKUP_ARM_OFFSETS_DEG:
                seed = previous_q_deg.copy()
                seed[0] = q1
                seed[1:4] += np.asarray(offsets, dtype=float)
                seed[4] = q5
                seed = np.clip( seed, JOINT_LIMIT_LOWER_DEG, JOINT_LIMIT_UPPER_DEG, )
                key = tuple(np.round(seed, 5))

                if key not in seen:
                    seen.add(key)
                    seeds.append(seed)

    return seeds


def solve_sequence( chain, waypoints: list[dict], initial_q_deg: np.ndarray, ) -> list[dict] | None:
    q_deg = np.asarray(initial_q_deg, dtype=float)
    solved: list[dict] = []

    for waypoint in waypoints[1:]:
        mode = waypoint["mode"]
        result = solve_waypoint(chain, waypoint, q_deg)

        if not result["success"]:
            if _ACTIVE_NODE is not None:
                _ACTIVE_NODE.get_logger().error(
                    "IK 실패 | "
                    f"waypoint={waypoint.get('name', '') or 'INTERMEDIATE'} | "
                    f"mode={mode} | "
                    f"xyz={np.round(waypoint['xyz'], 1).tolist()} | "
                    f"yaw={float(waypoint['yaw']):.1f}deg | "
                    f"seed_q={np.round(q_deg, 1).tolist()} | "
                    f"best_q={np.round(result.get('q', np.full(5, np.nan)), 1).tolist()} | "
                    f"pos_err={float(result.get('position_error', np.inf)):.2f}mm | "
                    f"tool_err={float(result.get('tool_error', np.inf)):.2f}deg | "
                    f"yaw_err={float(result.get('yaw_error', np.inf)):.2f}deg"
                )

            return None

        q_next = append_j5_continuous_result( solved, waypoint, mode, q_deg, result["q"], )

        if q_next is None:
            return None

        q_deg = q_next

    return solved


def make_keypoints( start_xyz: np.ndarray, pick_xyz: np.ndarray, place_xyz: np.ndarray,
    start_yaw: float, pick_yaw: float, place_yaw: float, target_class_name: str | None = None,
) -> list[dict]:
    if target_class_name == "Bottle":
        place_above_z = ( place_xyz[2] + BOTTLE_PLACE_APPROACH_HEIGHT_MM )
        pick_above_z = max( pick_xyz[2] + BOTTLE_PICK_CLEARANCE_MM, BOTTLE_SAFE_TRANSPORT_Z_MM, )
    else:
        place_above_z = ( place_xyz[2] + PLACE_APPROACH_HEIGHT_MM )
        pick_above_z = max( pick_xyz[2] + 100.0, SAFE_ABOVE_Z_MM, )

    pick_above = np.array([
        pick_xyz[0], pick_xyz[1], pick_above_z,
    ])
    place_above = np.array([
        place_xyz[0], place_xyz[1], place_above_z,
    ])

    return [ {
            "name": "START",
            "xyz": start_xyz,
            "yaw": start_yaw,
            "mode": POSITION_ONLY,
            "target_class_name": target_class_name,
        }, {
            "name": "PICK_ABOVE",
            "xyz": pick_above,
            "yaw": pick_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        }, {
            "name": "PICK",
            "xyz": pick_xyz,
            "yaw": pick_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        }, {
            "name": "PICK_RETURN",
            "xyz": pick_above,
            "yaw": pick_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        }, {
            "name": "PLACE_ABOVE",
            "xyz": place_above,
            "yaw": place_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        }, {
            "name": "PLACE",
            "xyz": place_xyz,
            "yaw": place_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        }, {
            "name": "PLACE_RETURN",
            "xyz": place_above,
            "yaw": place_yaw,
            "mode": FULL_POSE,
            "target_class_name": target_class_name,
        },
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
        transport = ( start["name"] == "PICK_RETURN" and goal["name"] == "PLACE_ABOVE" )
        is_bottle_transport = ( transport and start.get("target_class_name") == "Bottle" )
        yaw_delta = wrap_to_180_deg( goal["yaw"] - start["yaw"] )
        control = None
        p0 = p1 = p2 = p3 = None

        if is_bottle_transport:
            p0 = np.asarray(start["xyz"], dtype=float)
            safe_z = max( float(p0[2]), float(BOTTLE_SAFE_TRANSPORT_Z_MM), )
            p1 = np.array([
                p0[0], p0[1], safe_z,
            ], dtype=float)
            p2 = np.array([
                goal["xyz"][0], goal["xyz"][1], safe_z,
            ], dtype=float)
            p3 = np.asarray(goal["xyz"], dtype=float)
            length = ( np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2) )
        elif transport:
            control = bezier_control( start["xyz"], goal["xyz"], )
            length = ( np.linalg.norm(control - start["xyz"])
                + np.linalg.norm(goal["xyz"] - control)
            )
        else:
            length = np.linalg.norm( goal["xyz"] - start["xyz"] )
        if transport:
            count = max( int(np.ceil(length / CARTESIAN_STEP_MM)), 1, )
        else:
            count = max( int(np.ceil(length / CARTESIAN_STEP_MM)),
                int(np.ceil(abs(yaw_delta) / YAW_STEP_DEG)), 1,
            )
        for index in range(1, count + 1):
            alpha = index / count

            if is_bottle_transport:
                xyz = ( (1.0 - alpha) ** 3 * p0
                    + 3.0 * (1.0 - alpha) ** 2 * alpha * p1 + 3.0 * (1.0 - alpha) * alpha**2 * p2
                    + alpha**3 * p3
                )
            elif control is None:
                xyz = ( (1.0 - alpha) * start["xyz"] + alpha * goal["xyz"] )
            else:
                xyz = ( (1.0 - alpha) ** 2 * start["xyz"]
                    + 2.0 * (1.0 - alpha) * alpha * control + alpha ** 2 * goal["xyz"]
                )
            if transport:
                is_final = index == count

                if is_final:
                    yaw = goal["yaw"]
                    mode = goal["mode"]
                else:
                    yaw = start["yaw"]
                    mode = POSITION_ONLY
            else:
                yaw = wrap_to_180_deg( start["yaw"] + alpha * yaw_delta )
                mode = goal["mode"]

            waypoints.append({
                "name": (
                    goal["name"]
                    if index == count else ""
                ),
                "xyz": xyz,
                "yaw": yaw,
                "mode": mode,
                "target_class_name": start.get(
                    "target_class_name"
                ),
            })

    return waypoints


def joint_transition( start_q_deg: np.ndarray, goal_q_deg: np.ndarray, ) -> list[dict]:
    difference = goal_q_deg - start_q_deg
    count = max( int(np.ceil(np.max(np.abs(difference)) / MAX_START_JOINT_STEP_DEG)), 1, )

    return [ {
            "waypoint": {"name": "START" if index == 0 else "PICK_ABOVE" if index == count else ""},
            "q_deg": start_q_deg + (index / count) * difference,
        }
        for index in range(count + 1)
    ]


def choose_first_waypoint_j5_branch( candidate_q_deg: np.ndarray,
    start_q_deg: np.ndarray, place_xyz: np.ndarray, place_yaw_deg: float,
) -> np.ndarray | None:
    candidate_q_deg = np.asarray(candidate_q_deg, dtype=float).copy()
    start_q_deg = np.asarray(start_q_deg, dtype=float)
    place_xyz = np.asarray(place_xyz, dtype=float)

    raw_j5 = float(candidate_q_deg[4])
    pick_equivalents = np.array( [raw_j5 - 360.0, raw_j5, raw_j5 + 360.0], dtype=float, )
    pick_equivalents = pick_equivalents[ (pick_equivalents >= JOINT_LIMIT_LOWER_DEG[4])
        & (pick_equivalents <= JOINT_LIMIT_UPPER_DEG[4])
    ]

    selected = None

    if pick_equivalents.size > 0:
        place_q1_deg = wrap_to_180_deg( np.rad2deg( np.arctan2( place_xyz[1], place_xyz[0], ) ) )
        place_nominal_j5_deg = yaw_consistent_q5_deg( place_q1_deg, place_yaw_deg, )

        place_equivalents = np.array( [
                place_nominal_j5_deg - 360.0, place_nominal_j5_deg, place_nominal_j5_deg + 360.0,
            ], dtype=float,
        )
        place_equivalents = place_equivalents[ (place_equivalents >= JOINT_LIMIT_LOWER_DEG[4])
            & (place_equivalents <= JOINT_LIMIT_UPPER_DEG[4])
        ]

        if place_equivalents.size > 0:
            best_pick_j5 = None
            best_cost = np.inf

            for pick_j5 in pick_equivalents:
                start_delta = abs(pick_j5 - start_q_deg[4])
                place_delta = float( np.min( np.abs( place_equivalents - pick_j5 ) ) )

                cost = ( float(J5_BRANCH_START_WEIGHT) * start_delta
                    + float(J5_BRANCH_PLACE_WEIGHT) * place_delta
                )

                if cost < best_cost:
                    best_cost = cost
                    best_pick_j5 = float(pick_j5)

            if best_pick_j5 is not None:
                candidate_q_deg[4] = best_pick_j5
                selected = candidate_q_deg

    return selected


def make_j5_path_continuous(path: list[dict]) -> list[dict] | None:
    if not path:
        return path

    continuous_path: list[dict] = []
    previous_j5 = float(np.asarray(path[0]["q_deg"], dtype=float)[4])

    if not JOINT_LIMIT_LOWER_DEG[4] <= previous_j5 <= JOINT_LIMIT_UPPER_DEG[4]:
        print(f"[J5_LIMIT_FAIL] state=0 j5={previous_j5:.3f}")

        return None
    for index, item in enumerate(path):
        copied = {
            "waypoint": dict(item["waypoint"]),
            "q_deg": np.asarray(item["q_deg"], dtype=float).copy(),
        }

        if index > 0:
            raw_j5 = float(copied["q_deg"][4])

            if ( JOINT_LIMIT_LOWER_DEG[4] <= raw_j5 <= JOINT_LIMIT_UPPER_DEG[4]
                and abs(raw_j5 - previous_j5) <= MAX_J5_WAYPOINT_STEP_DEG + 1.0e-9
            ):
                continuous_j5 = raw_j5
            else:
                continuous_j5 = nearest_equivalent_j5_deg( raw_j5, previous_j5, )
            if continuous_j5 is None:
                print(
                    f"[J5_CONTINUITY_FAIL] state={index} "
                    f"previous={previous_j5:.3f} "
                    f"solver={copied['q_deg'][4]:.3f}"
                )

                return None

            j5_step = abs(continuous_j5 - previous_j5)

            if j5_step > MAX_J5_WAYPOINT_STEP_DEG:
                print(
                    f"[J5_STEP_FAIL] state={index} "
                    f"previous={previous_j5:.3f} "
                    f"next={continuous_j5:.3f} "
                    f"delta={j5_step:.3f}"
                )

                return None

            copied["q_deg"][4] = continuous_j5

        previous_j5 = float(copied["q_deg"][4])
        continuous_path.append(copied)

    return continuous_path


def plan( chain, pick_xyz: np.ndarray, place_xyz: np.ndarray, pick_yaw: float, place_yaw: float,
    start_ticks: np.ndarray | None = None, target_class_name: str | None = None,
) -> list[dict]:
    if start_ticks is None:
        start_ticks = np.asarray( ASSUMED_START_TICKS, dtype=np.int64, ).reshape(5)
    else:
        start_ticks = np.asarray( start_ticks, dtype=np.int64, ).reshape(5)

    start_q_rad = ticks_to_model_rad(start_ticks)
    start_q_deg = np.rad2deg(start_q_rad)
    start = fk_state(chain, start_q_rad)
    pick_xyz_world = np.asarray(pick_xyz, dtype=float).reshape(3)
    place_xyz_world = np.asarray(place_xyz, dtype=float).reshape(3)
    pick_xyz_model = world_xyz_to_model_xyz(pick_xyz_world)
    place_xyz_model = world_xyz_to_model_xyz(place_xyz_world)
    pick_yaw_world = float(pick_yaw)
    place_yaw_world = float(place_yaw)
    pick_yaw_model = world_yaw_to_model_yaw(pick_yaw_world)
    place_yaw_model = world_yaw_to_model_yaw(place_yaw_world)
    keypoints = make_keypoints( start["xyz"], pick_xyz_model, place_xyz_model,
        start["yaw"], pick_yaw_model, place_yaw_model, target_class_name=target_class_name,
    )
    first_waypoint = {
        "xyz_mm": keypoints[1]["xyz"],
        "yaw_deg": keypoints[1]["yaw"],
    }
    _, candidates = solve_first_waypoint( robot_chain=chain,
        waypoint=first_waypoint, start_q_deg=start_q_deg,
    )

    if _ACTIVE_NODE is not None:
        successful_count = sum(1 for item in candidates if item["success"])
        _ACTIVE_NODE.get_logger().info(
            "PICK_ABOVE first-waypoint IK 결과 | "
            f"success={successful_count}/{len(candidates)} | "
            f"xyz={np.round(first_waypoint['xyz_mm'], 1).tolist()} | "
            f"yaw={float(first_waypoint['yaw_deg']):.1f}deg"
        )

    candidate_weights = np.asarray( IK_CANDIDATE_JOINT_WEIGHTS, dtype=float, ).reshape(5)

    candidates = sorted( successful_results(candidates), key=lambda item: ( float(
                np.linalg.norm( ( np.asarray( item["state"]["q_model_deg"], dtype=float,
                        ) - start_q_deg
                    ) * candidate_weights
                )
            ), -float(item["limit_info"]["minimum_margin_deg"]),
        ),
    )[:4]

    if not candidates:
        raise RuntimeError(
            "PICK_ABOVE first-waypoint IK가 모든 seed에서 실패했습니다."
        )
    for candidate in candidates:
        raw_candidate_q = np.asarray( candidate["state"]["q_model_deg"], dtype=float, )
        candidate_place_yaw_model = choose_place_symmetric_yaw_for_min_j5(
            target_yaw_deg=place_yaw_model, place_xyz_mm=place_xyz_model,
            reference_j5_deg=float(raw_candidate_q[4]),
        )
        candidate_q = choose_first_waypoint_j5_branch( candidate_q_deg=raw_candidate_q,
            start_q_deg=start_q_deg, place_xyz=place_xyz_model,
            place_yaw_deg=candidate_place_yaw_model,
        )

        if candidate_q is None:
            continue

        candidate_keypoints = make_keypoints( start["xyz"], pick_xyz_model, place_xyz_model,
            start["yaw"], pick_yaw_model,
            candidate_place_yaw_model, target_class_name=target_class_name,
        )

        if _ACTIVE_NODE is not None:
            _ACTIVE_NODE.get_logger().info(
                "Planner candidate 시도 | "
                f"seed={candidate.get('seed_name', 'unknown')} | "
                f"q={np.round(candidate_q, 1).tolist()} | "
                f"place_yaw={candidate_place_yaw_model:.1f}deg"
            )

        solved = solve_sequence( chain, interpolate(candidate_keypoints[1:]), candidate_q, )

        if solved is None:
            if _ACTIVE_NODE is not None:
                _ACTIVE_NODE.get_logger().warn(
                    "Planner candidate 실패 | "
                    f"seed={candidate.get('seed_name', 'unknown')}"
                )
            continue
        if solved is not None:
            cartesian_path = solved
            candidate_q = candidate_q.copy()
            full_path = [
                *joint_transition(start_q_deg, candidate_q), *cartesian_path,
            ]
            continuous_path = make_j5_path_continuous(full_path)

            if continuous_path is None:
                continue

            return continuous_path

    raise RuntimeError("FAST planner가 전체 Pick & Place 경로를 찾지 못했습니다.")


def path_to_ticks( path: list[dict], expected_start_ticks: np.ndarray, ) -> list[np.ndarray]:
    expected = np.asarray( expected_start_ticks, dtype=np.int64, ).reshape(5)
    ticks_path = []

    for index, item in enumerate(path):
        q_deg = np.asarray( item["q_deg"], dtype=float, ).reshape(5)
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


def transform_point( transform: np.ndarray, point_xyz_m: np.ndarray, ) -> np.ndarray:
    point_h = np.ones(4, dtype=float)
    point_h[:3] = np.asarray( point_xyz_m, dtype=float, ).reshape(3)

    return ( np.asarray( transform, dtype=float, ).reshape(4, 4)
        @ point_h
    )[:3]


def nearest_gripper_symmetric_yaw_deg( target_yaw_deg: float, reference_yaw_deg: float, ) -> float:
    delta = ( ( float(target_yaw_deg) - float(reference_yaw_deg) + 90.0 ) % 180.0 - 90.0 )

    return wrap_to_180_deg( float(reference_yaw_deg) + delta )


def choose_place_symmetric_yaw_for_min_j5( target_yaw_deg: float,
    place_xyz_mm: np.ndarray, reference_j5_deg: float,
) -> float:
    if not bool(USE_GRIPPER_180_SYMMETRY_FOR_PLACE):
        return float(target_yaw_deg)

    place_xyz_mm = np.asarray(place_xyz_mm, dtype=float).reshape(3)
    place_q1_deg = wrap_to_180_deg( np.rad2deg(np.arctan2(place_xyz_mm[1], place_xyz_mm[0])) )
    yaw_candidates = ( wrap_to_180_deg(float(target_yaw_deg)),
        wrap_to_180_deg(float(target_yaw_deg) + 180.0),
    )
    best_yaw = float(yaw_candidates[0])
    best_delta = float("inf")

    for yaw_deg in yaw_candidates:
        nominal_j5 = yaw_consistent_q5_deg( place_q1_deg, float(yaw_deg), )
        equivalent_j5 = nearest_equivalent_j5_deg( nominal_j5, float(reference_j5_deg), )

        if equivalent_j5 is None:
            continue

        delta = abs(float(equivalent_j5) - float(reference_j5_deg))

        if delta < best_delta:
            best_delta = delta
            best_yaw = float(yaw_deg)

    return best_yaw


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

    rotation = np.asarray( data["R_cam2gripper"], dtype=float, ).reshape(3, 3)
    translation = np.asarray( data["t_cam2gripper"], dtype=float, ).reshape(3)

    if not np.allclose( rotation @ rotation.T, np.eye(3), atol=1.0e-3, ):
        raise ValueError(
            "R_cam2gripper가 직교 회전행렬이 아닙니다."
        )
    if abs( float(np.linalg.det(rotation)) - 1.0 ) > 1.0e-3:
        raise ValueError(
            "R_cam2gripper determinant가 1이 아닙니다."
        )

    transform = np.eye( 4, dtype=float, )
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def point_to_numpy(point):
    return np.array( [
            float(point.x), float(point.y), float(point.z),
        ], dtype=float,
    )


class ArmControlNode(Node):
    def __init__(self) -> None:
        global _ACTIVE_NODE
        super().__init__("task3_arm_control_node")
        _ACTIVE_NODE = self
        self._emergency_torque_off_done = False
        self.callback_group = ReentrantCallbackGroup()
        self.motion_lock = threading.RLock()
        self.detection_condition = threading.Condition(threading.RLock())
        self.declare_parameter("detections_topic", "/mission3/detections")
        self.declare_parameter("handeye_path", DEFAULT_HANDEYE_PATH)
        self.declare_parameter("minimum_confidence", MINIMUM_CONFIDENCE)
        self.declare_parameter("search_timeout_sec", SEARCH_TIMEOUT_SEC)
        self.declare_parameter("floor_z_offset_m", FLOOR_Z_OFFSET_M)
        self.declare_parameter(
            "search_pose_ticks",
            [int(v) for v in np.asarray(SEARCH_POSE_TICKS, dtype=np.int64)],
        )
        self.declare_parameter(
            "basket_aruco_search_pose_ticks",
            [int(v) for v in np.asarray(BASKET_ARUCO_SEARCH_POSE_TICKS, dtype=np.int64)],
        )
        self.declare_parameter("place_yaw_deg", PLACE_YAW_DEG)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.minimum_confidence = float( self.get_parameter("minimum_confidence").value )
        self.search_timeout_sec = float( self.get_parameter("search_timeout_sec").value )
        self.floor_z_offset_m = float( self.get_parameter("floor_z_offset_m").value )
        self.search_pose_ticks = np.asarray( self.get_parameter("search_pose_ticks").value,
            dtype=np.int64,
        ).reshape(5)
        self.basket_aruco_search_pose_ticks = np.asarray(
            self.get_parameter("basket_aruco_search_pose_ticks").value, dtype=np.int64,
        ).reshape(5)
        self.place_yaw_deg = float( self.get_parameter("place_yaw_deg").value )
        self.t_flange_camera = load_t_flange_camera( str(self.get_parameter("handeye_path").value) )
        self.chain = create_robot_chain()
        self.latest_detection_msg: DetectionArray | None = None
        self.latest_detection_stamp_ns = 0
        self.placed_count = 0
        self.valuable_placed_count = 0
        self.trash_placed_count = 0
        self.search_cycle_active = False
        self.basket_anchor_floor_m: np.ndarray | None = None
        self.name_tag_place_surface_floor_m: np.ndarray | None = None
        self.auto_sequence_lock = threading.Lock()
        self.auto_sequence_active = False
        self.auto_sequence_done = threading.Event()
        self.auto_sequence_thread_id: int | None = None
        self.auto_sequence_success = False
        self.auto_sequence_message = ""
        self.auto_duplicate_begin_seen = False
        self.auto_waiting_manager_begin_ack = False
        self.aruco_floor_m: np.ndarray | None = None
        self.aruco_yaw_deg: float | None = None
        self.move_ticks_client = self.create_client( MoveToTicks,
            "/motor/move_to_ticks",
            callback_group=self.callback_group,
        )
        self.execute_client = self.create_client( ExecutePickPlace,
            "/arm/execute_pick_place",
            callback_group=self.callback_group,
        )
        self.torque_off_client = self.create_client( Trigger,
            "/motor/torque_off",
            callback_group=self.callback_group,
        )
        self.start_detect_client = self.create_client( Trigger,
            "/mission3/start_detect",
            callback_group=self.callback_group,
        )
        self.lift_down_client = self.create_client( Trigger,
            "/lift/down",
            callback_group=self.callback_group,
        )
        self.subscription = self.create_subscription( DetectionArray,
            self.detections_topic, self.on_detections, 10, callback_group=self.callback_group,
        )
        self.begin_seat_service = self.create_service( Trigger,
            "/arm/begin_seat",
            self.handle_begin_seat, callback_group=self.callback_group,
        )
        self.pick_place_service = self.create_service( PickPlace,
            "/arm/pick_place",
            self.handle_pick_place, callback_group=self.callback_group,
        )
        self.home_service = self.create_service( Trigger,
            "/arm/home",
            self.handle_home, callback_group=self.callback_group,
        )
        self.get_logger().info(
            "Task3 Arm Control ready | "
            f"detections={self.detections_topic} | "
            f"search_ticks={self.search_pose_ticks.tolist()} | "
            f"basket_aruco_ticks={self.basket_aruco_search_pose_ticks.tolist()} | "
            f"search_timeout={self.search_timeout_sec:.1f}s | "
            f"trash_basket=mirror_aruco_y | "
            f"keep_pick_yaw_on_place={bool(KEEP_PICK_YAW_ON_PLACE)}"
        )

    def _reset_search_cycle_state(self) -> None:
        self.placed_count = 0
        self.valuable_placed_count = 0
        self.trash_placed_count = 0
        self.search_cycle_active = True
        self.aruco_floor_m = None
        self.aruco_yaw_deg = None
        self.basket_anchor_floor_m = None
        self.name_tag_place_surface_floor_m = None

    def emergency_torque_off(self) -> None:
        if self._emergency_torque_off_done:
            return

        self._emergency_torque_off_done = True

        try:
            if self.torque_off_client.wait_for_service(timeout_sec=0.5):
                future = self.torque_off_client.call_async(Trigger.Request())
                deadline = time.monotonic() + 1.0

                while time.monotonic() < deadline and not future.done():
                    time.sleep(0.01)
        except Exception as error:
            print(f"[EMERGENCY] motor torque-off request failed: {error}")

    def destroy_node(self):
        self.emergency_torque_off()

        return super().destroy_node()

    def on_detections(self, msg: DetectionArray) -> None:
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds

        with self.detection_condition:
            if stamp_ns >= self.latest_detection_stamp_ns:
                self.latest_detection_msg = deepcopy(msg)
                self.latest_detection_stamp_ns = stamp_ns
                self.detection_condition.notify_all()

    def _valid_candidates( self, msg: DetectionArray, category: str, ) -> list[Detection]:
        if REQUIRE_FLANGE_FRAME and msg.header.frame_id != "flange":
            return []

        category = str(category)
        candidates = []

        for detection in msg.detections:
            if str(getattr(detection, "category", "")) != category:
                continue

            confidence = float(getattr(detection, "confidence", 0.0))

            if not np.isfinite(confidence) or confidence < self.minimum_confidence:
                continue

            candidates.append(detection)

        return candidates

    def wait_for_category_after( self, category: str, gate_stamp_ns: int, timeout_sec: float,
    ) -> tuple[Detection | None, DetectionArray | None]:
        deadline = time.monotonic() + float(timeout_sec)

        with self.detection_condition:
            while time.monotonic() < deadline:
                msg = self.latest_detection_msg
                stamp_ns = self.latest_detection_stamp_ns

                if msg is not None and stamp_ns > gate_stamp_ns:
                    if REQUIRE_FLANGE_FRAME and msg.header.frame_id != "flange":
                        self.get_logger().error(
                            f"Detection frame='{msg.header.frame_id}'. "
                            "flange frame이 아니므로 pick에 사용하지 않습니다."
                        )
                    else:
                        candidates = self._valid_candidates( msg, category, )

                        if candidates:
                            def sort_key(det: Detection):
                                distance = float( getattr(det, "distance_m", np.inf) )
                                if ( not np.isfinite(distance) or distance <= 0.0 ):
                                    distance = np.inf
                                return ( -float(det.confidence), distance, )
                            selected = min( candidates, key=sort_key, )

                            return ( deepcopy(selected), deepcopy(msg), )

                remaining = deadline - time.monotonic()

                if remaining <= 0.0:
                    break

                self.detection_condition.wait( timeout=min(remaining, 0.10) )

        return None, None

    @staticmethod
    def message_has_valid_aruco(msg: DetectionArray) -> bool:
        return bool( getattr(msg, "aruco_detected", False)
            and not getattr(msg, "aruco_is_temp", False)
        )

    def wait_for_valid_aruco_after( self, gate_stamp_ns: int, timeout_sec: float,
    ) -> DetectionArray | None:
        deadline = time.monotonic() + float(timeout_sec)

        with self.detection_condition:
            while time.monotonic() < deadline:
                msg = self.latest_detection_msg
                stamp_ns = self.latest_detection_stamp_ns

                if msg is not None and stamp_ns > gate_stamp_ns:
                    if ( (not REQUIRE_FLANGE_FRAME or msg.header.frame_id == "flange")
                        and self.message_has_valid_aruco(msg)
                    ):
                        return deepcopy(msg)

                remaining = deadline - time.monotonic()

                if remaining <= 0.0:
                    break

                self.detection_condition.wait( timeout=min(remaining, 0.10) )

        return None

    def move_to_ticks(self, goal_ticks: np.ndarray, label: str) -> np.ndarray:
        if not self.move_ticks_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/motor/move_to_ticks service is not available")

        request = MoveToTicks.Request()
        request.goal_ticks = [
            int(v) for v in np.asarray(goal_ticks, dtype=np.int64).reshape(5)
        ]
        request.label = str(label)
        request.profile_velocity = int(SEARCH_PROFILE_VELOCITY)
        request.timeout_sec = float(SEARCH_MOVE_TIMEOUT_SEC)
        future = self.move_ticks_client.call_async(request)
        deadline = time.monotonic() + float(SEARCH_MOVE_TIMEOUT_SEC) + 5.0

        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"{label}: motor move service timeout")

        result = future.result()

        if result is None or not result.success:
            raise RuntimeError( result.message if result is not None else f"{label}: no response" )

        reached = np.asarray(result.reached_ticks, dtype=np.int64).reshape(5)
        self.get_logger().info(f"{label} 도착: {reached.tolist()}")

        return reached

    def move_to_search_pose(self) -> np.ndarray:
        return self.move_to_ticks(self.search_pose_ticks, "TASK3_SEARCH_POSE")

    def move_to_basket_aruco_search_pose(self) -> np.ndarray:
        return self.move_to_ticks( self.basket_aruco_search_pose_ticks,
            "TASK3_BASKET_ARUCO_SEARCH_POSE",
        )

    def move_home(self) -> np.ndarray:
        return self.move_to_ticks( np.asarray(ASSUMED_START_TICKS, dtype=np.int64),
            "TASK3_HOME",
        )

    def base_transform_from_ticks(self, ticks: np.ndarray) -> np.ndarray:
        q_model_rad = ticks_to_model_rad( np.asarray(ticks, dtype=float).reshape(5) )

        return np.asarray( self.chain.forward_kinematics( model_q_to_ikpy_vector(q_model_rad)
            ), dtype=float,
        )

    def detection_to_floor_xyz_m( self, detection: Detection, search_ticks: np.ndarray,
    ) -> np.ndarray:
        t_base_flange = self.base_transform_from_ticks(search_ticks)
        p_flange_m = point_to_numpy(detection.pose)
        p_base_m = transform_point(t_base_flange, p_flange_m)

        return p_base_m + np.array( [0.0, 0.0, self.floor_z_offset_m], dtype=float, )

    def aruco_to_floor_xyz_m( self, msg: DetectionArray, search_ticks: np.ndarray, ) -> np.ndarray:
        if not self.message_has_valid_aruco(msg):
            raise RuntimeError(
                "유효한 ArUco marker가 없습니다."
            )
        if REQUIRE_FLANGE_FRAME and msg.header.frame_id != "flange":
            raise RuntimeError(
                f"ArUco frame='{msg.header.frame_id}' 입니다. "
                "flange frame만 place에 사용할 수 있습니다."
            )

        t_base_flange = self.base_transform_from_ticks( search_ticks )
        p_flange_m = point_to_numpy(msg.aruco_pose)
        p_base_m = transform_point( t_base_flange, p_flange_m, )

        return p_base_m + np.array( [0.0, 0.0, self.floor_z_offset_m], dtype=float, )

    def cache_aruco_from_message( self, msg: DetectionArray, search_ticks: np.ndarray, ) -> bool:
        if not self.message_has_valid_aruco(msg):
            return False

        self.aruco_floor_m = self.aruco_to_floor_xyz_m( msg, search_ticks, )
        yaw = float( getattr(msg, "aruco_yaw_deg", 0.0) )
        self.aruco_yaw_deg = ( yaw if np.isfinite(yaw) else 0.0 )
        self.get_logger().info(
            "ArUco cache 갱신 | "
            f"xyz_mm={np.round(self.aruco_floor_m * 1000.0, 1).tolist()} | "
            f"yaw={self.aruco_yaw_deg:.1f}deg"
        )

        return True

    def camera_yaw_to_floor_yaw_deg( self, detection: Detection, search_ticks: np.ndarray,
    ) -> float:
        if not bool(getattr(detection, "yaw_valid", False)):
            return float(DEFAULT_PICK_YAW_DEG)

        camera_yaw_deg = float(getattr(detection, "yaw_deg", 0.0))

        if not np.isfinite(camera_yaw_deg):
            return float(DEFAULT_PICK_YAW_DEG)

        t_base_flange = self.base_transform_from_ticks(search_ticks)
        r_base_camera = ( t_base_flange[:3, :3]
            @ self.t_flange_camera[:3, :3]
        )
        yaw_rad = np.deg2rad(camera_yaw_deg)
        heading_camera = np.array( [np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=float, )
        heading_base = r_base_camera @ heading_camera

        if np.linalg.norm(heading_base[:2]) < 1.0e-9:
            return float(DEFAULT_PICK_YAW_DEG)

        raw_yaw = wrap_to_180_deg( np.rad2deg(np.arctan2(heading_base[1], heading_base[0])) )
        start_state = fk_state( self.chain,
            ticks_to_model_rad(np.asarray(search_ticks, dtype=float).reshape(5)),
        )

        return nearest_gripper_symmetric_yaw_deg( raw_yaw, float(start_state["yaw"]), )

    def search_once( self, category: str, ) -> tuple[
        Detection | None, np.ndarray | None, float | None, np.ndarray, DetectionArray | None, int,
    ]:
        reached_ticks = self.move_to_search_pose()
        time.sleep(SEARCH_SETTLE_SEC)
        gate_stamp_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f"{category} 탐색 시작: 최대 "
            f"{self.search_timeout_sec:.1f}초"
        )
        detection, source_msg = self.wait_for_category_after( category=category,
            gate_stamp_ns=gate_stamp_ns, timeout_sec=self.search_timeout_sec,
        )

        if detection is None or source_msg is None:
            self.get_logger().warn(
                f"{self.search_timeout_sec:.1f}초 동안 "
                f"category={category} 미검출"
            )

            return ( None, None, None, reached_ticks, None, gate_stamp_ns, )
        if self.basket_anchor_floor_m is None:
            self.cache_aruco_from_message( source_msg, reached_ticks, )

        class_name = str(detection.class_name)
        pick_floor_m = self.detection_to_floor_xyz_m( detection, reached_ticks, )
        grasp_depth_m = float( PICK_GRASP_DEPTH_FROM_TOP_M.get(
                class_name, DEFAULT_PICK_GRASP_DEPTH_FROM_TOP_M,
            )
        )
        pick_floor_m[2] += ( TOOL_HORN_TO_GRASP_M - grasp_depth_m )
        fine_offset_m = np.asarray( PICK_FINE_OFFSET_M.get( class_name, DEFAULT_PICK_FINE_OFFSET_M,
            ), dtype=float,
        ).reshape(3)
        pick_floor_m += fine_offset_m
        pick_yaw_deg = self.camera_yaw_to_floor_yaw_deg( detection, reached_ticks, )
        self.get_logger().info(
            "분실물 검출 | "
            f"category={category} | "
            f"class={class_name} | "
            f"conf={float(detection.confidence):.3f} | "
            f"pick_mm={np.round(pick_floor_m * 1000.0, 1).tolist()} | "
            f"yaw={pick_yaw_deg:.1f}deg"
        )

        return ( detection, pick_floor_m * 1000.0, float(pick_yaw_deg),
            reached_ticks, source_msg, gate_stamp_ns,
        )

    def prepare_basket_aruco_anchor(self) -> np.ndarray:
        aruco_ticks = self.move_to_basket_aruco_search_pose()
        time.sleep(SEARCH_SETTLE_SEC)
        gate_stamp_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            "AUTO SEQUENCE: 1) Basket ArUco marker 인식 시작 | "
            f"ticks={aruco_ticks.tolist()} | timeout={self.search_timeout_sec:.1f}s"
        )
        aruco_msg = self.wait_for_valid_aruco_after( gate_stamp_ns=gate_stamp_ns,
            timeout_sec=self.search_timeout_sec,
        )

        if aruco_msg is None:
            raise RuntimeError("basket ArUco marker를 찾지 못했습니다.")

        self.cache_aruco_from_message(aruco_msg, aruco_ticks)
        raw_aruco_floor_m = np.asarray( self.aruco_floor_m, dtype=float, ).reshape(3).copy()
        aruco_place_offset_m = np.asarray( BASKET_ARUCO_PLACE_OFFSET_MM, dtype=float,
        ).reshape(3) / 1000.0
        self.basket_anchor_floor_m = ( raw_aruco_floor_m + aruco_place_offset_m )
        self.name_tag_place_surface_floor_m = ( self.basket_anchor_floor_m.copy() )
        self.get_logger().info(
            "AUTO SEQUENCE: Basket ArUco 기준점 고정 저장 | "
            f"raw_mm={np.round(raw_aruco_floor_m * 1000.0, 1).tolist()} | "
            f"offset_mm={np.round(BASKET_ARUCO_PLACE_OFFSET_MM, 1).tolist()} | "
            f"anchor_mm={np.round(self.basket_anchor_floor_m * 1000.0, 1).tolist()}"
        )

        return aruco_ticks

    def _name_tag_place_surface( self, request: PickPlace.Request, ) -> np.ndarray:
        place_floor_m = np.asarray( self.basket_anchor_floor_m, dtype=float, ).reshape(3).copy()
        place_floor_m += point_to_numpy(request.place_offset)

        return place_floor_m

    def _valuable_place_surface(self) -> np.ndarray:
        if self.name_tag_place_surface_floor_m is None:
            raise RuntimeError(
                "name_tag place 기준점이 없습니다. Basket ArUco/name_tag 단계가 먼저 완료되어야 합니다."
            )

        place_floor_m = np.asarray( self.name_tag_place_surface_floor_m, dtype=float,
        ).reshape(3).copy()
        base_offset_mm = np.asarray( VALUABLE_PLACE_OFFSET_FROM_NAME_TAG_MM, dtype=float,
        ).reshape(3)
        step_offset_mm = np.asarray( VALUABLE_ITEM_SPACING_MM, dtype=float, ).reshape(3)
        place_floor_m += ( base_offset_mm + self.valuable_placed_count * step_offset_mm ) / 1000.0

        return place_floor_m

    def _trash_place_surface(self) -> np.ndarray:
        if self.basket_anchor_floor_m is None:
            raise RuntimeError(
                "trash basket 기준 계산에 필요한 Basket ArUco anchor가 없습니다."
            )

        place_floor_m = np.asarray( self.basket_anchor_floor_m, dtype=float, ).reshape(3).copy()
        place_floor_m[1] *= -1.0
        place_floor_m += np.asarray( TRASH_BASKET_FINE_OFFSET_MM, dtype=float, ).reshape(3) / 1000.0
        trash_step_mm = np.asarray( TRASH_ITEM_SPACING_MM, dtype=float, ).reshape(3)
        place_floor_m += ( self.trash_placed_count * trash_step_mm ) / 1000.0

        return place_floor_m

    def calculate_place_target( self, request: PickPlace.Request, target_class_name: str,
        search_ticks: np.ndarray, gate_stamp_ns: int,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        class_name = str(target_class_name)
        place_mode = int(request.place_mode)
        planner_start_ticks = np.asarray( search_ticks, dtype=np.int64, ).reshape(5).copy()

        if place_mode == PickPlace.Request.PLACE_ARUCO_OFFSET:
            if self.basket_anchor_floor_m is None:
                aruco_ticks = self.prepare_basket_aruco_anchor()
                planner_start_ticks = aruco_ticks.copy()

            place_floor_m = self._name_tag_place_surface(request)
            place_yaw_deg = float(request.place_yaw_deg)
            self.name_tag_place_surface_floor_m = place_floor_m.copy()
        elif place_mode == PickPlace.Request.PLACE_FIXED:
            category = str(request.category)

            if category == Detection.CATEGORY_VALUABLES:
                place_floor_m = self._valuable_place_surface()
            elif category == Detection.CATEGORY_TRASH:
                place_floor_m = self._trash_place_surface()
            else:
                raise ValueError(
                    f"PLACE_FIXED는 valuables/trash에만 사용합니다: category={category}"
                )

            place_yaw_deg = float(self.place_yaw_deg)
        else:
            raise ValueError(
                f"지원하지 않는 place_mode: {place_mode}"
            )

        grasp_height_m = float( PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M.get(
                class_name, DEFAULT_PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M,
            )
        )
        place_floor_m[2] += ( grasp_height_m + TOOL_HORN_TO_GRASP_M )
        place_floor_m += np.asarray( PLACE_FINE_OFFSET_M.get(
                class_name, DEFAULT_PLACE_FINE_OFFSET_M,
            ), dtype=float,
        ).reshape(3)

        return ( place_floor_m * 1000.0, place_yaw_deg, planner_start_ticks, )

    def execute_pick_place( self, pick_xyz_mm: np.ndarray, place_xyz_mm: np.ndarray,
        pick_yaw_deg: float, place_yaw_deg: float, start_ticks: np.ndarray, target_class_name: str,
    ) -> tuple[bool, str]:
        start_ticks = np.asarray(start_ticks, dtype=np.int64).reshape(5)
        path = plan( self.chain,
            np.asarray(pick_xyz_mm, dtype=float), np.asarray(place_xyz_mm, dtype=float),
            float(pick_yaw_deg), float(place_yaw_deg),
            start_ticks=start_ticks, target_class_name=target_class_name,
        )
        ticks_path = path_to_ticks(path, start_ticks)
        profile = GRIPPER_POSITION_PROFILE.get( target_class_name, DEFAULT_GRIPPER_POSITION_PROFILE,
        )
        names = [
            str(item["waypoint"].get("name", ""))
            for item in path
        ]
        flat = [ int(v)
            for row in ticks_path
            for v in np.asarray(row, dtype=np.int64).reshape(5)
        ]

        if not self.execute_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/arm/execute_pick_place service is not available")

        request = ExecutePickPlace.Request()
        request.joint_ticks = flat
        request.waypoint_names = names
        request.point_count = len(ticks_path)
        request.gripper_open_tick = int(profile["open_tick"])
        request.gripper_close_tick = int(profile["close_tick"])
        future = self.execute_client.call_async(request)
        deadline = time.monotonic() + float(ARM_EXECUTE_TIMEOUT_SEC)

        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError("motor execute service timeout")

        result = future.result()

        if result is None:
            raise RuntimeError("motor execute returned no response")
        if not result.success:
            raise RuntimeError(str(result.message))

        return True, str(result.message)

    @staticmethod
    def _collision_error(message: str) -> bool:
        return 'COLLISION_DETECTED' in str(message)

    def _wait_trigger_client( self, client, name: str, timeout_sec: float, ) -> bool:
        success = False

        if client.wait_for_service( timeout_sec=min(float(timeout_sec), 5.0) ):
            future = client.call_async(Trigger.Request())
            deadline = time.monotonic() + float(timeout_sec)

            while ( rclpy.ok() and not future.done() and time.monotonic() < deadline ):
                time.sleep(0.01)

            if not future.done():
                self.get_logger().error(
                    f"{name} service timeout"
                )
            else:
                result = future.result()

                if result is None or not result.success:
                    self.get_logger().error(
                        f"{name} failed: "
                        f"{getattr(result, 'message', 'no response')}"
                    )
                else:
                    success = True
        else:
            self.get_logger().error(
                f"{name} service is not available"
            )

        return success

    @staticmethod
    def _motor_port_error(message: str) -> bool:
        text = str(message).lower()

        return any( keyword in text
            for keyword in (
                "port is in use",
                "device reports readiness to read but returned no data",
                "device disconnected",
                "present position 최종 실패",
                "cannot open",
                "serial",
            )
        )

    def _recover_after_collision(self, context: str) -> bool:
        self.get_logger().warn(
            f'COLLISION RECOVERY 시작 | context={context} | HOME -> SEARCH'
        )

        try:
            with self.motion_lock:
                self.move_home()
                self.move_to_search_pose()

            self.get_logger().info('COLLISION RECOVERY 완료')

            return True
        except Exception as error:
            self.get_logger().error(f'COLLISION RECOVERY 실패: {error}')

            return False

    @staticmethod
    def _auto_request(category: str, place_mode: int) -> PickPlace.Request:
        request = PickPlace.Request()
        request.category = str(category)
        request.place_mode = int(place_mode)
        request.place_offset.x = 0.0
        request.place_offset.y = 0.0
        request.place_offset.z = 0.0
        request.place_yaw_deg = 0.0

        return request

    def _run_auto_request( self, category: str, place_mode: int, *, allow_nothing: bool,
    ) -> tuple[bool, bool, str]:
        request = self._auto_request( category, place_mode, )

        result = ( False, False,
            f"{category}: retry exhausted",
        )

        for attempt in range( 1, int(AUTO_MAX_ARM_RETRY) + 1, ):
            response = self.handle_pick_place( request, PickPlace.Response(), )
            message = str(response.message)

            if response.success:
                result = ( True, False, message, )
                break

            if response.nothing_detected:
                result = ( bool(allow_nothing), True, message, )
                break

            self.get_logger().warn(
                f"AUTO {category} 실패 "
                f"{attempt}/{AUTO_MAX_ARM_RETRY}: "
                f"{message}"
            )

            if self._motor_port_error(message):
                self.get_logger().error(
                    f"AUTO {category}: "
                    "Motor serial/port 오류 -> "
                    f"자동 재시도 중단 | {message}"
                )
                result = ( False, False, message, )
                break

            if self._collision_error(message):
                if not self._recover_after_collision(category):
                    result = ( False, False, message, )
                    break

        return result

    def auto_task3_sequence(self) -> None:
        self.auto_sequence_thread_id = threading.get_ident()
        success = False
        message = ''

        try:
            self.get_logger().info(
                'AUTO SEQUENCE: Mission3 자동 뒷정리 시작 '
                '(basket ArUco anchor fixed -> name_tag/valuables same basket -> trash initial-anchor mirrored-Y basket -> lift down)'
            )
            self._wait_trigger_client( self.start_detect_client,
                '/mission3/start_detect',
                AUTO_START_DETECT_WAIT_SEC,
            )

            with self.motion_lock:
                self.prepare_basket_aruco_anchor()
                self.move_to_search_pose()

            ok, nothing, msg = self._run_auto_request( Detection.CATEGORY_NAME_TAG,
                PickPlace.Request.PLACE_ARUCO_OFFSET, allow_nothing=False,
            )

            if not ok:
                raise RuntimeError(
                    'name_tag 자동 원위치 실패: ' + (msg or str(nothing))
                )
            for category in ( Detection.CATEGORY_VALUABLES, Detection.CATEGORY_TRASH, ):
                for count in range(int(AUTO_MAX_COLLECT_ITEMS)):
                    ok, nothing, msg = self._run_auto_request( category,
                        PickPlace.Request.PLACE_FIXED, allow_nothing=True,
                    )

                    if nothing:
                        self.get_logger().info(
                            f'AUTO SEQUENCE: {category} 수거 완료 | count={count}'
                        )
                        break
                    if not ok:
                        raise RuntimeError(f'{category} 자동 수거 실패: {msg}')
                else:
                    raise RuntimeError(
                        f'{category}: AUTO_MAX_COLLECT_ITEMS 초과'
                    )
            with self.motion_lock:
                self.move_home()
                self.search_cycle_active = False
            if not self._wait_trigger_client( self.lift_down_client,
                '/lift/down',
                AUTO_LIFT_DOWN_WAIT_SEC,
            ):
                raise RuntimeError('Mission3 완료 후 /lift/down 실패')

            success = True
            message = 'Mission3 automatic sequence completed'
            self.get_logger().info('AUTO SEQUENCE: Mission3 전체 완료')
        except Exception as error:
            message = str(error)
            self.get_logger().error(f'AUTO SEQUENCE 실패: {error}')

            if self._motor_port_error(message):
                self.get_logger().error(
                    'AUTO SEQUENCE: Motor serial/port 오류이므로 HOME 자동 복구를 생략합니다.'
                )
            else:
                try:
                    with self.motion_lock:
                        self.move_home()
                except Exception as recovery_error:
                    self.get_logger().error(
                        f'AUTO SEQUENCE 실패 후 HOME 복귀 실패: {recovery_error}'
                    )
        finally:
            with self.auto_sequence_lock:
                self.auto_sequence_success = bool(success)
                self.auto_sequence_message = str(message)
                self.auto_sequence_active = False
                self.auto_waiting_manager_begin_ack = not self.auto_duplicate_begin_seen
                self.auto_sequence_thread_id = None
                self.auto_sequence_done.set()

    def _serve_manager_compat_pick_place( self, category: str, response: PickPlace.Response,
    ) -> PickPlace.Response:
        self.auto_sequence_done.wait(timeout=float(AUTO_MANAGER_COMPAT_WAIT_SEC))

        with self.auto_sequence_lock:
            success = bool(self.auto_sequence_success)
            message = str(self.auto_sequence_message)
        if not success:
            response.success = False
            response.task_complete = False
            response.nothing_detected = False
            response.message = 'automatic Mission3 failed: ' + message

            return response
        if category == Detection.CATEGORY_NAME_TAG:
            response.success = True
            response.task_complete = False
            response.nothing_detected = False
            response.message = 'already completed by automatic Mission3 sequence'

            return response
        if category in ( Detection.CATEGORY_VALUABLES, Detection.CATEGORY_TRASH, ):
            response.success = False
            response.task_complete = False
            response.nothing_detected = True
            response.message = 'category already cleared by automatic Mission3 sequence'

            return response

        response.success = False
        response.task_complete = False
        response.nothing_detected = False
        response.message = f'unsupported category after automatic sequence: {category}'

        return response

    def handle_begin_seat( self, request: Trigger.Request, response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        with self.auto_sequence_lock:
            if self.auto_sequence_active:
                self.auto_duplicate_begin_seen = True
                response.success = True
                response.message = 'Mission3 automatic sequence already running'

                return response
            if self.auto_waiting_manager_begin_ack:
                self.auto_waiting_manager_begin_ack = False
                response.success = bool(self.auto_sequence_success)
                response.message = (
                    'Mission3 automatic sequence already completed: '
                    + str(self.auto_sequence_message)
                )

                return response

            self.auto_sequence_active = True
            self.auto_sequence_done.clear()
            self.auto_sequence_success = False
            self.auto_sequence_message = ''
            self.auto_duplicate_begin_seen = False
        try:
            with self.motion_lock:
                self._reset_search_cycle_state()
                reached = np.asarray(self.search_pose_ticks, dtype=np.int64).copy()

            threading.Thread( target=self.auto_task3_sequence, daemon=True, ).start()
            response.success = True
            response.message = (
                'Mission3 automatic sequence initialized (ArUco first) | '
                f'ticks={reached.tolist()} | placed_count reset'
            )
            self.get_logger().info(response.message)
        except Exception as error:
            with self.auto_sequence_lock:
                self.auto_sequence_active = False
                self.auto_sequence_message = str(error)
                self.auto_sequence_done.set()

            response.success = False
            response.message = str(error)
            self.get_logger().error(f'begin_seat(search init) 실패: {error}')

        return response

    def handle_home( self, request: Trigger.Request, response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        with self.auto_sequence_lock:
            if self.auto_sequence_active:
                response.success = True
                response.message = (
                    "Mission3 automatic sequence running; HOME is managed internally"
                )

                return response
        try:
            with self.motion_lock:
                self.move_home()
                self.search_cycle_active = False

            response.success = True
            response.message = "task3 home reached"
        except Exception as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f"home 이동 실패: {error}")

        return response

    def handle_pick_place( self, request: PickPlace.Request, response: PickPlace.Response,
    ) -> PickPlace.Response:
        category = str(request.category)

        with self.auto_sequence_lock:
            auto_active = self.auto_sequence_active
            auto_thread_id = self.auto_sequence_thread_id
            auto_done = self.auto_sequence_done.is_set()
            auto_success = self.auto_sequence_success
        if threading.get_ident() != auto_thread_id and (auto_active or (auto_done and auto_success)):
            return self._serve_manager_compat_pick_place(category, response)
        try:
            with self.motion_lock:
                if not self.search_cycle_active:
                    self._reset_search_cycle_state()
                    self.get_logger().warn(
                        "/arm/begin_seat 없이 pick_place 호출됨 -> "
                        "cycle을 초기화하고 계속 진행"
                    )

                self.get_logger().info(
                    "pick_place 요청 | "
                    f"category={category} | "
                    f"place_mode={int(request.place_mode)} | "
                    f"placed_count={self.placed_count}"
                )
                ( detection, pick_xyz_mm, pick_yaw_deg, search_ticks, source_msg, gate_stamp_ns,
                ) = self.search_once( category=category, )

                if detection is None:
                    response.success = False
                    response.task_complete = False
                    response.nothing_detected = True
                    response.message = (
                        f"category={category} not detected for "
                        f"{self.search_timeout_sec:.1f}s"
                    )
                    self.get_logger().warn( response.message )

                    return response

                class_name = str( detection.class_name )
                ( place_xyz_mm, place_yaw_deg, planner_start_ticks,
                ) = self.calculate_place_target( request=request,
                    target_class_name=class_name, search_ticks=search_ticks,
                    gate_stamp_ns=gate_stamp_ns,
                )

                if bool(KEEP_PICK_YAW_ON_PLACE):
                    place_yaw_deg = float(pick_yaw_deg)

                self.get_logger().info(
                    "Pick & Place 계획 | "
                    f"index={self.placed_count} | "
                    f"category={category} | "
                    f"class={class_name} | "
                    f"place_mm="
                    f"{np.round(place_xyz_mm, 1).tolist()} | "
                    f"place_yaw={place_yaw_deg:.1f}deg"
                )
                ( motor_success, motor_message,
                ) = self.execute_pick_place( pick_xyz_mm=pick_xyz_mm,
                    place_xyz_mm=place_xyz_mm, pick_yaw_deg=float( pick_yaw_deg
                    ), place_yaw_deg=float( place_yaw_deg
                    ), start_ticks=planner_start_ticks, target_class_name=class_name,
                )

                if ( motor_success and int(request.place_mode) == PickPlace.Request.PLACE_FIXED ):
                    self.placed_count += 1

                    if category == Detection.CATEGORY_VALUABLES:
                        self.valuable_placed_count += 1
                    elif category == Detection.CATEGORY_TRASH:
                        self.trash_placed_count += 1

                response.success = bool( motor_success )
                response.task_complete = False
                response.nothing_detected = False
                response.picked = detection
                response.message = (
                    f"{motor_message}; "
                    f"placed_count={self.placed_count}"
                )

                return response
        except Exception as error:
            self.get_logger().error(
                "Task3 pick_place 처리 실패 | "
                f"category={category}: {error}"
            )
            response.success = False
            response.task_complete = False
            response.nothing_detected = False
            response.message = str(error)

            return response


def main(args=None) -> None:
    global _ACTIVE_NODE
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    def emergency_signal_handler(signum, frame):
        del signum, frame
        active = _ACTIVE_NODE
        if active is not None:
            active.emergency_torque_off()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, emergency_signal_handler)
    signal.signal(signal.SIGTERM, emergency_signal_handler)

    try:
        node = ArmControlNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.emergency_torque_off()
    finally:
        executor.shutdown()

        if node is not None:
            node.destroy_node()

        _ACTIVE_NODE = None

        if rclpy.ok():
            rclpy.shutdown()
if __name__ == "__main__":
    main()
