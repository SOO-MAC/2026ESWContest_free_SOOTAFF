#!/usr/bin/env python3
"""Robot geometry, joint mapping, and Dynamixel settings shared by Mission 1."""

from __future__ import annotations

import numpy as np
from ikpy.chain import Chain
from ikpy.link import DHLink, Link, OriginLink


DEG = np.pi / 180.0


# ---------------------------------------------------------------------
# Joint / Dynamixel mapping
# ---------------------------------------------------------------------

TICKS_PER_REV = 4096.0

ASSUMED_START_TICKS = np.array(
    [1950, -200, 7300, 3000, 2048],
    dtype=np.int64,
)

TASK2_START_TICKS = np.array(
    [1950, -200, 7300, 2600, 2048],
    dtype=np.int64,
)

START_MAX_ABS_DELTA_TICKS = np.array(
    [1800, 4500, 6500, 2500, 2200],
    dtype=np.int64,
)

DXL_CENTER_TICK = np.array(
    [1950.0, 2048.0, 2048.0, 2048.0, 2048.0],
    dtype=float,
)

DXL_DIRECTION = np.ones(5, dtype=float)

GEAR_RATIO = np.array(
    [1.0, 5.0, 5.0, 1.0, 1.0],
    dtype=float,
)

JOINT_ZERO_OFFSET_DEG = np.array(
    [0.0, 0.0, 0.0, 8.0, 0.0],
    dtype=float,
)

JOINT_ZERO_OFFSET_RAD = np.deg2rad(JOINT_ZERO_OFFSET_DEG)


# ---------------------------------------------------------------------
# Joint limits
# ---------------------------------------------------------------------

JOINT_LIMIT_LOWER_DEG = np.array(
    [-185.0, -90.0, 0.0, 0.0, -200.0],
    dtype=float,
)

JOINT_LIMIT_UPPER_DEG = np.array(
    [185.0, 90.0, 135.0, 140.0, 200.0],
    dtype=float,
)

JOINT_LIMIT_LOWER_RAD = np.deg2rad(JOINT_LIMIT_LOWER_DEG)
JOINT_LIMIT_UPPER_RAD = np.deg2rad(JOINT_LIMIT_UPPER_DEG)


# ---------------------------------------------------------------------
# Coordinate / tool definition
# ---------------------------------------------------------------------

WORLD_TO_MODEL_ROTATION = np.eye(3, dtype=float)
MODEL_TO_WORLD_ROTATION = WORLD_TO_MODEL_ROTATION.T

WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)

TOOL_HORN_TO_GRASP_M = 0.08735
TOOL_HORN_TO_GRASP_MM = TOOL_HORN_TO_GRASP_M * 1000.0


# ---------------------------------------------------------------------
# Dynamixel communication
# ---------------------------------------------------------------------

ARM_IDS = (1, 2, 3, 4, 5)
GRIPPER_ID = 6

DEVICENAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_"
    "FT6RW85Z-if00-port0"
)

BAUDRATE = 1_000_000
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

OP_POSITION_CONTROL = 3
OP_EXTENDED_POSITION = 4


# ---------------------------------------------------------------------
# Arm motion
# ---------------------------------------------------------------------

ARM_PROFILE_VELOCITY = 15
J2_PROFILE_VELOCITY_SCALE = 5.0
J3_PROFILE_VELOCITY_SCALE = 5.0

ARM_STREAM_DT_SEC = 0.04
ARM_WAYPOINT_TIMEOUT_SEC = 15.0

ARM_POSITION_THRESHOLDS = np.array(
    [60, 40, 100, 30, 30],
    dtype=np.int64,
)

ARM_ALIGN_THRESHOLDS = np.array(
    [60, 40, 75, 15, 35],
    dtype=np.int64,
)

PHASE_MIN_COMMANDS = 12
PHASE_MAX_COMMANDS = 160

PHASE_MAX_TICK_STEP = np.array(
    [80, 160, 320, 160, 80],
    dtype=np.int64,
)

TRANSPORT_MIN_COMMANDS = 80
TRANSPORT_MAX_COMMANDS = 220
TRANSPORT_YAW_COMMANDS = 30

ARM_READ_RETRIES = 4
ARM_READ_RETRY_DT_SEC = 0.03


# Startup / direct tick move

START_PROFILE_VELOCITY = 8

DIRECT_PROFILE_VELOCITY = 8
DIRECT_MAX_TICK_STEP = 60
DIRECT_COMMAND_DT_SEC = 0.03
DIRECT_TIMEOUT_SEC = 60.0


# ---------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------

GRIPPER_PROFILE_VELOCITY = 40
GRIPPER_POSITION_THRESHOLD_TICK = 30
GRIPPER_MOVE_TIMEOUT_SEC = 6.0
GRIPPER_SAMPLE_DT_SEC = 0.05
GRIPPER_SETTLE_SEC = 0.30

GRIPPER_MIN_TICK = 0
GRIPPER_MAX_TICK = 2048
DEFAULT_GRIPPER_OPEN_TICK = 85


# ---------------------------------------------------------------------
# IKPy chain
# ---------------------------------------------------------------------

class FixedDHLink(DHLink):
    """Standard-DH link with explicit name and bounds initialization."""

    def __init__(
        self,
        name,
        d=0.0,
        a=0.0,
        alpha=0.0,
        theta=0.0,
        bounds=None,
        length=0.0,
    ):
        Link.__init__(
            self,
            name=str(name),
            length=float(length),
            bounds=bounds,
        )

        self.d = float(d)
        self.a = float(a)
        self.alpha = float(alpha)
        self.theta = float(theta)

        self.has_rotation = True
        self.has_translation = False
        self.joint_type = "revolute"


def create_robot_chain() -> Chain:
    return Chain(
        name="soomac_5dof_standard_dh",
        links=[
            OriginLink(),
            FixedDHLink(
                "J1",
                d=0.06125,
                alpha=-90.0 * DEG,
                bounds=(
                    JOINT_LIMIT_LOWER_RAD[0],
                    JOINT_LIMIT_UPPER_RAD[0],
                ),
            ),
            FixedDHLink(
                "J2",
                a=0.25,
                theta=-90.0 * DEG,
                bounds=(
                    JOINT_LIMIT_LOWER_RAD[1],
                    JOINT_LIMIT_UPPER_RAD[1],
                ),
            ),
            FixedDHLink(
                "J3",
                a=0.25,
                bounds=(
                    JOINT_LIMIT_LOWER_RAD[2],
                    JOINT_LIMIT_UPPER_RAD[2],
                ),
            ),
            FixedDHLink(
                "J4",
                alpha=90.0 * DEG,
                theta=90.0 * DEG,
                bounds=(
                    JOINT_LIMIT_LOWER_RAD[3],
                    JOINT_LIMIT_UPPER_RAD[3],
                ),
            ),
            FixedDHLink(
                "J5",
                d=0.129,
                bounds=(
                    JOINT_LIMIT_LOWER_RAD[4],
                    JOINT_LIMIT_UPPER_RAD[4],
                ),
            ),
            FixedDHLink(
                "terminal",
                bounds=(0.0, 0.0),
            ),
        ],
        active_links_mask=[
            False,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
    )


# ---------------------------------------------------------------------
# Joint conversion
# ---------------------------------------------------------------------

def model_q_to_ikpy_vector(q_model_rad: np.ndarray) -> np.ndarray:
    q_model_rad = np.asarray(q_model_rad, dtype=float).reshape(-1)

    if q_model_rad.size != 5:
        raise ValueError(
            f"q_model_rad needs 5 joints, got {q_model_rad.size}"
        )

    return np.array(
        [0.0, *q_model_rad.tolist(), 0.0],
        dtype=float,
    )


def ikpy_vector_to_model_q(ikpy_angles: np.ndarray) -> np.ndarray:
    ikpy_angles = np.asarray(ikpy_angles, dtype=float).reshape(-1)

    if ikpy_angles.size != 7:
        raise ValueError(
            f"IKPy vector length must be 7, got {ikpy_angles.size}"
        )

    return ikpy_angles[1:6].copy()


def ticks_to_model_rad(ticks: np.ndarray) -> np.ndarray:
    ticks = np.asarray(ticks, dtype=float).reshape(5)

    motor_rad = (
        (ticks - DXL_CENTER_TICK)
        * (2.0 * np.pi / TICKS_PER_REV)
        / DXL_DIRECTION
    )

    return (
        motor_rad / GEAR_RATIO
        + JOINT_ZERO_OFFSET_RAD
    )


def ticks_to_model_deg(ticks: np.ndarray) -> np.ndarray:
    return np.rad2deg(ticks_to_model_rad(ticks))


def model_deg_to_ticks(q_model_deg: np.ndarray) -> np.ndarray:
    q_model_deg = np.asarray(
        q_model_deg,
        dtype=float,
    ).reshape(5)

    if not np.all(np.isfinite(q_model_deg)):
        raise ValueError("joint angle contains NaN/inf")

    outside_limits = (
        np.any(q_model_deg < JOINT_LIMIT_LOWER_DEG - 1.0e-6)
        or np.any(q_model_deg > JOINT_LIMIT_UPPER_DEG + 1.0e-6)
    )

    if outside_limits:
        raise ValueError(
            f"joint limit exceeded: {np.round(q_model_deg, 3)}"
        )

    motor_deg = (
        q_model_deg - JOINT_ZERO_OFFSET_DEG
    ) * GEAR_RATIO

    ticks = (
        DXL_CENTER_TICK
        + DXL_DIRECTION
        * motor_deg
        * TICKS_PER_REV
        / 360.0
    )

    return np.rint(ticks).astype(np.int64)


def get_start_q_model_rad() -> np.ndarray:
    return ticks_to_model_rad(ASSUMED_START_TICKS)


def get_start_q_model_deg() -> np.ndarray:
    return ticks_to_model_deg(ASSUMED_START_TICKS)


# ---------------------------------------------------------------------
# Coordinate / yaw helpers
# ---------------------------------------------------------------------

def world_xyz_to_model_xyz(world_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(world_xyz, dtype=float).reshape(3)
    return WORLD_TO_MODEL_ROTATION @ xyz


def model_xyz_to_world_xyz(model_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(model_xyz, dtype=float).reshape(3)
    return MODEL_TO_WORLD_ROTATION @ xyz


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))

    if norm < 1.0e-12:
        raise ValueError("zero vector")

    return vector / norm


def wrap_to_180_deg(angle_deg: float) -> float:
    return float(
        (float(angle_deg) + 180.0)
        % 360.0
        - 180.0
    )


def calculate_vector_angle_deg(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    vector_a = normalize_vector(vector_a)
    vector_b = normalize_vector(vector_b)

    cosine = float(
        np.clip(
            np.dot(vector_a, vector_b),
            -1.0,
            1.0,
        )
    )

    return float(np.rad2deg(np.arccos(cosine)))


def get_tool_axis(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    return normalize_vector(rotation[:, 2])


def get_gripper_heading_axis(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    return normalize_vector(-rotation[:, 0])


def projected_yaw_deg(heading_axis: np.ndarray) -> float:
    heading_axis = np.asarray(
        heading_axis,
        dtype=float,
    ).reshape(3)

    if np.linalg.norm(heading_axis[:2]) < 1.0e-10:
        return float("nan")

    return wrap_to_180_deg(
        np.rad2deg(
            np.arctan2(
                heading_axis[1],
                heading_axis[0],
            )
        )
    )


def calculate_tool_yaw_state(rotation: np.ndarray) -> dict:
    tool_axis = get_tool_axis(rotation)
    heading_axis = get_gripper_heading_axis(rotation)

    return {
        "tool_axis": tool_axis,
        "heading_axis": heading_axis,
        "tool_down_error_deg": calculate_vector_angle_deg(
            tool_axis,
            WORLD_DOWN,
        ),
        "yaw_deg": projected_yaw_deg(heading_axis),
    }


def calculate_yaw_error_deg(
    current_yaw_deg: float,
    target_yaw_deg: float,
) -> float:
    if not np.isfinite(current_yaw_deg):
        return float("inf")

    return wrap_to_180_deg(
        float(current_yaw_deg)
        - float(target_yaw_deg)
    )


def yaw_consistent_q5_deg(
    q1_deg: float,
    target_yaw_deg: float,
) -> float:
    return wrap_to_180_deg(
        float(q1_deg)
        - float(target_yaw_deg)
    )


def calculate_tool_down_yaw_from_q_deg(
    q_model_deg: np.ndarray,
) -> float:
    q_model_deg = np.asarray(
        q_model_deg,
        dtype=float,
    ).reshape(5)

    return wrap_to_180_deg(
        q_model_deg[0]
        - q_model_deg[4]
    )


def world_yaw_to_model_yaw(world_yaw_deg: float) -> float:
    yaw_rad = np.deg2rad(float(world_yaw_deg))

    heading_world = np.array(
        [
            np.cos(yaw_rad),
            np.sin(yaw_rad),
            0.0,
        ]
    )

    heading_model = (
        WORLD_TO_MODEL_ROTATION
        @ heading_world
    )

    return projected_yaw_deg(heading_model)


def model_yaw_to_world_yaw(model_yaw_deg: float) -> float:
    yaw_rad = np.deg2rad(float(model_yaw_deg))

    heading_model = np.array(
        [
            np.cos(yaw_rad),
            np.sin(yaw_rad),
            0.0,
        ]
    )

    heading_world = (
        MODEL_TO_WORLD_ROTATION
        @ heading_model
    )

    return projected_yaw_deg(heading_world)


def create_target_yaw_axis(target_yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(target_yaw_deg))

    return np.array(
        [
            np.cos(yaw_rad),
            np.sin(yaw_rad),
            0.0,
        ],
        dtype=float,
    )
