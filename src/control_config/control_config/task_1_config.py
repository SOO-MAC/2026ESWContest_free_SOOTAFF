#!/usr/bin/env python3
"""Mission 1 object, perception, and planner settings."""

import numpy as np

from control_config.robot_config import ASSUMED_START_TICKS


# ---------------------------------------------------------------------
# Scan / detection poses
# ---------------------------------------------------------------------

SCAN_POSES = {
    "CENTER": np.array(
        [1950, *np.asarray(ASSUMED_START_TICKS, dtype=np.int64).reshape(5)[1:]],
        dtype=np.int64,
    ),
}

OBJECT_DETECTION_POSES = {
    "name_tag": np.array([272, -896, 8370, 3070, 2209], dtype=np.int64),
    "Snack_1": np.array([926, 345, 6319, 3508, 2048], dtype=np.int64),
    "Snack_2": np.array([2995, -337, 7466, 3378, 2048], dtype=np.int64),
    "Snack_3": np.array([3709, 612, 6950, 3266, 2784], dtype=np.int64),
    "Bottle": np.array([3979, -2279, 8232, 3160, 2048], dtype=np.int64),
}


# ---------------------------------------------------------------------
# Mission sequence / perception
# ---------------------------------------------------------------------

PICK_SEQUENCE = (
    "name_tag",
    "Snack_1",
    "Snack_2",
    "Snack_3",
    "Bottle",
)

OVERVIEW_POSE_NAME = "CENTER"
ARUCO_INITIAL_SCAN_POSE = "CENTER"

ARUCO_BEGIN_SEAT_WAIT_SEC = 1.0
ARUCO_PICK_GATE_WAIT_SEC = 10.0

DETECTION_WAIT_SEC_PER_POSE = 3.0
DETECTION_SETTLE_SEC = 0.60

REQUIRE_FLANGE_FRAME = True

FLOOR_Z_OFFSET_M = 0.041

DEFAULT_HANDEYE_PATH = (
    "/home/seungwon/soomac_ws/"
    "src/mission_1/mission_1/hand_eye_result.json"
)


# ---------------------------------------------------------------------
# Mission1 automatic placement / retry
# ---------------------------------------------------------------------

AUTO_PLACE_CONFIG = {
    "name_tag": {
        "offset_m": np.array([0.0, 0.0, 0.0], dtype=float),
        "place_yaw_deg": 0.0,
    },
    "Snack_1": {
        "offset_m": np.array([-0.11, -0.12, 0.0], dtype=float),
        "place_yaw_deg": 0.0,
    },
    "Snack_2": {
        "offset_m": np.array([-0.11, 0.00, 0.0], dtype=float),
        "place_yaw_deg": 0.0,
    },
    "Snack_3": {
        "offset_m": np.array([-0.11, 0.12, 0.0], dtype=float),
        "place_yaw_deg": 0.0,
    },
    "Bottle": {
        "offset_m": np.array([-0.04, 0.12, 0.0], dtype=float),
        "place_yaw_deg": 0.0,
    },
}

AUTO_MAX_ARM_RETRY = 2


# ---------------------------------------------------------------------
# Object grasp / place offsets
# ---------------------------------------------------------------------

PICK_GRASP_DEPTH_FROM_TOP_M = {
    "name_tag": 0.000,
    "Snack_1": 0.010,
    "Snack_2": 0.008,
    "Snack_3": 0.010,
    "Bottle": 0.055,
}

PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M = {
    "name_tag": 0.000,
    "Snack_1": 0.010,
    "Snack_2": 0.008,
    "Snack_3": 0.010,
    "Bottle": 0.075,
}

PICK_FINE_OFFSET_M = {
    "name_tag": np.array([-0.01, 0.010, -0.025]),
    "Snack_1": np.array([-0.015, 0.0, -0.050]),
    "Snack_2": np.array([0.01, 0.01, -0.035]),
    "Snack_3": np.array([-0.01, 0.0, -0.035]),
    "Bottle": np.array([-0.03, 0.01, 0.05]),
}

PLACE_FINE_OFFSET_M = {
    "name_tag": np.array([0.0, 0.02, 0.030]),
    "Snack_1": np.array([-0.005, 0.0, -0.030]),
    "Snack_2": np.array([0.0, 0.0, -0.045]),
    "Snack_3": np.array([0.0, 0.0, -0.035]),
    "Bottle": np.array([0.0, 0.0, 0.01]),
}

DEFAULT_PICK_YAW_DEG = {
    "name_tag": 0.0,
    "Snack_1": 0.0,
    "Snack_2": 0.0,
    "Snack_3": 0.0,
    "Bottle": 0.0,
}

NAME_TAG_PLACE_YAW_DEG = -90.0


# ---------------------------------------------------------------------
# Gripper position targets
# ---------------------------------------------------------------------

GRIPPER_POSITION_PROFILE = {
    "name_tag": {"open_tick": 85, "close_tick": 2048},
    "Snack_1": {"open_tick": 50, "close_tick": 650},
    "Snack_2": {"open_tick": 420, "close_tick": 900},
    "Snack_3": {"open_tick": 420, "close_tick": 1100},
    "Bottle": {"open_tick": 85, "close_tick": 950},
}

DEFAULT_GRIPPER_POSITION_PROFILE = {
    "open_tick": 85,
    "close_tick": 2048,
}


# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------

MAX_START_JOINT_STEP_DEG = 5.0

SAFE_ABOVE_Z_MM = 230.0
PLACE_APPROACH_HEIGHT_MM = 75.0

BOTTLE_PICK_ABOVE_MIN_Z_MM = 300.0
BOTTLE_PICK_CLEARANCE_MM = 40.0

# Bottle PLACE_ABOVE는 공통 PLACE_APPROACH_HEIGHT_MM를 기준으로
# 이 값만 추가 보정한다.
# 100 + (-50) = Bottle의 실제 place approach 50 mm.
BOTTLE_PLACE_APPROACH_OFFSET_MM = -50.0

# Bottle의 PICK_RETURN -> PLACE_ABOVE 운반 구간만
# 일반 물체 transport 기준 높이보다 추가로 높인다.
BOTTLE_TRANSPORT_OFFSET_MM = 50.0

CARTESIAN_STEP_MM = 35.0
YAW_STEP_DEG = 12.0
MAX_J5_WAYPOINT_STEP_DEG = 20.0

BEZIER_MIN_RADIUS_MM = 320.0
BEZIER_CONTROL_Z_MM = 300.0

POSITION_ONLY = "POSITION_ONLY"
FULL_POSE = "FULL_POSE"


# Solver tolerances

POSITION_TOLERANCE_MM = 7.0
TOOL_DOWN_TOLERANCE_DEG = 2.0
YAW_TOLERANCE_DEG = 2.0

POSITION_POSTURE_WEIGHT = 0.02
POSITION_MAX_NFEV = 45

FAST_FULL_POSE_MAX_NFEV = 220
FAST_FULL_POSE_POSTURE_WEIGHT = 0.02
FAST_FULL_POSE_POSITION_TOLERANCE_MM = 5.0
FAST_FULL_POSE_TOOL_TOLERANCE_DEG = 5.0
FAST_FULL_POSE_YAW_TOLERANCE_DEG = 5.0

PLACE_ABOVE_BACKUP_Q1_OFFSETS_DEG = (
    -20.0,
    0.0,
    20.0,
)

PLACE_ABOVE_BACKUP_ARM_OFFSETS_DEG = (
    (0.0, 0.0, 0.0),
    (5.0, -5.0, 0.0),
    (-5.0, 5.0, 0.0),
    (-10.0, 20.0, -10.0),
    (10.0, 20.0, -20.0),
    (-20.0, 30.0, -20.0),
    (20.0, 30.0, -30.0),
)


# First-waypoint multi-seed solver

POSITION_RESIDUAL_SCALE_MM = 10.0
TOOL_RESIDUAL_WEIGHT = 10.0
YAW_RESIDUAL_WEIGHT = 10.0

MAX_FUNCTION_EVALUATIONS = 1500

XTOL = 1.0e-12
FTOL = 1.0e-12
GTOL = 1.0e-12

PREVIOUS_POSITION_SOLUTION_DEG = np.array(
    [0.0, 7.874154, 94.077644, 114.397512, 0.0],
    dtype=float,
)


# ---------------------------------------------------------------------
# Arm -> Motor direct move defaults
# ---------------------------------------------------------------------

SCAN_PROFILE_VELOCITY = 15
SCAN_TIMEOUT_SEC = 60.0