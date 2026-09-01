#!/usr/bin/env python3

import numpy as np

from control_config.robot_config import ASSUMED_START_TICKS

SEARCH_POSE_TICKS = np.asarray(
    ASSUMED_START_TICKS,
    dtype=np.int64,
).copy()

BASKET_ARUCO_SEARCH_POSE_TICKS = np.array(
    [3709, 612, 6950, 3266, 2784],
    dtype=np.int64,
)

SEARCH_SETTLE_SEC = 0.60
SEARCH_TIMEOUT_SEC = 5.0
MINIMUM_CONFIDENCE = 0.60
REQUIRE_FLANGE_FRAME = True

DEFAULT_HANDEYE_PATH = (
    "/home/seungwon/soomac_ws/"
    "src/mission_3/mission_3/hand_eye_result.json"
)

FLOOR_Z_OFFSET_M = 0.041

PICK_GRASP_DEPTH_FROM_TOP_M = {
    "name_tag": 0.0,
    "water_bottle": 0.060,
    "snack_1": 0.010,
    "white_heim": 0.008,
    "snack_3": 0.010,
    "Wallet": 0.0,
    "Phone": 0.0,
}
DEFAULT_PICK_GRASP_DEPTH_FROM_TOP_M = 0.0

PICK_FINE_OFFSET_M = {
    "name_tag": np.array([0.02, -0.01, -0.025], dtype=float),
    "water_bottle": np.array([-0.03, 0.01, 0.05], dtype=float),
    "snack_1": np.array([0.01, -0.015, -0.050], dtype=float),
    "white_heim": np.array([0.01, 0.005, -0.040], dtype=float),
    "snack_3": np.array([-0.01, 0.0, -0.040], dtype=float),
    "Wallet": np.array([0.0, 0.01, -0.03], dtype=float),
    "Phone": np.array([0.0, 0.0, 0.0], dtype=float),
}
DEFAULT_PICK_FINE_OFFSET_M = np.zeros(3, dtype=float)
DEFAULT_PICK_YAW_DEG = 0.0

BASKET_ARUCO_PLACE_OFFSET_MM = np.array([0.0, 0.0, 0.0], dtype=float)

VALUABLE_PLACE_OFFSET_FROM_NAME_TAG_MM = np.array([70.0, 0.0, 0.0], dtype=float)
VALUABLE_ITEM_SPACING_MM = np.array([50.0, 0.0, 0.0], dtype=float)

TRASH_BASKET_FINE_OFFSET_MM = np.array([0.0, 0.0, 0.0], dtype=float)
TRASH_ITEM_SPACING_MM = np.array([50.0, 0.0, 0.0], dtype=float)

KEEP_PICK_YAW_ON_PLACE = True
PLACE_YAW_DEG = 0.0  # KEEP_PICK_YAW_ON_PLACE=False일 때만 사용

PLACE_BASE_SURFACE_XYZ_MM = np.array(
    [-70.0, -130.0, 50.0],
    dtype=float,
)
PLACE_Y_OFFSET_MM = 50.0

PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M = {
    "name_tag": 0.0,
    "water_bottle": 0.070,
    "snack_1": 0.010,
    "white_heim": 0.008,
    "snack_3": 0.010,
    "Wallet": 0.0,
    "Phone": 0.0,
}
DEFAULT_PLACE_GRASP_HEIGHT_ABOVE_SURFACE_M = 0.0

PLACE_FINE_OFFSET_M = {
    "name_tag": np.array([0.0, 0.0, 0.0], dtype=float),
    "water_bottle": np.array([0.0, 0.0, 0.0], dtype=float),
    "snack_1": np.array([0.0, 0.0, 0.0], dtype=float),
    "white_heim": np.array([0.0, 0.0, 0.0], dtype=float),
    "snack_3": np.array([0.0, 0.0, 0.0], dtype=float),
    "Wallet": np.array([0.0, 0.0, 0.0], dtype=float),
    "Phone": np.array([0.0, 0.0, 0.0], dtype=float),
}
DEFAULT_PLACE_FINE_OFFSET_M = np.zeros(3, dtype=float)

GRIPPER_POSITION_PROFILE = {
    "name_tag": {
        "open_tick": 85,
        "close_tick": 2048,
    },
    "snack_1": {
        "open_tick": 200,
        "close_tick": 650,
    },
    "white_heim": {
        "open_tick": 420,
        "close_tick": 900,
    },
    "snack_3": {
        "open_tick": 420,
        "close_tick": 1100,
    },
    "water_bottle": {
        "open_tick": 85,
        "close_tick": 950,
    },
}
DEFAULT_GRIPPER_POSITION_PROFILE = {
    "open_tick": 85,
    "close_tick": 500,
}

MAX_START_JOINT_STEP_DEG = 5.0

SAFE_ABOVE_Z_MM = 230.0
PLACE_APPROACH_HEIGHT_MM = 100.0

BOTTLE_PLACE_APPROACH_HEIGHT_MM = 50.0
BOTTLE_SAFE_TRANSPORT_Z_MM = 330.0
BOTTLE_PICK_CLEARANCE_MM = 40.0

CARTESIAN_STEP_MM = 35.0
YAW_STEP_DEG = 12.0
MAX_J5_WAYPOINT_STEP_DEG = 20.0

IK_CANDIDATE_JOINT_WEIGHTS = np.array([3.0, 1.0, 1.0, 1.0, 3.0], dtype=float)

J5_BRANCH_START_WEIGHT = 1.0
J5_BRANCH_PLACE_WEIGHT = 1.0

USE_GRIPPER_180_SYMMETRY_FOR_PLACE = True

BEZIER_MIN_RADIUS_MM = 320.0
BEZIER_CONTROL_Z_MM = 300.0

POSITION_ONLY = "POSITION_ONLY"
FULL_POSE = "FULL_POSE"

POSITION_TOLERANCE_MM = 2.0
TOOL_DOWN_TOLERANCE_DEG = 2.0
YAW_TOLERANCE_DEG = 2.0

POSITION_POSTURE_WEIGHT = 0.02
POSITION_MAX_NFEV = 45

FAST_FULL_POSE_MAX_NFEV = 220
FAST_FULL_POSE_POSITION_TOLERANCE_MM = 2.0
FAST_FULL_POSE_TOOL_TOLERANCE_DEG = 2.0
FAST_FULL_POSE_YAW_TOLERANCE_DEG = 2.0

PLACE_ABOVE_BACKUP_Q1_OFFSETS_DEG = (
    -20.0,
    0.0,
    20.0,
)

PLACE_ABOVE_BACKUP_ARM_OFFSETS_DEG = (
    (0.0, 0.0, 0.0),
    (5.0, -5.0, 0.0),
    (-5.0, 5.0, 0.0),
)

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

SEARCH_PROFILE_VELOCITY = 8
SEARCH_MOVE_TIMEOUT_SEC = 60.0
ARM_EXECUTE_TIMEOUT_SEC = 180.0

AUTO_MAX_ARM_RETRY = 2
AUTO_MAX_COLLECT_ITEMS = 30
AUTO_MANAGER_COMPAT_WAIT_SEC = 240.0

AUTO_START_DETECT_WAIT_SEC = 10.0
AUTO_LIFT_DOWN_WAIT_SEC = 30.0

COLLISION_ACTIVE_ERROR_TICKS = np.array(
    [450, 500, 1000, 500, 200],
    dtype=np.int64,
)
COLLISION_MIN_PROGRESS_TICK = 3
COLLISION_STALL_SEC = 2.50
COLLISION_CHECK_PERIOD_SEC = 0.05

TASK3_PHASE_MAX_TICK_STEP = np.array(
    [50, 160, 320, 160, 80],
    dtype=np.int64,
)

TASK3_ARM_ALIGN_THRESHOLDS = np.array(
    [60, 40, 75, 20, 30],
    dtype=np.int64,
)