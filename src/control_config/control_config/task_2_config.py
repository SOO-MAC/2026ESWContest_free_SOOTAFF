#!/usr/bin/env python3
"""Mission 2 configuration."""

import numpy as np


# ROS interfaces

DETECTION_TOPIC = "/mission2/detections"

ARM_BEGIN_SEAT_SERVICE = "/arm/begin_seat"
ARM_DELIVER_SERVICE = "/arm/deliver_mic"
ARM_RETURN_SERVICE = "/arm/return_mic"
ARM_HOME_SERVICE = "/arm/home"

MOTOR_MOVE_SERVICE = "/motor/move_to_ticks"
MOTOR_DELIVER_MIKE_SERVICE = "/motor/deliver_mike"
MOTOR_PICK_PLACE_SERVICE = "/arm/execute_pick_place"
MOTOR_MIKE_GRIPPER_SERVICE = "/motor/mike_gripper"
MOTOR_TORQUE_OFF_SERVICE = "/motor/torque_off"

MOTOR_JOINT_STATE_TOPIC = "/motor/joint_state"
MOTOR_GRIPPER_STATE_TOPIC = "/motor/gripper_state"


# Detection / coordinate conversion

DEFAULT_HANDEYE_PATH = (
    "/home/seungwon/soomac_ws/"
    "src/mission_2/mission_2/hand_eye_result.json"
)

MINIMUM_CONFIDENCE = 0.83
DETECTION_WAIT_SEC = 5.0
DETECTION_SETTLE_SEC = 0.60
REQUIRE_FLANGE_FRAME = True

# floor -> robot base horn origin.
FLOOR_Z_OFFSET_M = 0.041


# Task 2 detection poses
# RIGHT basket mic detection pose.
RIGHT_MIC_DETECTION_TICKS = np.array(
    [336, 465, 6850, 3366, 2028],
    dtype=np.int64,
)

# Table mic detection.
TABLE_MIC_DETECTION_TICKS = np.array(
    [1950, -200, 7300, 3200, 2048],
    dtype=np.int64,
)


# Right basket return target
# RIGHT basket place offset [m].
RIGHT_BASKET_PLACE_FINE_OFFSET_M = np.array(
    [0.0, 0.0, -0.01],
    dtype=float,
)

# RIGHT basket place yaw offset.
RIGHT_BASKET_PLACE_YAW_OFFSET_DEG = 0.0


# Saved-hand delivery
# HAND delivery offset [m].
HAND_PLACE_FINE_OFFSET_M = np.array(
    [0.0, 0.0, 0.0],
    dtype=float,
)

# HAND approach distance for MIC_UP delivery.
HAND_APPROACH_DISTANCE_MM = 80.0

# Delay before microphone release.
HAND_RELEASE_DELAY_SEC = 3.0


# Microphone pick tuning / orientation

# Mic pick offsets [m].
RIGHT_MIC_PICK_FINE_OFFSET_M = np.array(
    [-0.01, -0.01, 0.03],
    dtype=float,
)

TABLE_MIC_PICK_FINE_OFFSET_M = np.array(
    [0.0, 0.0, 0.03],
    dtype=float,
)

DEFAULT_MIC_PICK_YAW_DEG = 0.0


# Gripper

MIC_GRIPPER_OPEN_TICK = 85
MIC_GRIPPER_CLOSE_TICK = 1400


# Direct arm motion

DIRECT_MOVE_PROFILE_VELOCITY = 8
DIRECT_MOVE_TIMEOUT_SEC = 60.0
MOTOR_SERVICE_WAIT_SEC = 10.0
MOTOR_SERVICE_TIMEOUT_SEC = 180.0

DYNAMIC_START_MAX_ERROR_TICK = 100


# Collision detection / recovery
# Collision stall detection.
WAYPOINT_TRACK_STALL_TIMEOUT_SEC = 1.0
WAYPOINT_TRACK_SAMPLE_DT_SEC = 0.10
WAYPOINT_TRACK_PROGRESS_EPS_RATIO = 0.02

# Collision recovery.
COLLISION_RECOVERY_WAIT_SEC = 2.0
COLLISION_RECOVERY_PROFILE_VELOCITY = 5
COLLISION_RECOVERY_MAX_TICK_STEP = 40
COLLISION_RECOVERY_COMMAND_DT_SEC = 0.04
COLLISION_RECOVERY_TIMEOUT_SEC = 60.0


# Planner

MAX_START_JOINT_STEP_DEG = 5.0

SAFE_ABOVE_Z_MM = 230.0
PLACE_APPROACH_HEIGHT_MM = 100.0

CARTESIAN_STEP_MM = 35.0
YAW_STEP_DEG = 12.0
MAX_J5_WAYPOINT_STEP_DEG = 20.0

BEZIER_MIN_RADIUS_MM = 320.0
BEZIER_CONTROL_Z_MM = 300.0

POSITION_ONLY = "POSITION_ONLY"
FULL_POSE = "FULL_POSE"
MIC_UP_POSE = "MIC_UP_POSE"

POSITION_TOLERANCE_MM = 2.0
TOOL_DOWN_TOLERANCE_DEG = 2.0
YAW_TOLERANCE_DEG = 2.0

POSITION_POSTURE_WEIGHT = 0.02
POSITION_MAX_NFEV = 45

FAST_FULL_POSE_MAX_NFEV = 220
FAST_FULL_POSE_POSITION_TOLERANCE_MM = 2.0
FAST_FULL_POSE_TOOL_TOLERANCE_DEG = 2.0
FAST_FULL_POSE_YAW_TOLERANCE_DEG = 2.0

# MIC_UP solver.
MIC_UP_POSITION_TOLERANCE_MM = 3.0
MIC_UP_AXIS_TOLERANCE_DEG = 3.0
MIC_UP_AXIS_WEIGHT = 10.0
MIC_UP_POSTURE_WEIGHT = 0.01
MIC_UP_MAX_NFEV = 500

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
    [
        0.0,
        7.874154,
        94.077644,
        114.397512,
        0.0,
    ],
    dtype=float,
)

# Shared planner compatibility.
BOTTLE_PLACE_APPROACH_HEIGHT_MM = 50.0
BOTTLE_SAFE_TRANSPORT_Z_MM = 330.0
BOTTLE_PICK_CLEARANCE_MM = 40.0
