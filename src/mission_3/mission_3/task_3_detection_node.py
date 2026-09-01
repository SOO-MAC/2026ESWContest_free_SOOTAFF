#!/usr/bin/env python3
"""
Mission 3 Detection Node - 행사장 테이블 뒷정리

검출된 물체를 name_tag / valuables(지갑, 휴대폰) / trash(나머지 전부)로 분류해서
soomac_interfaces/DetectionArray를 /mission3/detections로 계속 publish한다.
ArUco는 name_tag를 원래 자리로 되돌릴 때의 기준 좌표로 쓴다.
/mission3/start_detect (Trigger)는 상태 리셋용.

[분류 규칙]
  실제 학습 클래스: 0 Bottle, 1 Snack_1, 2 Snack_2, 3 Snack_3, 4 name_tag, 5 Wallet, 6 Phone
  - "name_tag"                  -> CATEGORY_NAME_TAG
  - 이름에 wallet / phone 포함   -> CATEGORY_VALUABLES
  - 그 외 전부                  -> CATEGORY_TRASH
  class_id를 하드코딩하면 재학습할 때마다 깨지므로 model.names(문자열)를 읽어 분류한다.
  model.names를 못 읽는 예외 상황용으로 CLASS_ID_FALLBACK만 남겨둔다.

[좌표 규약]
  pose        : DetectionArray.header.frame_id 좌표계, 단위 m
                "flange" = 정상(Hand-Eye 적용) / "camera" = 강등(pick 금지)
  yaw_deg     : 마스크 장축 각도. 삐딱한 지갑/쓰레기를 집을 때 그리퍼 roll 정렬용

★ P_flange는 팔이 그 자세에 정지해 있는 동안만 유효하다.
  제어단은 자신이 정지한 시각 이후 header.stamp 를 가진 메시지만 써야 한다.

★ aruco_yaw_deg는 정보용이다
  place offset은 base 축 기준으로 더하기로 확정했으므로, 이 값으로 offset을
  회전시키면 안 된다.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import threading
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import Subscriber, ApproximateTimeSynchronizer

from soomac_interfaces.msg import Detection, DetectionArray


# 여기부터 비전 공용 유틸. mission_1/2/3 detection 노드에 그대로 복사돼 있으므로
# yaw나 좌표변환을 고칠 때는 세 파일을 같이 손볼 것.
#
# 단위는 REP-103을 따른다 (길이 m, 각도 deg).
# 픽셀(u,v)+depth -> P_cam 핀홀 역투영 -> P_flange Hand-Eye 적용까지가 비전단이고,
# P_base = FK(q_now) @ P_flange 와 IK는 제어단 몫이다.

# 장축/단축 비가 이 값 미만이면 방향이 무의미하다고 보고 yaw_valid=false.
# 원형/정사각형은 그리퍼를 어느 각도로 돌려도 같으니 기본 자세로 파지한다.
YAW_MIN_ASPECT_RATIO = 1.15

# 얇은 돌출부(마이크 케이블, 끈 등)를 지우는 모폴로지 opening 커널(px).
# 돌출부가 남아 있으면 minAreaRect가 그것까지 감싸느라 장축이 10도 이상 틀어진다.
# 합성 검증에서 opening 없이 평균오차 10.2도 -> 적용 시 0.7도였다.
# 돌출부 없는 물체엔 영향이 없어서 그냥 항상 켜둔다.
YAW_OPEN_PX = 13


# Hand-Eye translation 크기 sanity 범위. 실측 카메라-플랜지 거리가 ~0.036m다.
HANDEYE_MIN_DIST_M = 0.01
HANDEYE_MAX_DIST_M = 0.15


DEFAULT_WEIGHTS = "/home/seungwon/soomac_ws/src/mission_3/mission_3/best.pt"
DEFAULT_HANDEYE = "/home/seungwon/soomac_ws/src/mission_3/mission_3/hand_eye_result.json"  # calibrateHandEye 결과 (t 단위: m)

YOLO_DEVICE = 0
PROCESS_PERIOD_SEC = 0.10

NAME_TAG_CLASS_NAME = "name_tag"
VALUABLE_KEYWORDS = ("wallet", "phone")

# model.names를 못 읽는 예외 상황용 fallback (실제 학습된 클래스 순서)
CLASS_ID_FALLBACK = {
    0: "Bottle", 1: "Snack_1", 2: "Snack_2", 3: "Snack_3",
    4: "name_tag", 5: "Wallet", 6: "Phone",
}

# ArUco 미검출 시 임시값. aruco_is_temp=true 로 발행되며 좌표계 의미가 없다.
TEMP_ARUCO_POSE = [0.50, 0.0, 0.10]

ARUCO_CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
ARUCO_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

CATEGORY_COLORS = {
    Detection.CATEGORY_NAME_TAG: (255, 255, 0),
    Detection.CATEGORY_VALUABLES: (0, 215, 255),
    Detection.CATEGORY_TRASH: (0, 255, 0),
}


def classify_category(class_name: str) -> str:
    """YOLO 원본 클래스명 -> Detection.category 상수"""
    lowered = class_name.lower()
    if lowered == NAME_TAG_CLASS_NAME:
        return Detection.CATEGORY_NAME_TAG
    if any(k in lowered for k in VALUABLE_KEYWORDS):
        return Detection.CATEGORY_VALUABLES
    return Detection.CATEGORY_TRASH


def depth_image_to_meters(depth_img, encoding):
    """ROS depth image -> meter 단위 float32 배열."""
    if depth_img is None:
        return None
    if encoding in ("16UC1", "mono16") or depth_img.dtype == np.uint16:
        return depth_img.astype(np.float32) * 0.001
    return depth_img.astype(np.float32)   # 32FC1이거나, 모르는 인코딩이면 일단 float


def mask_distance_m(mask, depth_m):
    """마스크 안쪽 유효 depth에서 IQR 아웃라이어를 걷어낸 평균(m)."""
    valid = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0)]
    if len(valid) <= 10:
        return 0.0
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    pure = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    return float(np.mean(pure if len(pure) else valid))


def patch_depth_m(depth_m, px, py, radius=3):
    """한 점 주변 패치의 median depth(m). ArUco 중심처럼 마스크가 없을 때 사용."""
    h, w = depth_m.shape[:2]
    x1 = max(0, int(px) - radius)
    y1 = max(0, int(py) - radius)
    x2 = min(w, int(px) + radius + 1)
    y2 = min(h, int(py) + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def pixel_to_pose_m(px, py, dist_m, camera_info, T_flange_cam):
    """
    픽셀 + depth -> 카메라 좌표계 3D (m). 실패 시 None.
    P_cam(m) -> P_flange(m). T가 None이면 None.(호출부에서 camera 프레임 강등하고, 제어단은 그 좌표로 pick 금지)
    """
    if camera_info is None or dist_m <= 0:
        return None, None
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])
    if fx == 0.0 or fy == 0.0:
        return None, None

    z = float(dist_m)
    cam_pose = [(float(px) - cx) * z / fx, (float(py) - cy) * z / fy, z]
    if T_flange_cam is None:
        return cam_pose, cam_pose

    P = T_flange_cam @ np.array(
        [cam_pose[0], cam_pose[1], cam_pose[2], 1.0], dtype=np.float64)
    return [float(P[0]), float(P[1]), float(P[2])], cam_pose


def load_handeye(path, logger=None):
    """
    calibrateHandEye 결과 json -> 4x4 T_flange<-cam (translation 단위 m).
    json은 R_cam2gripper (3x3), t_cam2gripper (3x1, 단위 m)를 가진다고 가정.
    실패/이상 시 None -> 노드는 camera 프레임으로 강등 발행.
    """
    def log(level, msg):
        fn = getattr(logger, level, None) if logger is not None else None
        (fn or print)(msg)

    try:
        with open(path) as f:
            data = json.load(f)
        R = np.array(data["R_cam2gripper"], dtype=np.float64)
        t_m = np.array(data["t_cam2gripper"], dtype=np.float64).flatten()
    except Exception as e:
        log("error", f"[HandEye] 로드 실패: {e}")
        return None

    if R.shape != (3, 3) or t_m.shape != (3,):
        log("error", f"[HandEye] 형상 오류: R{R.shape}, t{t_m.shape}")
        return None

    # 회전행렬 유효성 (직교 + det=1)
    if abs(np.linalg.det(R) - 1.0) > 1e-3 or not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        log("error", "[HandEye] R_cam2gripper가 유효한 회전행렬이 아님 -> camera 프레임으로 발행")
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_m

    d = float(np.linalg.norm(t_m))
    log("info", f"[HandEye] 로드 완료. 카메라 위치(플랜지 기준) = "
                f"{np.round(t_m, 4).tolist()} m, 거리 {d:.3f} m")
    if not (HANDEYE_MIN_DIST_M < d < HANDEYE_MAX_DIST_M):
        log("warn", f"[HandEye] 경고: 거리 {d:.3f} m가 실측(~0.036 m)과 크게 다름. "
                    f"캘리브레이션/단위(m인지) 확인!")
    return T


def build_object_mask(poly_pts, shape_hw, close_kernel, erode_kernel, erode_iters=1):
    """YOLO seg 폴리곤 -> 이진 마스크. close로 구멍 메우고 erode로 경계 depth 노이즈를 깎는다."""
    pts = np.asarray(poly_pts, dtype=np.int32)
    if len(pts) < 3:
        return None
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if erode_iters > 0:
        mask = cv2.erode(mask, erode_kernel, iterations=erode_iters)
    return mask


def mask_yaw_deg(mask, open_px=YAW_OPEN_PX, min_aspect=YAW_MIN_ASPECT_RATIO):
    """
    세그 mask 장축 각도(deg). 반환 (yaw_deg, yaw_valid).
    cv2.minAreaRect 기반, (-90, 90] 정규화. 원형/정방형이면 (0.0, False).

    top-down scan 자세에서만 그리퍼 roll로 그대로 사용 가능하다.

    [opening] 계산 전에 얇은 돌출부를 제거한다. YAW_OPEN_PX 주석 참조.
    """
    m = (mask > 0).astype(np.uint8) * 255

    if open_px and open_px >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px), int(open_px)))
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        if cv2.countNonZero(opened) >= 50:   # 과하게 깎이면 원본 유지
            m = opened

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, False
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return 0.0, False

    (_, _), (w, h), angle = cv2.minAreaRect(c)
    if w <= 0 or h <= 0:
        return 0.0, False
    if w < h:                 # 긴 변이 h면 90도 보정
        angle += 90.0
        w, h = h, w
    if (w / max(h, 1e-6)) < min_aspect:
        return 0.0, False

    return normalize_deg(angle), True


def marker_yaw_deg(marker_corners):
    """
    ArUco 마커 in-plane 회전각(deg). TL->TR 변 방향 기준, (-90, 90] 정규화.

    정보/디버깅용으로 발행한다.
    """
    (tl_x, tl_y), (tr_x, tr_y) = marker_corners[0], marker_corners[1]
    return normalize_deg(float(np.rad2deg(np.arctan2(tr_y - tl_y, tr_x - tl_x))))


def normalize_deg(deg):
    """각도를 (-90, 90]으로 접는다."""
    deg = float(deg)
    while deg > 90.0:
        deg -= 180.0
    while deg <= -90.0:
        deg += 180.0
    return deg


def mask_centroid(mask):
    m = cv2.moments(mask)
    if not m["m00"]:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def list_to_point(v):
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class Mission3DetectionNode(Node):
    def __init__(self):
        super().__init__("mission3_detection_node")

        self.declare_parameter("weights_path", DEFAULT_WEIGHTS)
        self.declare_parameter("handeye_path", DEFAULT_HANDEYE)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("conf_thres", 0.83)
        self.declare_parameter("show_debug_window", False)
        self.declare_parameter("use_temp_aruco_if_missing", True)
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("target_aruco_id", -1)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.use_temp_aruco_if_missing = bool(self.get_parameter("use_temp_aruco_if_missing").value)
        self.aruco_dictionary = str(self.get_parameter("aruco_dictionary").value)
        self.target_aruco_id = int(self.get_parameter("target_aruco_id").value)

        # ---- Hand-Eye ----
        self.T_flange_cam = load_handeye(
            str(self.get_parameter("handeye_path").value), self.get_logger())
        self.coord_frame = "flange" if self.T_flange_cam is not None else "camera"
        if self.coord_frame == "camera":
            self.get_logger().warn(
                "Hand-Eye 미적용 -> frame='camera'로 발행. 제어단은 이 좌표로 pick 금지!")

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.synced = None
        self.processing = False
        self.last_rx_time = 0.0

        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.erode_kernel = np.ones((5, 5), np.uint8)

        weights = str(self.get_parameter("weights_path").value)
        self.get_logger().info(f"YOLO 로딩 중: {weights}")
        self.model = YOLO(weights)
        self.model_names = getattr(self.model, "names", {})
        self.get_logger().info(f"YOLO 로딩 완료. 클래스: {self.model_names}")

        self.detections_pub = self.create_publisher(DetectionArray, "/mission3/detections", 10)
        self.create_service(Trigger, "/mission3/start_detect", self.handle_start_detect)

        color_sub = Subscriber(self, Image, self.color_topic)
        depth_sub = Subscriber(self, Image, self.depth_topic)
        info_sub = Subscriber(self, CameraInfo, self.camera_info_topic)
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        self.create_timer(PROCESS_PERIOD_SEC, self.process_latest_frame)
        self.create_timer(2.0, self.check_camera_alive)

        self.get_logger().info(f"Mission3 Detection Node 시작 (frame={self.coord_frame})")

    def handle_start_detect(self, request, response):
        del request
        self.get_logger().info("start_detect 수신")
        response.success = True
        response.message = "detect active"
        return response

    def sync_callback(self, color_msg, depth_msg, info_msg):
        self.last_rx_time = time.time()
        with self.frame_lock:
            self.synced = (color_msg, depth_msg, info_msg)

    def check_camera_alive(self):
        if self.last_rx_time == 0.0:
            self.get_logger().warn("카메라 프레임 수신 없음 - realsense2_camera 실행 확인")
        elif time.time() - self.last_rx_time > 2.0:
            self.get_logger().warn("최근 2초 이상 카메라 프레임 없음")

    def class_name_from_id(self, cls_id: int) -> str:
        if isinstance(self.model_names, dict) and cls_id in self.model_names:
            return str(self.model_names[cls_id])
        if isinstance(self.model_names, list) and cls_id < len(self.model_names):
            return str(self.model_names[cls_id])
        return CLASS_ID_FALLBACK.get(cls_id, str(cls_id))

    def detect_aruco(self, color, depth_m, camera_info, draw):
        """반환: {"pose", "camera_pose", "id", "yaw_deg"} 또는 None"""
        if not hasattr(cv2, "aruco"):
            return None
        try:
            dict_id = getattr(cv2.aruco, self.aruco_dictionary)
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                params = cv2.aruco.DetectorParameters_create()
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

            if ids is None or len(corners) == 0:
                if draw is not None:
                    cv2.putText(draw, "ArUco: not detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return None

            ids_flat = ids.flatten().tolist()
            idx = 0
            if self.target_aruco_id >= 0:
                if self.target_aruco_id not in ids_flat:
                    if draw is not None:
                        cv2.putText(draw, f"ArUco: target {self.target_aruco_id} not in {ids_flat}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    return None
                idx = ids_flat.index(self.target_aruco_id)

            mc = corners[idx][0]
            cx = float(np.mean(mc[:, 0]))
            cy = float(np.mean(mc[:, 1]))

            if draw is not None:
                cv2.aruco.drawDetectedMarkers(draw, corners, ids)
                for (px, py), col, label in zip(mc, ARUCO_CORNER_COLORS, ARUCO_CORNER_LABELS):
                    cv2.circle(draw, (int(px), int(py)), 8, col, -1)
                    cv2.putText(draw, label, (int(px) + 10, int(py) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                cv2.circle(draw, (int(cx), int(cy)), 6, (255, 255, 255), -1)

            dist = patch_depth_m(depth_m, cx, cy, radius=4)
            pose, cam_pose = pixel_to_pose_m(cx, cy, dist, camera_info, self.T_flange_cam)
            if cam_pose is None:
                return None
            return {
                "pose": pose,
                "camera_pose": cam_pose,
                "id": int(ids_flat[idx]),
                "yaw_deg": marker_yaw_deg(mc),
            }
        except Exception as e:
            self.get_logger().warn(f"ArUco 처리 오류: {e}")
            return None

    def process_latest_frame(self):
        with self.frame_lock:
            if self.processing or self.synced is None:
                return
            color_msg, depth_msg, info_msg = self.synced
            self.processing = True
        try:
            self.run_inference(color_msg, depth_msg, info_msg)
        except Exception as e:
            self.get_logger().error(f"추론 루프 예외: {e}")
        finally:
            with self.frame_lock:
                self.processing = False

    def run_inference(self, color_msg, depth_msg, info_msg):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_m = depth_image_to_meters(depth_raw, depth_msg.encoding)
        except Exception as e:
            self.get_logger().error(f"image 변환 실패: {e}")
            return

        debug = color.copy() if self.show_debug_window else None

        aruco = self.detect_aruco(color, depth_m, info_msg, debug)
        aruco_is_temp = False
        if aruco is None and self.use_temp_aruco_if_missing:
            aruco = {"pose": TEMP_ARUCO_POSE, "camera_pose": None, "id": -1, "yaw_deg": 0.0}
            aruco_is_temp = True

        try:
            results = self.model.predict(color, device=YOLO_DEVICE, conf=self.conf_thres,
                                         verbose=False, retina_masks=True)
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 실패: {e}")
            return

        detections = []
        for r in results:
            if r.boxes is None or r.masks is None:
                continue
            for box, poly in zip(r.boxes, r.masks.xy):
                class_name = self.class_name_from_id(int(box.cls[0]))
                category = classify_category(class_name)

                mask = build_object_mask(poly, depth_m.shape[:2],
                                         self.close_kernel, self.erode_kernel, erode_iters=1)
                if mask is None:
                    continue

                dist = mask_distance_m(mask, depth_m)
                center = mask_centroid(mask)
                if center is None or dist <= 0:
                    continue
                cx, cy = center

                out_pose, cam_pose = pixel_to_pose_m(cx, cy, dist, info_msg,
                                                    self.T_flange_cam)
                if cam_pose is None:
                    continue

                yaw_deg, yaw_valid = mask_yaw_deg(mask)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                d = Detection()
                d.class_name = class_name
                d.category = category
                d.pose = list_to_point(out_pose)
                d.camera_pose = list_to_point(cam_pose)
                d.distance_m = float(dist)
                d.yaw_deg = float(yaw_deg)
                d.yaw_valid = bool(yaw_valid)
                d.box = [x1, y1, x2, y2]
                d.center_pixel = [cx, cy]
                d.confidence = float(box.conf[0]) if box.conf is not None else 0.0
                d.text = ""
                detections.append(d)

                if debug is not None:
                    col = CATEGORY_COLORS.get(category, (200, 200, 200))
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
                        cv2.drawContours(debug, [c], -1, col, 2)
                    cv2.rectangle(debug, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(debug, f"{category}:{class_name} {dist:.2f}m",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    cv2.circle(debug, (cx, cy), 4, (255, 0, 0), -1)
                    if yaw_valid:
                        rad = np.deg2rad(yaw_deg)
                        dx, dy = int(40 * np.cos(rad)), int(40 * np.sin(rad))
                        cv2.line(debug, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)
                        cv2.putText(debug, f"{yaw_deg:+.0f}deg", (cx + 8, cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        msg = DetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.coord_frame
        msg.detections = detections
        msg.aruco_detected = aruco is not None
        msg.aruco_is_temp = aruco_is_temp
        if aruco is not None:
            msg.aruco_id = int(aruco["id"])
            msg.aruco_pose = list_to_point(aruco["pose"])
            if aruco["camera_pose"] is not None:
                msg.aruco_camera_pose = list_to_point(aruco["camera_pose"])
            msg.aruco_yaw_deg = float(aruco["yaw_deg"])
        else:
            msg.aruco_id = -1
        self.detections_pub.publish(msg)

        if debug is not None:
            cv2.imshow("Mission3 Detection", debug)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Mission3DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()