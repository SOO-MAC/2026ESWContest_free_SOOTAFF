#!/usr/bin/env python3
"""
Mission 2 Detection Node - 마이크 전달/회수

Hand, Mic 두 클래스를 YOLO segmentation + depth로 잡아서
soomac_interfaces/DetectionArray를 /mission2/detections로 계속 publish한다.
/mission2/start_detect (Trigger)는 상태 리셋용인데 지금은 로그만 남긴다.

판단/제어는 하지 않는다. 몇 m까지 접근할지와 언제 로봇팔을 부를지는 Task Manager가,
실제 pick/place 좌표를 쓰는 건 제어단이 알아서 한다.

[좌표 규약]
  pose        : DetectionArray.header.frame_id 좌표계, 단위 m
                "flange" = 정상(Hand-Eye 적용) / "camera" = 강등(pick 금지)
  camera_pose : 디버깅용 raw 카메라 좌표, 단위 m
  distance_m  : depth 기반 카메라-물체 거리, 단위 m
  yaw_deg     : 마스크 장축 각도, deg. 삐딱한 마이크를 집을 때 그리퍼 roll 정렬용

★ P_flange는 팔이 그 자세에 정지해 있는 동안만 유효하다.
  제어단은 자신이 정지한 시각 이후 header.stamp 를 가진 메시지만 써야 한다.

★ distance_m 은 예외적으로 프레임과 무관하다
  카메라-물체 직선거리라서 팔 자세가 바뀌어도 의미가 유지된다.
  Task Manager의 0.3m 접근 제어는 이 값만 쓰므로 stamp 게이팅이 필요 없다.
  (접근 중 움직이는 것은 SCOUT이고 팔은 정지해 있다)

ArUco는 mission2에서 쓰지 않는다 -> aruco_detected=false, aruco_id=-1 고정.
"""

import threading
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber

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


DEFAULT_WEIGHTS = "/home/seungwon/soomac_ws/src/mission_2/mission_2/best.pt"
DEFAULT_HANDEYE = "/home/seungwon/soomac_ws/src/mission_2/mission_2/hand_eye_result.json"  # calibrateHandEye 결과 (t 단위: m)

YOLO_DEVICE = 0
PROCESS_PERIOD_SEC = 0.10   # 10Hz

# YOLO class_id -> (원본 클래스명, Detection.category)
CLASS_MAP = {
    0: ("hand", Detection.CATEGORY_HAND),
    1: ("mic", Detection.CATEGORY_MIC),
}


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


def normalize_deg(deg):
    """각도를 (-90, 90]으로 접는다. 장축은 180도 주기이므로."""
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


class Mission2DetectionNode(Node):
    def __init__(self):
        super().__init__("mission2_detection_node")

        self.declare_parameter("weights_path", DEFAULT_WEIGHTS)
        self.declare_parameter("handeye_path", DEFAULT_HANDEYE)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("conf_thres", 0.83)
        self.declare_parameter("show_debug_window", False)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)

        # ---- Hand-Eye ----
        self.T_flange_cam = load_handeye(
            str(self.get_parameter("handeye_path").value), self.get_logger())
        self.coord_frame = "flange" if self.T_flange_cam is not None else "camera"
        if self.coord_frame == "camera":
            self.get_logger().warn(
                "Hand-Eye 미적용 -> frame='camera'로 발행. 제어단은 이 좌표로 pick 금지")

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
        self.get_logger().info("YOLO 로딩 완료")

        self.detections_pub = self.create_publisher(DetectionArray, "/mission2/detections", 10)
        self.create_service(Trigger, "/mission2/start_detect", self.handle_start_detect)

        color_sub = Subscriber(self, Image, self.color_topic)
        depth_sub = Subscriber(self, Image, self.depth_topic)
        info_sub = Subscriber(self, CameraInfo, self.camera_info_topic)
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        self.create_timer(PROCESS_PERIOD_SEC, self.process_latest_frame)
        self.create_timer(2.0, self.check_camera_alive)

        self.get_logger().info(f"Mission2 Detection Node 시작 (frame={self.coord_frame})")

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
                mapped = CLASS_MAP.get(int(box.cls[0]))
                if mapped is None:
                    continue
                class_name, category = mapped

                mask = build_object_mask(poly, depth_m.shape[:2],
                                         self.close_kernel, self.erode_kernel, erode_iters=2)
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
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug, f"{class_name}: {dist:.2f}m", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        msg.aruco_detected = False      # mission2는 ArUco 미사용
        msg.aruco_is_temp = False
        msg.aruco_id = -1
        self.detections_pub.publish(msg)

        if debug is not None:
            cv2.imshow("Mission2 Detection", debug)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Mission2DetectionNode()
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
