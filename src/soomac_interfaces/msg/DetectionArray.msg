# Detection node가 계속 publish하는 프레임 단위 검출 결과 (mission 1/2/3 공용)
# 기존 String(JSON) topic 대체: /mission1/detections, /mission2/detections, /mission3/detections
#
# ★ 단위 규칙: 길이는 전부 미터(m), 각도는 전부 도(deg)

std_msgs/Header header
# header.stamp    : 프레임 시각 (기존 "time" 필드 대체)
# header.frame_id : pose 기준 좌표계
#                   "flange" : 정상 (Hand-Eye 적용 완료)
#                   "camera" : Hand-Eye 로드 실패로 강등된 상태
#                              -> 제어단은 이 좌표로 pick 금지!

Detection[] detections

# ---- ArUco (mission 1, 3에서 사용. mission 2는 aruco_detected=false 고정) ----
bool aruco_detected              # 이번 프레임에서 마커 검출 여부
bool aruco_is_temp               # true면 TEMP 임시값 (좌표계 의미 없음, 절대 실사용 금지)
                                 #   -> 반드시 (aruco_detected && !aruco_is_temp) 로 검사할 것
int32 aruco_id                   # 검출된 마커 ID. 미검출/임시값이면 -1
geometry_msgs/Point aruco_pose         # header.frame_id 좌표계, 단위 m
geometry_msgs/Point aruco_camera_pose  # 디버깅용 raw 카메라 좌표, 단위 m
float32 aruco_yaw_deg            # 마커 in-plane 회전각, 단위 deg. name_tag 정렬 배치용. 미계산 시 0
