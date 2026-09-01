# 검출된 물체 1개 (mission 1/2/3 공용)
#
# ★ 단위 규칙: 길이는 전부 미터(m), 각도는 전부 도(deg)
#   ROS 2 표준(REP-103)을 따른다. cm를 쓰는 필드는 하나도 없다.

# category 상수 - 오타 방지용. 미션별로 쓰는 값만 사용
#   mission1: NAME_TAG, SNACK, BOTTLE
#   mission2: HAND, MIC
#   mission3: NAME_TAG, TRASH, VALUABLES
string CATEGORY_HAND="hand"
string CATEGORY_MIC="mic"
string CATEGORY_NAME_TAG="name_tag"
string CATEGORY_SNACK="snack"
string CATEGORY_BOTTLE="bottle"
string CATEGORY_TRASH="trash"
string CATEGORY_VALUABLES="valuables"

string class_name         # YOLO 원본 클래스명 (예: "Snack_1", "Wallet", "hand")
string category           # 미션 단위 분류 (위 상수 중 하나)

# ---- 좌표 (단위: m) ----
geometry_msgs/Point pose        # DetectionArray.header.frame_id 좌표계 (정상 시 "flange"), 단위 m
geometry_msgs/Point camera_pose # 디버깅용 raw 카메라 좌표, 단위 m
float32 distance_m              # depth 기반 물체까지 거리, 단위 m

# ---- 그리퍼 회전각 (삐딱한 물체 집기용) ----
# - 물체 mask 장축(principal axis)이 이미지 x축과 이루는 각도, 단위 deg, 범위 (-90, 90]
# - 제어단: 그리퍼 roll을 이 각도에 정렬하면 장축과 수직으로 파지
# - yaw_valid == false 이면 각도 계산 불가(원형 물체 등) -> 기본 자세로 파지
float32 yaw_deg
bool yaw_valid

# ---- 픽셀 정보 (디버깅/시각화용) ----
int32[4] box              # bbox [x1, y1, x2, y2] (pixel)
int32[2] center_pixel     # mask 무게중심 [cx, cy] (pixel)
float32 confidence        # YOLO conf (0.0 ~ 1.0)

string text               # name_tag OCR 결과. 없으면 빈 문자열
