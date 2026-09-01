# 미션 최종/중간 결과 (mission 1/2/3 공용)
# 기존 String(JSON) topic을 대체:
#   /missionN/mission_result, /mission2/delivery_result, /mission2/return_result

std_msgs/Header header
string mission     # "mission1" | "mission2" | "mission3"
string phase       # 결과 단계: "delivery" | "return" | "mission" 등. 최종 결과면 "mission"
bool success
string message     # 사람이 읽을 설명 (실패 사유 등)
