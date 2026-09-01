# Task Manager의 파이프라인 이벤트 / 상태 로그 (mission 1/2/3 공용)
# 기존 String(JSON) topic을 대체:
#   /missionN/task_status, /mission2/pipeline_event

std_msgs/Header header
string mission     # "mission1" | "mission2" | "mission3"
string event       # 이벤트명 (예: "approach_start", "pick_place_request")
string detail      # 부가 정보. 자유 텍스트, 필요하면 JSON 문자열 유지 가능
