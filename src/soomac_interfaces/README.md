# soomac_interfaces

SOOMAC mission 1/2/3 공용 메시지·서비스 정의.
**비전단과 제어단 사이의 계약서**입니다. 양쪽 모두 이 패키지만 의존하고, 서로를 직접 의존하지 않는다.

```
soomac_vision_*  ─┐
                  ├─→  soomac_interfaces
soomac_arm       ─┘
```

---

## 커스텀 메세지 사용

일단은 적용 코드 안 올린 상태고, 머지하기 전에 제어가 확인하고 코드 작성하라고 정리해둔 커스텀 메세지임.
비전단은 머지 전에 적용 코드 올릴 예정이다.

**커스텀 메세지 적용 코드 작성 시** = 코드에 'soomac_interfaces' import 하면!!
그 패키지의 package.xml에 의존성을 반드시 선언해야 한다.

```
<!-- soomac_vision_mission1/package.xml -->
<depend>soomac_interfaces</depend>
```
이 과정을 빼먹으면 colcon이 빌드 순서를 몰라서 터질 수 있다.
(병렬 빌드기 때문.)


---

## ★ 단위 규칙 (반드시 지킬 것)

| 종류 | 단위 |
|---|---|
| 길이 / 좌표 | **미터 (m)** |
| 각도 | **도 (deg)** |

ROS 2 표준(REP-103)을 따릅니다. (m 쓴다는 뜻)

> 상수 offset을 정의할 때도 반드시 m로 쓸 것.
> `[0.0, -10.0, 0.0]` (cm 의도) 를 m 좌표에 더하면 로봇이 10 m 밖으로 가게 됨.
> 올바른 값: `[0.0, -0.10, 0.0]`

---

## 좌표계 (frame_id)

`DetectionArray.header.frame_id` 값으로 구분합니다.

| 값 | 의미 | 제어단 동작 |
|---|---|---|
| `flange` | 정상. Hand-Eye 적용 완료 | pick 수행 가능 |
| `camera` | Hand-Eye 로드 실패로 강등됨 | **pick 금지.** 좌표계 의미 없음 |

---

## ArUco 사용 시 주의

`aruco_pose`를 쓰기 전에 **반드시 두 플래그를 함께** 검사할 것.

```python
if msg.aruco_detected and not msg.aruco_is_temp:
    use(msg.aruco_pose)
```

`aruco_is_temp == true`는 TEMP 임시값이라 좌표계 의미가 없다.
`aruco_detected`만 보고 쓰면 임시값이 진짜 좌표처럼 캐시됨.

---

## 메시지 목록

| 파일 | 용도 |
|---|---|
| `msg/Detection.msg` | 검출된 물체 1개 |
| `msg/DetectionArray.msg` | 프레임 단위 검출 결과 + ArUco (10Hz publish) |
| `msg/MissionEvent.msg` | Task Manager 상태 로그 |
| `msg/MissionResult.msg` | 미션 최종/중간 결과 |

## 서비스 목록

| 파일 | 용도 |
|---|---|
| `srv/DetectTarget.srv` | 특정 category의 최신 검출 1건 요청 |
| `srv/PickPlace.srv` | **pick & place 요청 (좌표 포함)** |

### PickPlace를 쓰는 이유

`/arm(제어단, 이름은 알아서)/*` 서비스를 `std_srvs/Trigger`(payload 없음)로 두면, 제어단이 `/detections`를
따로 구독해서 "호출 시점에 마지막으로 받은 좌표"로 IK를 푸는데

이 경우 **비전이 판단한 물체와 제어가 집는 물체가 서로 다른 프레임의 값**일 수 있다.

`PickPlace.srv`처럼 요청에 좌표를 실으면 이 문제가 원천 차단되고,
서비스 로그에 좌표가 남아 사후 디버깅도 가능합니다.

---

## 빌드 & 확인

```bash
colcon build --packages-select soomac_interfaces
source install/setup.bash

ros2 interface show soomac_interfaces/msg/Detection
ros2 interface show soomac_interfaces/srv/PickPlace
ros2 interface list | grep soomac
```

## 사용 예

```python
from soomac_interfaces.msg import Detection, DetectionArray
from soomac_interfaces.srv import PickPlace

# category는 상수로 비교 (오타 방지)
if det.category == Detection.CATEGORY_NAME_TAG:
    ...
```

**`package.xml`에 의존성 추가 잊지 말 것!**

```xml
<depend>soomac_interfaces</depend>
```

---

## 변경 이력

| 버전 | 내용 |
|---|---|
| 0.3.0 | 단위 주석 cm → m 정정 (코드는 원래 m). `PickPlace.srv` 추가 |
| 0.2.0 | 초기 정의 |

> 이 패키지를 수정하면 비전·제어 양쪽이 재빌드해야 함.
> 필드 변경 시 반드시 공유해주기...
