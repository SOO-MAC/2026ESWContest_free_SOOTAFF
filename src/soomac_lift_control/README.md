# SOOMAC Lift Control

SOOMAC 리프트 제어 패키지입니다.
노트북 ROS2에서 Arduino로 USB Serial 명령을 보내고, Arduino가 IBT-2 모터드라이버 2개를 이용해 리니어 액추에이터 2개를 제어합니다.

---

## 1. 시스템 구조

```text
노트북 ROS2
→ /lift/up, /lift/down 서비스 호출
→ lift_serial_server 노드
→ Arduino USB Serial
→ IBT-2 모터드라이버 2개
→ 리니어 액추에이터 2개
```

---

## 2. 제공 기능

### Service

| Service name | Type                   | 설명               |
| ------------ | ---------------------- | ---------------- |
| `/lift/up`   | `std_srvs/srv/Trigger` | 리프트 상승 / 나오는 동작  |
| `/lift/down` | `std_srvs/srv/Trigger` | 리프트 하강 / 들어가는 동작 |
| `/lift/stop` | `std_srvs/srv/Trigger` | 리프트 즉시 정지        |
| `/lift/home` | `std_srvs/srv/Trigger` | 현재는 정지만 수행       |

### Topic

| Topic name            | Type                | 설명             |
| --------------------- | ------------------- | -------------- |
| `/scout/move_enabled` | `std_msgs/msg/Bool` | Scout 이동 가능 여부 |

`/scout/move_enabled` 의미:

```text
data: true
→ 리프트가 완전히 내려간 상태
→ Scout 이동 가능

data: false
→ 리프트 상승 대기 / 상승 중 / 올라가 있음 / 하강 중 / 정지 상태
→ Scout 이동 금지
```

---

## 3. 기본 동작 시간

현재 기본값은 다음과 같습니다.

```text
신호 수신 후 대기 시간: 2초
/lift/up 동작 시간: 4초
/lift/down 동작 시간: 10초
```

즉 `/lift/up`을 호출하면:

```text
2초 대기
→ 4초 동안 리프트 상승
→ STOP 명령 전송
→ success=true 반환
```

`/lift/down`을 호출하면:

```text
2초 대기
→ 10초 동안 리프트 하강
→ STOP 명령 전송
→ success=true 반환
```

---

## 4. 하드웨어 구성

```text
Arduino UNO 1개
IBT-2 모터드라이버 2개
리니어 액추에이터 2개
12V 전원공급장치 1개
```

권장 전원:

```text
12V 15A 이상 권장
12V 20A 권장
```

리니어 모터가 1개당 약 5A라면, 2개 사용 시 최소 10A 이상이 필요합니다.
시작 순간이나 부하가 걸릴 때 전류가 더 커질 수 있으므로 여유 있는 전원을 사용합니다.

---

## 5. 배선

### Arduino → IBT-2 #1

```text
Arduino D5  → IBT-2 #1 RPWM
Arduino D6  → IBT-2 #1 LPWM
Arduino D7  → IBT-2 #1 R_EN
Arduino D8  → IBT-2 #1 L_EN
Arduino 5V  → IBT-2 #1 VCC
Arduino GND → IBT-2 #1 GND
```

### Arduino → IBT-2 #2

```text
Arduino D9   → IBT-2 #2 RPWM
Arduino D10  → IBT-2 #2 LPWM
Arduino D11  → IBT-2 #2 R_EN
Arduino D12  → IBT-2 #2 L_EN
Arduino 5V   → IBT-2 #2 VCC
Arduino GND  → IBT-2 #2 GND
```

### 12V 전원 → IBT-2 두 개

12V 전원은 IBT-2 두 개에 병렬로 공급합니다.

```text
12V 전원 +
├─ IBT-2 #1 B+
└─ IBT-2 #2 B+

12V 전원 -
├─ IBT-2 #1 B-
├─ IBT-2 #2 B-
└─ Arduino GND
```

중요:

```text
Arduino GND
IBT-2 #1 GND
IBT-2 #2 GND
12V 전원 -
```

위 GND들은 반드시 공통으로 연결해야 합니다.

### 모터 연결

```text
IBT-2 #1 M+ / M- → 리니어 모터 #1
IBT-2 #2 M+ / M- → 리니어 모터 #2
```

한쪽 모터만 반대로 움직이면 해당 모터의 `M+`, `M-` 선을 서로 바꾸면 됩니다.

---

## 6. 주의사항

절대 하면 안 되는 것:

```text
12V +를 Arduino 5V에 연결 금지
12V +를 IBT-2 VCC에 연결 금지
IBT-2 하나에 모터 2개를 같이 연결 금지
GND 공통 연결 없이 실행 금지
```

처음 테스트할 때는 반드시 리프트 주변을 비우고, 바로 전원을 끌 수 있는 상태에서 실행합니다.

리니어 액추에이터 2개는 완전히 같은 속도로 움직이지 않을 수 있습니다.
한쪽이 더 빠르면 Arduino 코드에서 해당 모터의 PWM 값을 낮춰 속도를 보정해야 합니다.

---

## 7. Arduino 코드

Arduino 코드는 다음 위치에 저장합니다.

```text
soomac_lift_control/arduino/lift_2motor/lift_2motor.ino
```

Arduino IDE에서 해당 `.ino` 파일을 열고 Arduino에 업로드합니다.

Arduino 코드의 안전 정지 시간은 다음 값으로 설정합니다.

```cpp
const unsigned long SAFETY_STOP_TIME_MS = 12000;
```

ROS2에서 10초 하강 명령을 보내므로, Arduino의 안전 정지는 그보다 긴 12초로 설정합니다.

---

## 8. 패키지 빌드

패키지가 포함된 워크스페이스로 이동합니다.

```bash
cd ~/soomac_lift_ws
```

ROS2 환경을 불러옵니다.

```bash
source /opt/ros/humble/setup.bash
```

빌드합니다.

```bash
colcon build --packages-select soomac_lift_control
```

빌드 후 환경을 불러옵니다.

```bash
source install/setup.bash
```

---

## 9. 실행 방법

### 9-1. Arduino 포트 확인

Arduino를 USB로 연결한 뒤 실행합니다.

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

일반적으로 Arduino UNO는 다음처럼 잡힙니다.

```text
/dev/ttyACM0
```

Nano 계열은 다음처럼 잡힐 수 있습니다.

```text
/dev/ttyUSB0
```

### 9-2. 포트 권한 설정

`/dev/ttyACM0`이면:

```bash
sudo chmod 666 /dev/ttyACM0
```

`/dev/ttyUSB0`이면:

```bash
sudo chmod 666 /dev/ttyUSB0
```

### 9-3. Launch 실행

기본 실행:

```bash
source /opt/ros/humble/setup.bash
cd ~/soomac_lift_ws
source install/setup.bash

ros2 launch soomac_lift_control lift.launch.py
```

기본값:

```text
port: /dev/ttyACM0
baudrate: 115200
start_delay_sec: 2.0
lift_up_sec: 4.0
lift_down_sec: 10.0
```

포트가 `/dev/ttyUSB0`이면:

```bash
ros2 launch soomac_lift_control lift.launch.py port:=/dev/ttyUSB0
```

시간을 직접 바꾸고 싶으면:

```bash
ros2 launch soomac_lift_control lift.launch.py \
  lift_up_sec:=4.0 \
  lift_down_sec:=10.0
```

---

## 10. 테스트 명령

Launch 파일을 실행한 터미널은 계속 켜둡니다.
새 터미널을 열고 아래 명령을 실행합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/soomac_lift_ws
source install/setup.bash
```

### 리프트 상승

```bash
ros2 service call /lift/up std_srvs/srv/Trigger "{}"
```

정상 동작:

```text
2초 대기
→ 4초 상승
→ 정지
→ success: true
```

### 리프트 하강

```bash
ros2 service call /lift/down std_srvs/srv/Trigger "{}"
```

정상 동작:

```text
2초 대기
→ 10초 하강
→ 정지
→ success: true
```

### 비상 정지

```bash
ros2 service call /lift/stop std_srvs/srv/Trigger "{}"
```

---

## 11. Scout 이동 가능 토픽 확인

새 터미널에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/soomac_lift_ws
source install/setup.bash

ros2 topic echo /scout/move_enabled
```

상태 의미:

```text
data: true
→ 리프트 완전 하강 상태
→ Scout 이동 가능

data: false
→ 리프트가 올라가 있거나 움직이는 중
→ Scout 이동 금지
```

---

## 12. Fake 모드

Arduino 없이 ROS2 동작만 확인할 때 사용합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/soomac_lift_ws
source install/setup.bash

ros2 launch soomac_lift_control lift.launch.py fake:=true
```

Fake 모드에서는 실제 모터가 움직이지 않고, 시간과 서비스 응답만 시뮬레이션합니다.

---

## 13. 터미널 하나로 실행하는 방법

일반적으로는 터미널 2개를 권장합니다.

터미널 1:

```bash
ros2 launch soomac_lift_control lift.launch.py
```

터미널 2:

```bash
ros2 service call /lift/up std_srvs/srv/Trigger "{}"
```

터미널 하나만 사용하려면 launch를 백그라운드로 실행합니다.

```bash
ros2 launch soomac_lift_control lift.launch.py &
```

그 후 같은 터미널에서 명령을 보낼 수 있습니다.

```bash
ros2 service call /lift/up std_srvs/srv/Trigger "{}"
ros2 service call /lift/down std_srvs/srv/Trigger "{}"
```

백그라운드 실행을 끄려면:

```bash
fg
```

그 다음 `Ctrl + C`를 누릅니다.

강제 종료:

```bash
pkill -f lift_serial_server
pkill -f "ros2 launch"
```

---

## 14. 속도 보정

두 리니어 액추에이터의 속도가 다르면 Arduino 코드에서 PWM 값을 조정합니다.

예를 들어 모터 2가 더 빠르면:

```cpp
const int M2_UP_SPEED = 230;
const int M2_DOWN_SPEED = 230;
```

모터 1이 더 빠르면:

```cpp
const int M1_UP_SPEED = 230;
const int M1_DOWN_SPEED = 230;
```

기본값은 다음과 같습니다.

```cpp
const int M1_UP_SPEED = 255;
const int M1_DOWN_SPEED = 255;

const int M2_UP_SPEED = 255;
const int M2_DOWN_SPEED = 255;
```

업 방향과 다운 방향의 속도가 다를 수 있으므로, 필요하면 따로 조정합니다.

```cpp
const int M1_UP_SPEED = 255;
const int M1_DOWN_SPEED = 230;

const int M2_UP_SPEED = 240;
const int M2_DOWN_SPEED = 255;
```

---

## 15. GitHub에 올려야 하는 파일

GitHub에는 아래 패키지 폴더를 올립니다.

```text
src/soomac_lift_control/
```

포함할 파일:

```text
soomac_lift_control/
├── package.xml
├── setup.py
├── setup.cfg
├── launch/
│   └── lift.launch.py
├── resource/
│   └── soomac_lift_control
├── soomac_lift_control/
│   ├── __init__.py
│   └── lift_serial_server.py
└── arduino/
    └── lift_2motor/
        └── lift_2motor.ino
```

올리면 안 되는 파일:

```text
build/
install/
log/
__pycache__/
*.pyc
*.zip
```

---

## 16. 팀원용 실행 요약

빌드:

```bash
cd ~/soomac_3.0-ESW
source /opt/ros/humble/setup.bash
colcon build --packages-select soomac_lift_control
source install/setup.bash
```

실행:

```bash
ros2 launch soomac_lift_control lift.launch.py
```

상승:

```bash
ros2 service call /lift/up std_srvs/srv/Trigger "{}"
```

하강:

```bash
ros2 service call /lift/down std_srvs/srv/Trigger "{}"
```

정지:

```bash
ros2 service call /lift/stop std_srvs/srv/Trigger "{}"
```

Scout 이동 가능 여부 확인:

```bash
ros2 topic echo /scout/move_enabled
```

