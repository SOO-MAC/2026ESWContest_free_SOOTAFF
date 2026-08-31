# 🤖 SOOTAFF
> 컨퍼런스 운영 효율화를 위한 다목적행사 보조 스태프 로봇 개발

<br>

## 🎯 What is SOOTAFF?

> SOOTAFF는 행사 준비부터 진행 보조, 종료 후 정리까지  
> 반복적인 행사 운영 업무를 자동화하기 위한 ROS2 기반 행사 보조 로봇입니다.

<br>

## 👥 Team

| 이름 | 역할 |
| --- | --- |
| 나승원 | 팀장 / 로봇팔 제어 |
| 박기령 | 비전 / 객체 탐지 |
| 김태현 | 비전 / 객체 탐지 / 리프트 제어 |
| 박종훈 | GUI / Scout 제어 |
| 김루하 | 하드웨어 설계 |

<br>

## 📌 Why we need SOOTAFF?

> 행사장에서는 명찰과 다과 배치, 마이크 전달, 테이블 정리와 같은
> 반복적인 업무가 지속적으로 발생하며 이는 운영 인력의 부담 및 피로 증가로 이어집니다
> SOOTAFF는 이러한 업무를 하나의 로봇 시스템으로 통합하여 수행합니다.

---

### SOOTAF Overview

**SCOUT MINI + Lift + 5-DoF Manipulator + Vision + GUI**

사용자가 GUI에서 원하는 MODE를 선택하면,

`MODE 선택 → 자율주행 → 높이 조절 → 객체 인식 → 로봇팔 작업 → 결과 확인 → 복귀`

과정을 하나의 시스템에서 수행합니다.

<br>

## 🎬 Mission Description

SOOTAFF는 행사 운영 흐름에 맞춰 세 가지 Mission으로 구성됩니다.

| Mission | Stage | Task |
| :---: | :---: | --- |
| 🪪 **MODE 1** | 행사 전 | 명찰 · 다과 · 물병 자리 세팅 |
| 🎤 **MODE 2** | 행사 중 | 질문자 좌석 이동 · 마이크 전달 및 회수 |
| 🧹 **MODE 3** | 행사 후 | 테이블 정리 · 분실물 및 쓰레기 수거 |

### MODE 1 — Seat Setup 🪪

행사 시작 전 각 테이블을 순회하며 필요한 물품을 배치합니다.

`Name Tag → Snacks → Bottle`

작업 후 다시 인식하여 부족한 물품만 선택적으로 재배치합니다.

### MODE 2 — Q&A Assistant 🎤

GUI에서 질문자의 좌석을 선택하면 해당 위치로 이동합니다.

손의 위치와 거리를 인식하여 질문자에게 접근한 뒤
마이크를 전달하고, 질문 종료 후 다시 마이크를 회수합니다.

### MODE 3 — Table Cleanup 🧹

행사 종료 후 테이블을 순회하며 남아 있는 물체를 정리합니다.

`Name Tag → Valuables → Trash`

ArUco Marker를 이용해 명찰의 원래 위치를 찾고,
휴대폰·지갑 등의 귀중품과 일반 쓰레기를 구분하여 수거합니다.

---

# ✨ Key Features

### 🚗 Autonomous Navigation

SLAM 기반으로 행사장 지도를 생성하고,
Nav2를 이용하여 지정된 테이블과 좌석으로 자율주행합니다.

`SLAM Map → Table Detection → Target Pose → Nav2`

### 👁️ Vision-Based Manipulation

RealSense D435와 YOLO Segmentation을 이용하여
물체의 종류와 3차원 위치를 인식합니다.

Depth 정보와 Hand-Eye Calibration을 이용해
카메라 좌표를 로봇팔 작업 좌표로 변환합니다.

### 🦾 5-DoF Robotic Manipulator

자체 제작한 5-DoF 로봇팔을 이용하여
명찰, 다과, 물병, 마이크 등 다양한 물체를 조작합니다.

IKPy 기반 역기구학과 관절 제한을 적용하여
실행 가능한 Pick & Place 경로를 생성합니다.

### ↕️ Adaptive Lift

Dual Scissor Lift를 이용하여
테이블 높이에 맞게 로봇팔의 작업 높이를 조절합니다.

### 🛡️ Safety Interlock

로봇팔 작업 또는 Lift 상승 상태에서
SCOUT가 이동하지 않도록 이동 명령을 차단합니다.

안전 조건이 만족된 경우에만 Nav2의 이동 명령이
SCOUT에 전달됩니다.

### 🔄 Selective Recovery

작업 완료 후 객체를 다시 인식하여
실패하거나 누락된 작업만 선택적으로 다시 수행합니다.

전체 Mission을 처음부터 반복하지 않고
필요한 단계만 복구하도록 구성했습니다.

---

# 🧩 System Architecture

SOOTAFF는 ROS 2의 **Topic / Service 기반 분산 노드 구조**로 구성됩니다.

<img width="925" height="421" alt="image" src="https://github.com/user-attachments/assets/9376670a-66da-4196-ace5-1078445cf27b" />

