#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_srvs.srv import Trigger

from soomac_interfaces.msg import MissionEvent, MissionResult
from soomac_interfaces.srv import StartMission2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# ROS 통신부
# ============================================================

class GuiRosNode(Node):
    """
    GUI가 Mission Task Manager와 통신하는 ROS2 노드.

    GUI -> Mission Managers
      /mission1/start_task      std_srvs/srv/Trigger
      /mission2/start_task      my_robot_interfaces/srv/StartMission2
      /mission2/question_end    std_srvs/srv/Trigger
      /mission3/start_task      std_srvs/srv/Trigger

    Mission Managers -> GUI
      /mission1/mission_result  std_msgs/msg/Bool
      /mission2/mission_result  std_msgs/msg/Bool
      /mission3/mission_result  std_msgs/msg/Bool
    """

    def __init__(self) -> None:
        super().__init__("gui")

        self.mission1_client = self.create_client(
            Trigger, "/mission1/start_task"
        )
        self.mission2_client = self.create_client(
            StartMission2, "/mission2/start_task"
        )
        self.question_end_client = self.create_client(
            Trigger, "/mission2/question_end"
        )
        self.mission3_client = self.create_client(
            Trigger, "/mission3/start_task"
        )

        self.create_subscription(
            MissionResult,
            "/mission1/mission_result",
            lambda msg: self._result_callback(1, msg),
            10,
        )
        self.create_subscription(
            MissionResult,
            "/mission2/mission_result",
            lambda msg: self._result_callback(2, msg),
            10,
        )
        self.create_subscription(
            MissionResult,
            "/mission3/mission_result",
            lambda msg: self._result_callback(3, msg),
            10,
        )

        self.create_subscription(
            MissionEvent,
            "/mission2/task_status",
            self._mission2_status_callback,
            10,
        )

        self.result_handler: Optional[Callable[[int, bool], None]] = None
        self.mission2_status_handler: Optional[Callable[[str, str], None]] = None
        self._pending_futures = set()

        self.get_logger().info("GUI ROS node ready.")

    def _result_callback(
        self,
        mission: int,
        msg: MissionResult,
    ) -> None:

        # Mission2는 delivery / return 결과도 같은 topic으로 보내므로
        # 전체 미션 종료 결과만 GUI 완료 처리한다.
        if msg.phase != "mission":
            return

        success = bool(msg.success)

        self.get_logger().info(
            f"Mission {mission} result received: "
            f"success={success}, message={msg.message}"
        )

        if self.result_handler is not None:
            self.result_handler(mission, success)

    def _mission2_status_callback(
        self,
        msg: MissionEvent,
    ) -> None:

        if self.mission2_status_handler is not None:
            self.mission2_status_handler(
                str(msg.event),
                str(msg.detail),
            )

    def _track_future(
        self,
        future,
        label: str,
        done_callback: Callable[[bool, str], None],
    ) -> None:
        self._pending_futures.add(future)

        def _done(f):
            self._pending_futures.discard(f)
            try:
                response = f.result()
                success = bool(response.success)
                message = str(response.message)
            except Exception as error:  # noqa: BLE001
                success = False
                message = f"{label} 서비스 오류: {error}"

            self.get_logger().info(
                f"{label} response: success={success}, message={message}"
            )
            done_callback(success, message)

        future.add_done_callback(_done)

    def start_mission1(
        self,
        done_callback: Callable[[bool, str], None],
    ) -> None:
        if not self.mission1_client.service_is_ready():
            done_callback(False, "/mission1/start_task 서비스가 아직 없습니다.")
            return

        future = self.mission1_client.call_async(Trigger.Request())
        self._track_future(future, "mission1/start_task", done_callback)

    def start_mission2(
        self,
        table_id: int,
        done_callback: Callable[[bool, str], None],
    ) -> None:
        if not self.mission2_client.service_is_ready():
            done_callback(False, "/mission2/start_task 서비스가 아직 없습니다.")
            return

        request = StartMission2.Request()
        request.table_id = int(table_id)

        future = self.mission2_client.call_async(request)
        self._track_future(
            future,
            f"mission2/start_task table_id={table_id}",
            done_callback,
        )

    def end_question(
        self,
        done_callback: Callable[[bool, str], None],
    ) -> None:
        if not self.question_end_client.service_is_ready():
            done_callback(False, "/mission2/question_end 서비스가 아직 없습니다.")
            return

        future = self.question_end_client.call_async(Trigger.Request())
        self._track_future(future, "mission2/question_end", done_callback)

    def start_mission3(
        self,
        done_callback: Callable[[bool, str], None],
    ) -> None:
        if not self.mission3_client.service_is_ready():
            done_callback(False, "/mission3/start_task 서비스가 아직 없습니다.")
            return

        future = self.mission3_client.call_async(Trigger.Request())
        self._track_future(future, "mission3/start_task", done_callback)

    def service_status(self) -> dict[str, bool]:
        return {
            "M1": self.mission1_client.service_is_ready(),
            "M2": self.mission2_client.service_is_ready(),
            "Q-END": self.question_end_client.service_is_ready(),
            "M3": self.mission3_client.service_is_ready(),
        }


# ============================================================
# 원형 로딩 위젯
# ============================================================

class Spinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(35)
        self.setFixedSize(90, 90)

    def rotate(self):
        self.angle = (self.angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)

        for i in range(12):
            alpha = 255 - (i * 18)
            color = QColor(255, 255, 255)
            color.setAlpha(max(alpha, 40))

            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(-5, -34, 10, 10)
            painter.rotate(30)


# ============================================================
# 메인 GUI
# ============================================================

class RobotGUI(QStackedWidget):
    def __init__(self, ros_node: GuiRosNode):
        super().__init__()

        self.ros_node = ros_node
        self.ros_node.result_handler = self.handle_mission_result
        self.ros_node.mission2_status_handler = self.handle_mission2_status

        self.active_mission: Optional[int] = None
        self.selected_table_id: Optional[int] = None
        self.question_end_sent = False

        self.resize(430, 900)
        self.setWindowTitle("행사 보조 로봇")

        self.setStyleSheet("""
            QWidget {
                background: #73B8BE;
                font-family: '맑은 고딕';
            }

            QLabel {
                color: white;
            }

            QPushButton {
                background: white;
                border: none;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
                color: #222222;
            }

            QPushButton:hover {
                background: #EFEFEF;
            }

            QPushButton:disabled {
                background: #D9D9D9;
                color: #888888;
            }
        """)

        self._build_main_page()
        self._build_work_page()
        self._build_seat_page()

        self.addWidget(self.page1)
        self.addWidget(self.page2)
        self.addWidget(self.page3)
        self.setCurrentWidget(self.page1)

        # 버튼 연결
        self.mode1.clicked.connect(self.start_mission1)
        self.mode2.clicked.connect(
            lambda: self.setCurrentWidget(self.page3)
        )
        self.mode3.clicked.connect(self.start_mission3)

        self.work_action_button.clicked.connect(
            self.handle_work_action_button
        )
        self.back_from_seats.clicked.connect(
            lambda: self.setCurrentWidget(self.page1)
        )

        # 일요일 통신 테스트 때 서버가 살아있는지 GUI에서 바로 보이게 함.
        self.connection_timer = QTimer(self)
        self.connection_timer.timeout.connect(self.update_connection_status)
        self.connection_timer.start(1000)
        self.update_connection_status()

    # --------------------------------------------------------
    # 공통 UI helper
    # --------------------------------------------------------

    @staticmethod
    def _shadow(widget, blur=20, y_offset=5):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, y_offset)
        shadow.setColor(QColor(0, 0, 0, 80))
        widget.setGraphicsEffect(shadow)

    @staticmethod
    def _center_widget(layout, widget):
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(widget)
        row.addStretch()
        layout.addLayout(row)

    def _logo_widget(self, size=420):
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)

        candidates = []

        try:
            share_dir = Path(get_package_share_directory("gui"))
            candidates.append(share_dir / "images" / "logo.png")
        except Exception:  # noqa: BLE001
            pass

        candidates.extend([
            Path.cwd() / "images" / "logo.png",
            Path(__file__).resolve().parent.parent / "images" / "logo.png",
            Path(__file__).resolve().parent / "images" / "logo.png",
        ])

        pixmap = None
        for candidate in candidates:
            if candidate.is_file():
                test = QPixmap(str(candidate))
                if not test.isNull():
                    pixmap = test
                    break

        if pixmap is not None:
            pixmap = pixmap.scaled(
                size,
                size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            logo.setPixmap(pixmap)
        else:
            logo.setText("手 MAC")
            logo.setStyleSheet("""
                font-size: 42px;
                font-weight: bold;
                color: white;
            """)

        return logo

    def _show_error(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    # --------------------------------------------------------
    # Main page
    # --------------------------------------------------------

    def _build_main_page(self):
        self.page1 = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addSpacing(25)

        layout.addWidget(self._logo_widget(450))
        layout.addSpacing(25)

        self.mode1 = QPushButton("MODE 1 (setting)")
        self.mode2 = QPushButton("MODE 2 (Q&&A)")
        self.mode3 = QPushButton("MODE 3 (clean)")

        for button in (self.mode1, self.mode2, self.mode3):
            button.setFixedSize(300, 75)
            self._shadow(button)
            self._center_widget(layout, button)
            layout.addSpacing(18)

        layout.addStretch()

        # ROS 연결 상태는 내부적으로만 확인하고 GUI에는 표시하지 않는다.
        self.connection_label = QLabel()
        self.connection_label.hide()

        self.page1.setLayout(layout)

    # --------------------------------------------------------
    # Work page
    # --------------------------------------------------------

    def _build_work_page(self):
        self.page2 = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addSpacing(25)

        layout.addWidget(self._logo_widget(450))

        self.work_text = QLabel("현재 로봇이 작업중입니다")
        self.work_text.setAlignment(Qt.AlignCenter)
        self.work_text.setWordWrap(True)
        self.work_text.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: white;
        """)

        layout.addSpacing(15)
        layout.addWidget(self.work_text)
        layout.addSpacing(20)

        self.detail_text = QLabel("")
        self.detail_text.setAlignment(Qt.AlignCenter)
        self.detail_text.setWordWrap(True)
        self.detail_text.setStyleSheet("""
            font-size: 15px;
            color: white;
        """)
        layout.addWidget(self.detail_text)
        layout.addSpacing(25)

        spinner = Spinner()
        self._center_widget(layout, spinner)

        layout.addStretch()

        self.work_action_button = QPushButton("작업 중")
        self.work_action_button.setFixedSize(250, 70)
        self.work_action_button.setEnabled(False)
        self._shadow(self.work_action_button)
        self._center_widget(layout, self.work_action_button)

        layout.addSpacing(35)
        self.page2.setLayout(layout)

    # --------------------------------------------------------
    # Seat page
    # --------------------------------------------------------

    def _build_seat_page(self):
        self.page3 = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addSpacing(20)

        layout.addWidget(self._logo_widget(390))
        layout.addSpacing(12)

        title = QLabel("로봇이 이동할 좌석을 선택해주세요")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: white;
        """)
        layout.addWidget(title)
        layout.addSpacing(22)

        self.seats = []
        big_layout = QGridLayout()
        big_layout.setHorizontalSpacing(25)
        big_layout.setVerticalSpacing(25)

        number = 1

        for block_row in range(2):
            for block_col in range(4):
                block = QGridLayout()
                block.setSpacing(5)

                for row in range(2):
                    for col in range(2):
                        table_id = number

                        button = QPushButton(str(table_id))
                        button.setFixedSize(40, 40)
                        button.setStyleSheet("""
                            QPushButton {
                                background: white;
                                border-radius: 10px;
                                font-size: 13px;
                                font-weight: bold;
                            }
                            QPushButton:hover {
                                background: #EFEFEF;
                            }
                        """)

                        self._shadow(button, blur=10, y_offset=2)

                        button.clicked.connect(
                            lambda checked=False, tid=table_id:
                            self.start_mission2(tid)
                        )

                        self.seats.append(button)
                        block.addWidget(button, row, col)
                        number += 1

                big_layout.addLayout(
                    block,
                    block_row,
                    block_col,
                )

        layout.addLayout(big_layout)
        layout.addStretch()

        self.back_from_seats = QPushButton("뒤로")
        self.back_from_seats.setFixedSize(250, 70)
        self._shadow(self.back_from_seats)
        self._center_widget(layout, self.back_from_seats)

        layout.addSpacing(35)
        self.page3.setLayout(layout)

    # --------------------------------------------------------
    # ROS service 호출
    # --------------------------------------------------------

    def start_mission1(self):
        self.active_mission = 1
        self.selected_table_id = None
        self.question_end_sent = False

        self.work_text.setText("MODE 1\n작업을 시작합니다")
        self.detail_text.clear()
        self.work_action_button.setText("작업 중")
        self.work_action_button.setEnabled(False)
        self.setCurrentWidget(self.page2)

        self.ros_node.start_mission1(
            lambda success, message:
            self._start_response(1, success, message)
        )

    def start_mission2(self, table_id: int):
        self.active_mission = 2
        self.selected_table_id = table_id
        self.question_end_sent = False

        self.work_text.setText(
            f"MODE 2\n{table_id}번 좌석으로 마이크를 전달 중입니다"
        )
        self.detail_text.clear()
        self.work_action_button.setText("질문 종료")
        self.work_action_button.setEnabled(False)
        self.setCurrentWidget(self.page2)

        self.ros_node.start_mission2(
            table_id,
            lambda success, message:
            self._start_response(2, success, message),
        )

    def start_mission3(self):
        self.active_mission = 3
        self.selected_table_id = None
        self.question_end_sent = False

        self.work_text.setText("MODE 3\n수거 작업을 시작합니다")
        self.detail_text.clear()
        self.work_action_button.setText("작업 중")
        self.work_action_button.setEnabled(False)
        self.setCurrentWidget(self.page2)

        self.ros_node.start_mission3(
            lambda success, message:
            self._start_response(3, success, message)
        )

    def _start_response(
        self,
        mission: int,
        success: bool,
        message: str,
    ):
        # 사용자가 다른 화면/미션으로 넘어간 뒤 늦은 응답이 오면 무시.
        if self.active_mission != mission:
            return

        if not success:
            self.work_text.setText(
                f"MODE {mission}\n작업 시작 실패"
            )
            self.detail_text.setText(
                "작업을 시작할 수 없습니다."
            )
            self.work_action_button.setText("홈")
            self.work_action_button.setEnabled(True)

            self._show_error(
                f"MODE {mission}",
                "작업을 시작할 수 없습니다.",
            )
            return

        # 서비스 응답의 내부 메시지는 사용자 화면에 표시하지 않는다.
        self.detail_text.clear()

        if mission == 1:
            self.work_text.setText("MODE 1\n세팅 작업 중입니다")
            self.work_action_button.setText("작업 중")
            self.work_action_button.setEnabled(False)

        elif mission == 2:
            self.work_text.setText(
                f"MODE 2\n{self.selected_table_id}번 좌석으로 마이크를 전달 중입니다"
            )
            self.work_action_button.setText("질문 종료")
            self.work_action_button.setEnabled(False)

        elif mission == 3:
            self.work_text.setText("MODE 3\n수거 작업 중입니다")
            self.work_action_button.setText("작업 중")
            self.work_action_button.setEnabled(False)

    def handle_work_action_button(self):
        # Mission 2 진행 중에는 질문 종료 버튼 역할.
        if (
            self.active_mission == 2
            and not self.question_end_sent
        ):
            self.question_end_sent = True
            self.work_action_button.setEnabled(False)
            self.work_action_button.setText("수거 중")
            self.work_text.setText("질문이 종료되었습니다")
            self.detail_text.setText(
                "마이크 수거를 요청하는 중입니다."
            )

            self.ros_node.end_question(
                self._question_end_response
            )
            return

        # 완료/실패 뒤에는 메인으로 돌아가는 버튼 역할.
        self.active_mission = None
        self.selected_table_id = None
        self.question_end_sent = False
        self.setCurrentWidget(self.page1)

    def _question_end_response(
        self,
        success: bool,
        message: str,
    ):
        if self.active_mission != 2:
            return

        if not success:
            # 실패 후 질문 종료 버튼에 갇히지 않고
            # 바로 메인 화면으로 돌아갈 수 있게 한다.
            self.question_end_sent = False
            self.active_mission = None
            self.selected_table_id = None

            self.work_text.setText("질문 종료 요청 실패")
            self.detail_text.setText(
                "질문 종료 요청을 처리할 수 없습니다."
            )

            self.work_action_button.setText("홈")
            self.work_action_button.setEnabled(True)
            return

        self.work_text.setText("MODE 2\n마이크를 수거 중입니다")
        self.detail_text.setText(
            message or "질문 종료 신호가 전달되었습니다."
        )
        self.work_action_button.setText("수거 중")
        self.work_action_button.setEnabled(False)

    # --------------------------------------------------------
    # Mission result topic
    # --------------------------------------------------------

    def handle_mission2_status(
        self,
        event: str,
        detail: str,
    ):
        if self.active_mission != 2:
            return

        # 정상 진행 중에는 내부 상태 문구를 중복 표시하지 않는다.
        self.detail_text.clear()

        if event == "question_wait":
            self.work_text.setText(
                f"MODE 2\n{self.selected_table_id}번 좌석 Q&A 진행 중"
            )
            self.work_action_button.setText("질문 종료")
            self.work_action_button.setEnabled(True)

        elif event == "question_end":
            self.work_text.setText("MODE 2\n마이크를 수거 중입니다")
            self.work_action_button.setText("수거 중")
            self.work_action_button.setEnabled(False)

        elif (
            "fail" in event.lower()
            or "error" in event.lower()
        ):
            user_detail = str(detail)

            # 사용자 화면에서는 Scout라는 내부 플랫폼 이름을 쓰지 않는다.
            user_detail = user_detail.replace("SCOUT", "로봇")
            user_detail = user_detail.replace("Scout", "로봇")
            user_detail = user_detail.replace("scout", "로봇")

            # 내부 Mission 용어도 사용자에게 노출하지 않는다.
            user_detail = user_detail.replace("Mission 1", "작업")
            user_detail = user_detail.replace("Mission 2", "작업")
            user_detail = user_detail.replace("Mission 3", "작업")
            user_detail = user_detail.replace("Mission1", "작업")
            user_detail = user_detail.replace("Mission2", "작업")
            user_detail = user_detail.replace("Mission3", "작업")
            user_detail = user_detail.replace("mission1", "작업")
            user_detail = user_detail.replace("mission2", "작업")
            user_detail = user_detail.replace("mission3", "작업")

            self.detail_text.setText(user_detail)

    def handle_mission_result(
        self,
        mission: int,
        success: bool,
    ):
        if self.active_mission != mission:
            return

        if success:
            self.work_text.setText(
                f"MODE {mission}\n작업이 완료되었습니다"
            )
        else:
            self.work_text.setText(
                f"MODE {mission}\n작업에 실패했습니다"
            )

        # ROS topic / Mission 같은 내부 정보는 사용자 화면에 표시하지 않는다.
        self.detail_text.clear()

        self.work_action_button.setText("홈")
        self.work_action_button.setEnabled(True)

    # --------------------------------------------------------
    # 연결 상태
    # --------------------------------------------------------

    def update_connection_status(self):
        status = self.ros_node.service_status()

        parts = []
        for name in ("M1", "M2", "Q-END", "M3"):
            mark = "✓" if status[name] else "×"
            parts.append(f"{name} {mark}")

        self.connection_label.setText(
            "ROS 통신: " + "   ".join(parts)
        )


# ============================================================
# 실행
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    ros_node = GuiRosNode()
    app = QApplication(sys.argv)
    window = RobotGUI(ros_node)
    window.show()

    # Qt event loop 안에서 ROS callback도 함께 처리한다.
    # 서비스 응답/mission_result topic 때문에 필요하다.
    ros_spin_timer = QTimer()
    ros_spin_timer.timeout.connect(
        lambda: rclpy.spin_once(
            ros_node,
            timeout_sec=0.0,
        )
    )
    ros_spin_timer.start(20)

    exit_code = 0

    try:
        exit_code = app.exec_()
    finally:
        ros_spin_timer.stop()
        ros_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    main()
