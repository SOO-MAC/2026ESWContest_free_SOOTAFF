#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import Trigger


class LiftFilmNode(Node):

    def __init__(self):
        super().__init__('lift_film_node')

        # ============================================================
        # 영상 촬영용 설정
        # 여기 숫자만 바꾸면 됨
        # ============================================================

        self.start_delay_sec = 10.0


        # ============================================================
        # Callback Group
        # ============================================================

        self.callback_group = ReentrantCallbackGroup()


        # ============================================================
        # 기존 Lift Node의 /lift/up
        # ============================================================

        self.lift_up_client = self.create_client(
            Trigger,
            '/lift/up',
            callback_group=self.callback_group
        )


        # ============================================================
        # Arm Control Node의 /arm/begin_seat
        # ============================================================

        self.begin_seat_client = self.create_client(
            Trigger,
            '/arm/begin_seat',
            callback_group=self.callback_group
        )


        # ============================================================
        # 10초 후 딱 한 번 실행
        # ============================================================

        self.start_timer = self.create_timer(
            self.start_delay_sec,
            self.start_sequence,
            callback_group=self.callback_group
        )

        self.sequence_started = False


        self.get_logger().info(
            '======================================'
        )
        self.get_logger().info(
            ' Lift Film Node Started'
        )
        self.get_logger().info(
            f' {self.start_delay_sec:.1f}초 후 /lift/up 호출'
        )
        self.get_logger().info(
            ' Lift 완료 후 /arm/begin_seat 호출'
        )
        self.get_logger().info(
            '======================================'
        )


    # ================================================================
    # 영상 촬영 시퀀스 시작
    # ================================================================

    def start_sequence(self):

        # Timer는 반복 Timer이므로 딱 한 번만 실행
        if self.sequence_started:
            return

        self.sequence_started = True
        self.start_timer.cancel()

        self.get_logger().info(
            '======================================'
        )
        self.get_logger().info(
            '10초 대기 완료 → /lift/up 호출'
        )
        self.get_logger().info(
            '======================================'
        )

        self.call_lift_up()


    # ================================================================
    # Lift UP 호출
    # ================================================================

    def call_lift_up(self):

        if not self.lift_up_client.wait_for_service(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                '/lift/up 서비스가 없습니다.'
            )
            return

        request = Trigger.Request()

        future = self.lift_up_client.call_async(
            request
        )

        future.add_done_callback(
            self.lift_up_response
        )


    # ================================================================
    # Lift UP 응답
    # ================================================================

    def lift_up_response(self, future):

        try:

            response = future.result()

        except Exception as error:

            self.get_logger().error(
                f'/lift/up 호출 오류: {error}'
            )
            return


        if not response.success:

            self.get_logger().error(
                f'Lift UP 실패: {response.message}'
            )
            return


        self.get_logger().info(
            f'Lift UP 성공: {response.message}'
        )

        self.get_logger().info(
            'Lift 상승 완료 → /arm/begin_seat 호출'
        )

        # Lift가 완전히 올라간 다음 Arm에 전달
        self.call_begin_seat()


    # ================================================================
    # Arm Begin Seat
    # ================================================================

    def call_begin_seat(self):

        if not self.begin_seat_client.wait_for_service(
            timeout_sec=5.0
        ):

            self.get_logger().error(
                '/arm/begin_seat 서비스가 없습니다.'
            )
            return


        request = Trigger.Request()

        future = self.begin_seat_client.call_async(
            request
        )

        future.add_done_callback(
            self.begin_seat_response
        )


    # ================================================================
    # Arm 응답
    # ================================================================

    def begin_seat_response(self, future):

        try:

            response = future.result()

        except Exception as error:

            self.get_logger().error(
                f'/arm/begin_seat 호출 오류: {error}'
            )
            return


        if response.success:

            self.get_logger().info(
                f'/arm/begin_seat 성공: {response.message}'
            )

        else:

            self.get_logger().error(
                f'/arm/begin_seat 실패: {response.message}'
            )


# ====================================================================
# main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = LiftFilmNode()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:

        executor.remove_node(node)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
