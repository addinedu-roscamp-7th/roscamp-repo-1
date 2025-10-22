import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty
import os
from shopee_interfaces.msg import PickeeMobileArrival, PickeeMobileSpeedControl, Pose2D

from pickee_mobile.test.goal_test import get_pose

class VelModifier(Node):
    """키보드 입력에 따라 scale 값을 조정하며 /cmd_vel을 수정 발행"""

    def __init__(self):
        super().__init__('vel_modifier')

        # 기본 scale 값
        self.scale = 1.0
        self.running = True

        # /cmd_vel, /pickee/mobile/speed_control 구독

        self.subscribe_speed_control = self.create_subscription(
            PickeeMobileSpeedControl, '/pickee/mobile/speed_control', self.speed_control_callback, 10
        )
        self.subscribe_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.modify_cmd_vel_callback, 10
        )

        # 수정된 /cmd_vel_modified 발행
        self.pickee_vel_publisher = self.create_publisher(Twist, '/cmd_vel_modified', 10)

    def speed_control_callback(self, msg: PickeeMobileSpeedControl):
        """PickeeMobileSpeedControl 메시지를 받아 scale 값 수정"""
        robot_id = msg.robot_id
        order_id = msg.order_id
        speed_mode = msg.speed_mode
        target_speed = msg.target_speed
        obstacles = msg.obstacles
        reason = msg.reason

        self.get_logger().info(f'🔧 Speed Control로부터 Scale 값 업데이트: {self.scale:.2f}')

    def modify_cmd_vel_callback(self, msg: Twist):
        """Twist 메시지를 현재 scale에 맞춰 수정 후 발행"""
        modified_twist = Twist()

        modified_twist.linear.x = msg.linear.x * self.scale
        modified_twist.linear.y = msg.linear.y * self.scale
        modified_twist.linear.z = msg.linear.z * self.scale
        modified_twist.angular.x = msg.angular.x * self.scale
        modified_twist.angular.y = msg.angular.y * self.scale
        modified_twist.angular.z = msg.angular.z * self.scale

        self.pickee_vel_publisher.publish(modified_twist)

        self.get_logger().info(
            f"📨 Scale={self.scale:.2f} | "
            f"Linear: ({msg.linear.x:.2f})→({modified_twist.linear.x:.2f}), "
            f"Angular: ({msg.angular.z:.2f})→({modified_twist.angular.z:.2f})"
        )


    def keyboard_input(self):
        """키 입력 감지 스레드"""
        tty.setcbreak(self.fd)
        try:
            while self.running:
                key = sys.stdin.read(1).lower()

                if key == 'z':
                    self.scale += 0.1
                elif key == 'x':
                    self.scale -= 0.1
                elif key == 'a':
                    self.scale = 0.0
                elif key == 's':
                    self.scale = 1.0
                elif key == 'c':
                    self.get_logger().info('🛑 프로그램 종료 명령(C) 입력됨')
                    self.stop_node()
                    break

                # scale 범위 제한 (0.0 ~ 2.0)
                self.scale = max(min(self.scale, 2.0), 0.0)
                self.get_logger().info(f'🔧 현재 Scale: {self.scale:.2f}')

        finally:
            self.restore_terminal()


def main(args=None):
    rclpy.init(args=args)
    node = VelModifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_node()
    finally:
        node.restore_terminal()


if __name__ == '__main__':
    main()
