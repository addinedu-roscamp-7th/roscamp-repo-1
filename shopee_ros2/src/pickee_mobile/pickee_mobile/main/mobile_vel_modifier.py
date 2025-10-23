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
        self.get_logger().info('🚀 VelModifier 노드 시작 🚀')

        # 기본 scale 값
        self.scale = 1.0
        self.running = True

        # /cmd_vel, /pickee/mobile/speed_control 구독
        self.speed_control_subscriber = self.create_subscription(
            PickeeMobileSpeedControl, '/pickee/mobile/speed_control', self.speed_control_callback, 10
        )
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.modify_cmd_vel_callback, 10
        )

        # 수정된 /cmd_vel_modified 발행
        self.vel_modified_publisher = self.create_publisher(
            Twist, '/cmd_vel_modified', 10
        )


    def speed_control_callback(self, msg: PickeeMobileSpeedControl):
        """PickeeMobileSpeedControl 메시지를 받아 scale 값 수정"""
        robot_id = msg.robot_id
        order_id = msg.order_id
        speed_mode = msg.speed_mode
        target_speed = msg.target_speed
        obstacles = msg.obstacles
        reason = msg.reason

        self.scale = target_speed

        self.get_logger().info(f"{msg}")


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
