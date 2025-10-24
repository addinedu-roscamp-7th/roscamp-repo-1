import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty
import os

from pickee_mobile.test.goal_test import get_pose

class TwistModifier(Node):
    """키보드 입력에 따라 scale 값을 조정하며 /cmd_vel을 수정 발행"""

    def __init__(self):
        super().__init__('twist_modifier')

        # 기본 scale 값
        self.scale = 1.0
        self.running = True

        # /cmd_vel 구독
        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.modify_cmd_vel_callback, 10
        )

        # 수정된 /cmd_vel 발행
        self.pickee_vel_publisher = self.create_publisher(Twist, '/cmd_vel_modified', 10)

        self.get_logger().info('✅ cmd_vel 노드 시작됨 (기본 scale=1.0)')

    # ---------------------------------------------------------------------
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


# -------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TwistModifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == '__main__':
    main()
