import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty
import os

class TwistModifier(Node):
    """Twist 메시지를 키보드 입력에 따라 수정하고 /cmd_vel_modify 로 발행"""

    def __init__(self):
        super().__init__('twist_modifier')



        # 기존 /cmd_vel 구독
        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.modify_cmd_vel_callback, 10
        )

        # 수정된 /cmd_vel 발행
        self.pickee_vel_publisher = self.create_publisher(Twist, '/cmd_vel_modify', 10)

        self.get_logger().info('cmd_vel Modifier 노드가 시작되었습니다.')

    def modify_cmd_vel_callback(self, msg: Twist):
        """Twist 메시지를 키 입력에 따라 수정 후 발행"""
        modified_twist = Twist()

        scale = 0.1

        # 속도 적용
        modified_twist.linear.x = msg.linear.x * scale
        modified_twist.linear.y = msg.linear.y * scale
        modified_twist.linear.z = msg.linear.z * scale
        modified_twist.angular.x = msg.angular.x * scale
        modified_twist.angular.y = msg.angular.y * scale
        modified_twist.angular.z = msg.angular.z * scale

        

        self.get_logger().info(
            f"Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}) -> "
            f"({modified_twist.linear.x:.2f}, {modified_twist.linear.y:.2f}), "
            f"Angular: ({msg.angular.z:.2f}) -> ({modified_twist.angular.z:.2f})"
        )

        self.pickee_vel_publisher.publish(modified_twist)

    

def main(args=None):

    rclpy.init(args=args)
    node = TwistModifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('cmd_vel Modifier 노드 종료')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
