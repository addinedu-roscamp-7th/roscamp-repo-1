import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty

class TwistModifier(Node):
    """Twist 메시지를 키보드 입력에 따라 수정하고 /cmd_vel 발행"""

    def __init__(self):
        super().__init__('twist_modifier')

        # 기존 /cmd_vel 토픽 구독
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.twist_callback,
            10
        )

        # 수정된 /cmd_vel 토픽 발행
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

    def twist_callback(self, msg: Twist):
        """Twist 메시지를 키보드 입력에 따라 수정 후 발행"""
        modified_twist = Twist()

        

        # 모든 선형 속도에 적용
        modified_twist.linear.x = msg.linear.x
        modified_twist.linear.y = msg.linear.y 
        modified_twist.linear.z = msg.linear.z 

        # 모든 각속도에 적용
        modified_twist.angular.x = msg.angular.x 
        modified_twist.angular.y = msg.angular.y 
        modified_twist.angular.z = msg.angular.z 

        # 메시지 발행
        self.publisher_.publish(modified_twist)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TwistModifier()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
