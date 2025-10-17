import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty

class TwistModifier(Node):
    """Twist 메시지를 키보드 입력에 따라 수정하고 /cmd_vel_modified로 발행"""

    def __init__(self):
        super().__init__('twist_modifier')

        # 기존 /cmd_vel 토픽 구독
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.twist_callback,
            10
        )

        # 수정된 /cmd_vel_modified 토픽 발행
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel_modified', 10)

        # 키보드 상태 초기값
        self.key_command = 'z'  # 기본: 그대로 전달

        # 키보드 입력 스레드 시작
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        self.get_logger().info('✅ Twist Modifier Node started (Keyboard control: z/x/c)')

    def twist_callback(self, msg: Twist):
        """Twist 메시지를 키보드 입력에 따라 수정 후 발행"""
        modified_twist = Twist()

        # 키보드 입력 처리
        if self.key_command == 'x':
            self.get_logger().info('감속')
            modified_twist.linear.x = msg.linear.x * 0.5
        elif self.key_command == 'z':
            f.get_logger().info('정상속도')
            modified_twist.linear.x = msg.linear.x
        elif self.key_command == 'c':
            f.get_logger().info('일시정지')
            modified_twist.linear.x = 0.0
        else:
            modified_twist.linear.x = msg.linear.x  # 기본

        # y, z 선형 속도는 그대로
        modified_twist.linear.y = msg.linear.y
        modified_twist.linear.z = msg.linear.z

        # 각속도는 그대로 전달
        modified_twist.angular.x = msg.angular.x
        modified_twist.angular.y = msg.angular.y
        modified_twist.angular.z = msg.angular.z

        # 메시지 발행
        self.publisher_.publish(modified_twist)

        self.get_logger().info(
            f"Key={self.key_command} | Original linear.x={msg.linear.x:.2f} → Modified linear.x={modified_twist.linear.x:.2f}"
        )

    def keyboard_listener(self):
        """키보드 입력을 비동기적으로 처리"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while True:
                ch = sys.stdin.read(1)
                if ch in ['x', 'z', 'c']:
                    self.key_command = ch
                    self.get_logger().info(f"Keyboard pressed: {ch}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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
