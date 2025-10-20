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
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel_modified', 10)

        # 키보드 상태 초기값
        self.key_command = 'z'  # 기본: 그대로 전달

        # 키보드 입력 스레드 시작
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        self.get_logger().info('✅ Twist Modifier Node started (Keyboard control: z/x/c)')

    def twist_callback(self, msg: Twist):
        """Twist 메시지를 키보드 입력에 따라 수정 후 발행"""
        modified_twist = Twist()

        # 키보드 입력에 따른 배율 결정
        if self.key_command == 'x':      # 감속
            scale = 0.5
            self.get_logger().info('감속')
        elif self.key_command == 'z':    # 정상속도
            scale = 1.0
            self.get_logger().info('정상속도')
        elif self.key_command == 'c':    # 일시정지
            scale = 0.0
            self.get_logger().info('일시정지')
        else:
            scale = 1.0

        # 모든 선형 속도에 적용
        modified_twist.linear.x = msg.linear.x * scale
        modified_twist.linear.y = msg.linear.y * scale
        modified_twist.linear.z = msg.linear.z * scale

        # 모든 각속도에 적용
        modified_twist.angular.x = msg.angular.x * scale
        modified_twist.angular.y = msg.angular.y * scale
        modified_twist.angular.z = msg.angular.z * scale

        # 메시지 발행
        self.publisher_.publish(modified_twist)

        self.get_logger().info(
            f"Key={self.key_command} | Scale={scale} | "
            f"Linear=({msg.linear.x:.2f},{msg.linear.y:.2f},{msg.linear.z:.2f}) -> "
            f"({modified_twist.linear.x:.2f},{modified_twist.linear.y:.2f},{modified_twist.linear.z:.2f}) | "
            f"Angular=({msg.angular.x:.2f},{msg.angular.y:.2f},{msg.angular.z:.2f}) -> "
            f"({modified_twist.angular.x:.2f},{modified_twist.angular.y:.2f},{modified_twist.angular.z:.2f})"
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
                elif ch == 'v':  # 종료
                    self.get_logger().info("✅ 'v' pressed. Shutting down node...")
                    rclpy.shutdown()
                    break
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
