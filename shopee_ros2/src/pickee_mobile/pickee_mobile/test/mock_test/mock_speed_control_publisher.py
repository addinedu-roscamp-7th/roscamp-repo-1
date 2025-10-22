import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileSpeedControl
import sys
import threading
import termios
import tty

class MockSpeedControlPublisher(Node):
    """
    PickeeMobileSpeedControl 토픽을 발행하는 Mock 노드.
    키보드 입력(z/x/c)에 따라 mode와 target_speed를 변경.
    """

    def __init__(self):
        super().__init__('mock_speed_control_publisher')
        self.get_logger().info('Mock Speed Control Publisher 노드가 시작되었습니다.')

        self.speed_control_publisher = self.create_publisher(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            10
        )

        # 키보드 상태 초기값
        self.key_command = 'z'  # 기본: normal

        # 키보드 입력 스레드 시작
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        # 1초마다 발행
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # 키보드 입력에 따라 speed_mode와 target_speed 결정
        if self.key_command == 'z':
            mode = 'normal'
            speed = 1.0
        elif self.key_command == 'x':
            mode = 'decelerate'
            speed = 0.5
        elif self.key_command == 'c':
            mode = 'stop'
            speed = 0.0
        else:
            mode = 'normal'
            speed = 1.0

        # 메시지 생성 및 발행
        speed_control_msg = PickeeMobileSpeedControl()
        speed_control_msg.speed_mode = mode
        speed_control_msg.target_speed = speed
        self.speed_control_publisher.publish(speed_control_msg)

        self.get_logger().info(f'Speed Control 발행: mode={mode}, speed={speed}')

    def keyboard_listener(self):
        """키보드 입력을 비동기적으로 처리"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while True:
                ch = sys.stdin.read(1)
                if ch in ['z', 'x', 'c']:
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
    node = MockSpeedControlPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
