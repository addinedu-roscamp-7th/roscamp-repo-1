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
    키보드 입력(z/x/c)에 따라 mode와 target_speed를 변경하고,
    키를 누를 때마다 즉시 발행합니다.
    """

    def __init__(self):
        super().__init__('mock_speed_control_publisher')
        self.get_logger().info('Mock Speed Control Publisher 노드가 시작되었습니다.')

        # 토픽 퍼블리셔 생성
        self.speed_control_publisher = self.create_publisher(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            10
        )

        # 키보드 입력 스레드 시작
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def keyboard_listener(self):
        """키보드 입력 감지 후, 입력 시 바로 메시지 발행"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while True:
                ch = sys.stdin.read(1)

                if ch == 'z':
                    mode, speed = 'normal', 1.0
                elif ch == 'x':
                    mode, speed = 'decelerate', 0.5
                elif ch == 'c':
                    mode, speed = 'stop', 0.0
                elif ch == 'v':  # 종료 키
                    self.get_logger().info("✅ 'v' pressed. Shutting down node...")
                    rclpy.shutdown()
                    break
                else:
                    continue  # 다른 키는 무시

                # 메시지 생성 및 발행
                msg = PickeeMobileSpeedControl()
                msg.speed_mode = mode
                msg.target_speed = speed
                self.speed_control_publisher.publish(msg)

                self.get_logger().info(f"🚀 Speed Control 발행: mode={mode}, speed={speed}")

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
