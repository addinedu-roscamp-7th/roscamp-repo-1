import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty
import os


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
        self.pickee_vel_publisher = self.create_publisher(Twist, '/cmd_vel_modify', 10)

        # 키 입력 스레드 시작
        self.input_thread = threading.Thread(target=self.keyboard_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        self.get_logger().info('✅ cmd_vel Modifier 노드 시작됨 (기본 scale=0.5)')
        self.get_logger().info('키 입력 명령: [Z:+0.1] [X:-0.1] [A:0] [S:1] [C:종료]')

    # ---------------------------------------------------------------------
    # 📦 콜백: /cmd_vel 수신 시
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

    # ---------------------------------------------------------------------
    # ⌨️ 키보드 입력 스레드
    # ---------------------------------------------------------------------
    def keyboard_input(self):
        """키 입력 감지 스레드"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
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
                    self.running = False
                    os._exit(0)

                # scale 범위 제한 (예: -2.0 ~ 2.0)
                self.scale = max(min(self.scale, 2.0), -2.0)
                self.get_logger().info(f'🔧 현재 Scale: {self.scale:.2f}')

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# -------------------------------------------------------------------------
# 🚀 메인
# -------------------------------------------------------------------------
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
