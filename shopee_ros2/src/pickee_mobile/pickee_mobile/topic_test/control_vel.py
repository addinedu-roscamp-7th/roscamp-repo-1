import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import threading
import termios
import tty
import os

class TwistModifier(Node):
    """Twist 메시지를 키보드 입력에 따라 수정하고 /cmd_vel로 발행"""

    def __init__(self):
        super().__init__('twist_modifier')

        # 기존 /cmd_vel 구독
        self.subscription = self.create_subscription(
            Twist, 'cmd_vel', self.modify_cmd_vel_callback, 10
        )

        # 수정된 /cmd_vel 발행
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # 키 상태
        self.key_command = 'z'  # 기본값: 정상 속도

        # 키보드 입력 스레드
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info('✅ Twist Modifier Node started (Keys: z=normal, x=slow, c=stop, v=exit)')

    def modify_cmd_vel_callback(self, msg: Twist):
        """Twist 메시지를 키 입력에 따라 수정 후 발행"""
        modified_twist = Twist()

        # 키 명령에 따른 배율 설정
        if self.key_command == 'x':
            scale = 0.8
            state = '감속'
        elif self.key_command == 'c':
            scale = 0.0
            state = '정지'
        else:
            scale = 1.0
            state = '정상속도'

        # 속도 적용
        modified_twist.linear.x = msg.linear.x * scale
        modified_twist.linear.y = msg.linear.y * scale
        modified_twist.linear.z = msg.linear.z * scale
        modified_twist.angular.x = msg.angular.x * scale
        modified_twist.angular.y = msg.angular.y * scale
        modified_twist.angular.z = msg.angular.z * scale

        self.publisher_.publish(modified_twist)

        self.get_logger().info(
            f"[{state}] Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}) -> "
            f"({modified_twist.linear.x:.2f}, {modified_twist.linear.y:.2f}), "
            f"Angular: ({msg.angular.z:.2f}) -> ({modified_twist.angular.z:.2f})"
        )

    def keyboard_listener(self):
        """키보드 입력 감시 (항상 종료 시 터미널 복원)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch in ['x', 'z', 'c']:
                    self.key_command = ch
                    self.get_logger().info(f"Keyboard pressed: {ch}")
                elif ch == 'v':  # 종료
                    self.get_logger().info("✅ 'v' pressed. Shutting down node...")
                    rclpy.shutdown()
                    break
        except Exception as e:
            self.get_logger().error(f"Keyboard listener error: {e}")
        finally:
            # 항상 터미널 복원
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            os.system("stty sane")
            print("\033[0m")  # 터미널 색상 초기화
            print("🔁 Terminal restored. You can type normally again.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TwistModifier()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🧹 KeyboardInterrupt detected. Cleaning up...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        os.system("stty sane")
        print("\033[0m")  # 색상 리셋
        print("✅ Node terminated. Terminal input restored.")

if __name__ == '__main__':
    main()
