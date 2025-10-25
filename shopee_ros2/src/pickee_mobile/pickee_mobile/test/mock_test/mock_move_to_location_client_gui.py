import sys
import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose2D
from shopee_interfaces.srv import PickeeMobileMoveToLocation
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PySide6.QtCore import QTimer


class MoveToLocationClient(Node):
    def __init__(self):
        super().__init__('gui_move_to_location_client')
        self.client = self.create_client(PickeeMobileMoveToLocation, '/pickee/mobile/move_to_location')

        # 서비스 대기
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서비스 대기 중...')

        self.get_logger().info('서비스 연결 완료 ✅')


class MoveToLocationGUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.init_ui()
        self.setWindowTitle("Pickee Mobile MoveToLocation GUI")

    def init_ui(self):
        layout = QVBoxLayout()

        # --- 입력 필드 ---
        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.theta_input = QLineEdit()

        layout.addWidget(QLabel("X 좌표:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y 좌표:"))
        layout.addWidget(self.y_input)
        layout.addWidget(QLabel("Theta (라디안):"))
        layout.addWidget(self.theta_input)

        # --- 버튼 ---
        self.send_button = QPushButton("이동 요청 보내기 🚀")
        self.send_button.clicked.connect(self.send_request)
        layout.addWidget(self.send_button)

        # --- 상태 메시지 ---
        self.status_label = QLabel("서비스 준비 완료 ✅")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def send_request(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            theta = float(self.theta_input.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "x, y, theta는 숫자만 입력해야 합니다.")
            return

        request = PickeeMobileMoveToLocation.Request()
        request.robot_id = 1
        request.order_id = 123
        request.location_id = 456
        request.target_pose = Pose2D(x=x, y=y, theta=theta)

        self.future = self.node.client.call_async(request)
        self.future.add_done_callback(self.response_callback)

        self.status_label.setText("요청 전송 중... ⏳")

    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                text = f"✅ 이동 성공: {response.message}"
            else:
                text = f"⚠ 이동 실패: {response.message}"
        except Exception as e:
            text = f"❌ 서비스 호출 실패: {e}"

        # GUI 스레드에서 안전하게 라벨 갱신
        QTimer.singleShot(0, lambda: self.status_label.setText(text))


def main():
    rclpy.init()
    node = MoveToLocationClient()

    app = QApplication(sys.argv)
    gui = MoveToLocationGUI(node)
    gui.show()

    # ROS2 스핀을 Qt 타이머로 주기적으로 호출
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.1))
    timer.start(100)

    app.exec()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
