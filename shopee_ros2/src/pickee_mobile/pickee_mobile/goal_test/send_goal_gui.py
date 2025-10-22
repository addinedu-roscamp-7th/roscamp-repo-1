import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel
)
from PyQt5.QtCore import Qt


class SendGoalGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.goal_publisher = self.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )




        self.setWindowTitle("ROS2 NavigateToPose (Jazzy용)")
        self.setGeometry(200, 200, 550, 400)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # 위치 입력
        self.x_input = QLineEdit("1.0")
        self.y_input = QLineEdit("1.0")
        self.z_input = QLineEdit("0.0")

        # 방향 입력 (쿼터니언)
        self.oz_input = QLineEdit("0.0")
        self.ow_input = QLineEdit("1.0")

        form_layout.addRow("Position X:", self.x_input)
        form_layout.addRow("Position Y:", self.y_input)
        form_layout.addRow("Position Z:", self.z_input)
        form_layout.addRow("Orientation Z:", self.oz_input)
        form_layout.addRow("Orientation W:", self.ow_input)

        layout.addLayout(form_layout)

        # 버튼
        self.btn_generate = QPushButton("명령 생성")
        self.btn_send = QPushButton("ROS2 명령 실행")
        layout.addWidget(self.btn_generate)
        layout.addWidget(self.btn_send)

        # 출력창
        layout.addWidget(QLabel("생성된 명령:"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

        # 이벤트 연결
        self.btn_generate.clicked.connect(self.generate_command)
        # self.btn_send.clicked.connect(self.send_goal_action)
        self.btn_send.clicked.connect(self.send_goal_topic)

    def generate_command(self):
        """입력값으로 명령문 생성"""
        x = self.x_input.text().strip()
        y = self.y_input.text().strip()
        z = self.z_input.text().strip()
        oz = self.oz_input.text().strip()
        ow = self.ow_input.text().strip()

        command = (
            f"ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "
            f"\"{{pose: {{header: {{frame_id: 'map'}}, "
            f"pose: {{position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{z: {oz}, w: {ow}}}}}}}}}\" "
            f"--feedback"
        )
        self.output.setPlainText(command)

    # def send_goal_action(self):
    #     """터미널에서 명령 실행"""
    #     command = self.output.toPlainText().strip()
    #     if not command:
    #         self.output.setPlainText("⚠️ 먼저 명령을 생성하세요.")
    #         return

    #     self.output.append("\n🚀 명령 실행 중...\n")
    #     try:
    #         result = subprocess.run(
    #             command, shell=True, capture_output=True, text=True
    #         )
    #         if result.stdout:
    #             self.output.append(result.stdout)
    #         if result.stderr:
    #             self.output.append(result.stderr)
    #     except Exception as e:
    #         self.output.append(f"❌ 실행 오류: {e}")

    def send_goal_topic(self, x, y, theta):


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SendGoalGUI()
    window.show()
    sys.exit(app.exec_())
