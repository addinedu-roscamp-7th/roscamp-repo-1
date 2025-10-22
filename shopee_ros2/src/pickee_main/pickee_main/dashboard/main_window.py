import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QTextEdit, QTreeWidget, QDockWidget, QTreeWidgetItem, QLabel
)
from PySide6.QtCore import Qt

from .ros_node import RosNodeThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pickee Main Dashboard")
        self.setGeometry(100, 100, 1400, 800)

        # 1. 노드 목록 패널
        node_dock = QDockWidget("Nodes", self)
        self.node_list_widget = QListWidget()
        node_dock.setWidget(self.node_list_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, node_dock)

        # 2. 서비스/토픽 패널
        comm_dock = QDockWidget("Services & Topics", self)
        self.comm_tree_widget = QTreeWidget()
        self.comm_tree_widget.setHeaderLabels(["Name", "Type"])
        comm_dock.setWidget(self.comm_tree_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, comm_dock)

        # 3. 상태 머신 패널
        state_dock = QDockWidget("State Machine", self)
        self.state_label = QLabel("Initializing...")
        self.state_label.setAlignment(Qt.AlignCenter)
        state_dock.setWidget(self.state_label)
        self.addDockWidget(Qt.TopDockWidgetArea, state_dock)

        # 4. 로그 출력 패널
        log_dock = QDockWidget("Logs", self)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_dock.setWidget(self.log_text_edit)
        self.addDockWidget(Qt.RightDockWidgetArea, log_dock)

        self.setCentralWidget(self.log_text_edit) # 중앙 위젯 설정

        # ROS2 스레드 시작
        self.ros_thread = RosNodeThread()
        self.ros_thread.log_received.connect(self.append_log)
        self.ros_thread.nodes_updated.connect(self.update_node_list)
        self.ros_thread.topics_updated.connect(self.update_topic_list)
        self.ros_thread.services_updated.connect(self.update_service_list)
        self.ros_thread.state_updated.connect(self.update_state)
        self.ros_thread.start()

        # 서비스/토픽 트리 초기화
        self.topics_root = QTreeWidgetItem(self.comm_tree_widget, ["Topics"])
        self.services_root = QTreeWidgetItem(self.comm_tree_widget, ["Services"])

    def get_node_style(self, node_name):
        if node_name.startswith('pickee_main_controller'):
            return 'color:orange; font-weight:bold;'
        elif node_name.startswith('mock_mobile_node'):
            return 'color:blue; font-weight:bold;'
        elif node_name.startswith('pickee_mobile_controller'):
            return 'color:blue; font-weight:bold;'
        elif node_name.startswith('pickee_mobile_wonho_node'):
            return 'color:blue; font-weight:bold;'
        elif node_name.startswith('mock_arm_node'):
            return 'color:darkviolet; font-weight:bold;'
        elif node_name.startswith('pickee_arm_controller'):
            return 'color:darkviolet; font-weight:bold;'
        elif node_name.startswith('mock_vision_node'):
            return 'color:green; font-weight:bold;'
        elif node_name.startswith('pickee_vision_system'):
            return 'color:green; font-weight:bold;'
        elif node_name.startswith('mock_shopee_main'):
            return 'color:crimson; font-weight:bold;'
        
        return ''

    def append_log(self, node_name, level, message, is_state_log):
        node_style = self.get_node_style(node_name)
    
        bg_color_style = ""
        if is_state_log:
            bg_color_style = "background-color:#FFFFE0;"

        # 로그 레벨별 색상 스타일 설정
        level_style = ""
        if level == "ERROR":
            level_style = "color:red; font-weight:bold;"
        elif level == "WARN" or level == "WARNING":
            level_style = "color:orange; font-weight:bold;"
        elif level == "DEBUG":
            level_style = "color:gray;"

        log_html = f'<span style="{bg_color_style}">'
        if level != 'INFO':
            log_html += f'<span style="{level_style}">[{level}]</span> '
        log_html += '</span>'

        log_html += f'<span style="{bg_color_style} {node_style}">[{node_name}]</span>'

        log_html += f'<span style="{bg_color_style}">: {message}</span>'

        self.log_text_edit.append(log_html)


    def update_state(self, state):
        self.state_label.setText(state)

    def closeEvent(self, event):
        # 윈도우 종료 시 ROS 스레드 정리
        self.ros_thread.stop()
        super().closeEvent(event)

    def update_node_list(self, nodes):
        current_nodes = {self.node_list_widget.item(i).text() for i in range(self.node_list_widget.count())}
        new_nodes = set(nodes)
        if current_nodes != new_nodes:
            self.node_list_widget.clear()
            self.node_list_widget.addItems(sorted(list(new_nodes)))

    def update_topic_list(self, topics):
        self.topics_root.takeChildren() # 기존 항목 삭제
        for name, types in topics:
            item = QTreeWidgetItem(self.topics_root, [name, ", ".join(types)])

    def update_service_list(self, services):
        self.services_root.takeChildren() # 기존 항목 삭제
        for name, types in services:
            item = QTreeWidgetItem(self.services_root, [name, ", ".join(types)])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
