# QT 대시보드 구현 계획서

이 문서는 `PICKEE_MAIN_DASHBOARD_DESIGN_QT.md`에 기술된 상세 설계를 바탕으로 PySide6를 사용한 ROS2 대시보드 구현 단계를 기술합니다.

---

## 0단계: 개발 환경 설정

### 1. 의존성 설치
대시보드 개발에 필요한 PySide6를 설치합니다.

```bash
source .venv/bin/activate
pip install PySide6
```

### 2. 기본 파일 구조 생성
`pickee_main` 패키지 내에 대시보드 관련 코드를 저장할 디렉토리와 파일을 생성합니다.

```bash
mkdir -p src/pickee_main/pickee_main/dashboard
touch src/pickee_main/pickee_main/dashboard/__init__.py
touch src/pickee_main/pickee_main/dashboard/main_window.py
touch src/pickee_main/pickee_main/dashboard/ros_node.py
touch src/pickee_main/pickee_main/dashboard/launcher.py
```

-   `main_window.py`: 메인 UI 및 위젯 구성
-   `ros_node.py`: ROS2 통신(노드, 구독, 폴링)을 처리하는 스레드
-   `launcher.py`: 대시보드 애플리케이션을 실행하는 진입점

## 1단계: 기본 UI 레이아웃 구현

`main_window.py` 파일에 `QMainWindow`를 기반으로 4개의 주요 패널을 배치합니다. `QDockWidget`을 사용하면 사용자가 패널의 위치를 자유롭게 변경하거나 분리할 수 있습니다.

**`src/pickee_main/pickee_main/dashboard/main_window.py`**
```python
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QTextEdit, QTreeWidget, QDockWidget
)
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pickee Main Dashboard")
        self.setGeometry(100, 100, 1200, 800)

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

        # 3. 로그 출력 패널
        log_dock = QDockWidget("Logs", self)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_dock.setWidget(self.log_text_edit)
        self.addDockWidget(Qt.RightDockWidgetArea, log_dock)

        self.setCentralWidget(self.log_text_edit) # 중앙 위젯 설정

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

## 2단계: ROS2 통신 스레드 구현

UI의 응답성을 유지하기 위해 ROS2 관련 작업(spin, 구독, 폴링)은 별도의 `QThread`에서 처리합니다. `ros_node.py`에 ROS2 노드 및 통신 로직을 작성합니다.

**`src/pickee_main/pickee_main/dashboard/ros_node.py`**
```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Log
from PySide6.QtCore import QThread, Signal

class RosNodeThread(QThread):
    # 로그 메시지를 전달하기 위한 시그널
    log_received = Signal(str, str, str) # node_name, level, message

    def __init__(self):
        super().__init__()
        self.node = None

    def run(self):
        rclpy.init()
        self.node = Node("pickee_dashboard_node")
        
        # /rosout 토픽을 구독하여 모든 로그 메시지 수신
        self.node.create_subscription(
            Log,
            '/rosout',
            self.log_callback,
            10
        )
        
        rclpy.spin(self.node)
        
        # 스레드 종료 시 노드 정리
        self.node.destroy_node()
        rclpy.shutdown()

    def log_callback(self, msg: Log):
        # 로그 레벨을 문자로 변환 (예: 10 -> DEBUG, 20 -> INFO)
        level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}
        level = level_map.get(msg.level, 'UNKNOWN')
        
        # 시그널을 통해 UI 스레드로 데이터 전달
        self.log_received.emit(msg.name, level, msg.msg)

    def stop(self):
        if self.node and rclpy.ok():
            rclpy.shutdown()
        self.wait() # 스레드가 완전히 종료될 때까지 대기
```

## 3단계: 로그 정보 실시간 표시

`main_window.py`에서 `RosNodeThread`를 생성하고, `log_received` 시그널을 `QTextEdit`에 로그를 추가하는 슬롯에 연결합니다.

**`src/pickee_main/pickee_main/dashboard/main_window.py` (수정)**
```python
# ... 기존 import ...
from .ros_node import RosNodeThread

class MainWindow(QMainWindow):
    def __init__(self):
        # ... 기존 __init__ 내용 ...

        # ROS2 스레드 시작
        self.ros_thread = RosNodeThread()
        self.ros_thread.log_received.connect(self.append_log)
        self.ros_thread.start()

    def append_log(self, node_name, level, message):
        log_text = f"[{level}] [{node_name}]: {message}"
        self.log_text_edit.append(log_text)

    def closeEvent(self, event):
        # 윈도우 종료 시 ROS 스레드 정리
        self.ros_thread.stop()
        super().closeEvent(event)

# ... 기존 if __name__ == '__main__': ...
```

## 4단계: 노드/서비스/토픽 목록 동적 갱신

`RosNodeThread`에 `QTimer`를 사용하여 주기적으로 ROS2 네트워크 정보를 폴링하고, 그 결과를 시그널로 UI에 전달합니다.

**`src/pickee_main/pickee_main/dashboard/ros_node.py` (수정)**
```python
# ... 기존 import ...
from PySide6.QtCore import QThread, Signal, QTimer

class RosNodeThread(QThread):
    log_received = Signal(str, str, str)
    nodes_updated = Signal(list)
    topics_updated = Signal(list)
    services_updated = Signal(list)

    def __init__(self):
        super().__init__()
        self.node = None
        self.timer = None

    def run(self):
        rclpy.init()
        self.node = Node("pickee_dashboard_node")
        # ... 기존 구독 ...

        # 1초마다 시스템 정보를 폴링하는 타이머 설정
        self.timer = self.node.create_timer(1.0, self.poll_system_info)

        rclpy.spin(self.node)
        # ... 기존 정리 코드 ...

    def log_callback(self, msg: Log):
        # ... 기존 콜백 내용 ...

    def poll_system_info(self):
        if not self.node: return

        # 노드 목록 가져오기
        node_names = self.node.get_node_names_and_namespaces()
        self.nodes_updated.emit([f"{ns}{n}" for n, ns in node_names])

        # 토픽 목록 가져오기
        topic_names_and_types = self.node.get_topic_names_and_types()
        self.topics_updated.emit(topic_names_and_types)

        # 서비스 목록 가져오기
        service_names_and_types = self.node.get_service_names_and_types()
        self.services_updated.emit(service_names_and_types)

    def stop(self):
        # ... 기존 stop 내용 ...
```

**`src/pickee_main/pickee_main/dashboard/main_window.py` (수정)**
```python
# ... 기존 import ...
from PySide6.QtWidgets import QTreeWidgetItem

class MainWindow(QMainWindow):
    def __init__(self):
        # ... 기존 __init__ 내용 ...
        self.ros_thread.nodes_updated.connect(self.update_node_list)
        self.ros_thread.topics_updated.connect(self.update_topic_list)
        self.ros_thread.services_updated.connect(self.update_service_list)

        # 서비스/토픽 트리 초기화
        self.topics_root = QTreeWidgetItem(self.comm_tree_widget, ["Topics"])
        self.services_root = QTreeWidgetItem(self.comm_tree_widget, ["Services"])

    # ... 기존 append_log, closeEvent ...

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
```

## 5단계: UI 인터랙션 및 상세 기능 구현

마지막으로 노드 선택 시 로그 필터링, 검색 등 사용자 편의 기능을 추가합니다.

-   **노드 필터링:** `QListWidget`의 `currentItemChanged` 시그널을 사용하여 선택된 노드 이름을 저장하고, `append_log` 함수에서 현재 선택된 노드의 로그만 표시하도록 수정합니다.
-   **검색:** 각 패널 상단에 `QLineEdit`을 추가하고 `textChanged` 시그널을 사용하여 목록을 필터링합니다.
-   **로그 레벨 필터링:** `QCheckBox`를 추가하고 상태 변경 시 `append_log`에 필터링 로직을 추가합니다.

이 단계들은 UI의 완성도를 높이는 과정으로, 위젯의 시그널/슬롯을 활용하여 비교적 간단하게 구현할 수 있습니다.

## 6단계: 실행기(Launcher) 작성

`launcher.py` 파일에 대시보드를 실행하는 코드를 작성하고, `setup.py`에 entry point를 추가하여 `ros2 run`으로 실행할 수 있게 합니다.

**`src/pickee_main/pickee_main/dashboard/launcher.py`**
```python
import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

**`src/pickee_main/setup.py` (수정)**
```python
# ...
    entry_points={
        'console_scripts': [
            'main_controller = pickee_main.main_controller:main',
            'mock_mobile_node = pickee_main.mock_mobile_node:main',
            'mock_arm_node = pickee_main.mock_arm_node:main',
            'mock_vision_node = pickee_main.mock_vision_node:main',
            'dashboard = pickee_main.dashboard.launcher:main', # 이 줄 추가
        ],
    },
# ...
```

이제 `colcon build` 후 `ros2 run pickee_main dashboard` 명령어로 대시보드를 실행할 수 있습니다.
