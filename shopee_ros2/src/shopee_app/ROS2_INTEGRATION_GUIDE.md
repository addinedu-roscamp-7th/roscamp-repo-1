# Shopee App ROS2 통합 가이드

## 📋 개요

현재 `shopee_app`은 일반 Python/PyQt6 애플리케이션으로 구성되어 있어 **ROS2 토픽/서비스를 사용할 수 없습니다.**

이 문서는 shopee_app을 ROS2 패키지로 변경하여 Main Service와 통신할 수 있도록 구조를 개선하는 방법을 안내합니다.

참고: `pickee_main/dashboard` 패키지의 구조를 참고하여 작성되었습니다.

---

## 🔍 현재 구조의 문제점

### 현재 디렉토리 구조
```
shopee_app/
├── app.py              # 일반 Python 진입점
├── dev.py              # 개발 모드 스크립트
├── pages/              # UI 페이지
├── ui/                 # Qt Designer 파일
├── ui_gen/             # 생성된 Python UI
└── requirements.txt    # Python 의존성
```

### 문제점
1. ❌ ROS2 패키지 구조가 아님 (`package.xml`, `setup.py` 없음)
2. ❌ `colcon build`로 빌드되지 않음
3. ❌ ROS2 토픽/서비스를 사용할 수 없음
4. ❌ ROS2 환경에서 실행 불가

---

## ✅ 권장 구조 (ROS2 통합)

### 변경 후 디렉토리 구조
```
shopee_app/
├── package.xml                    # ROS2 패키지 정의
├── setup.py                       # Python 패키지 설정 (entry_points 포함)
├── setup.cfg                      # 설치 설정
├── resource/
│   └── shopee_app                 # 빈 파일 (ROS2 패키지 마커)
├── launch/                        # Launch 파일 (선택사항)
│   └── app.launch.py
├── dev.py                         # 개발 모드 스크립트 (그대로 유지)
└── shopee_app/                    # 메인 패키지 디렉토리
    ├── __init__.py
    ├── launcher.py                # GUI 실행 진입점
    ├── ros_node.py                # ROS2 노드 (QThread로 구현)
    ├── pages/                     # UI 페이지들 (기존 유지)
    │   ├── main_window.py
    │   ├── admin_window.py
    │   ├── user_window.py
    │   ├── models/
    │   └── widgets/
    ├── ui/                        # Qt Designer 파일 (기존 유지)
    └── ui_gen/                    # 생성된 Python UI (기존 유지)
```

---

## 🔧 단계별 변경 방법

### 1단계: 디렉토리 구조 변경

```bash
cd /path/to/shopee_ros2/src/shopee_app

# 기존 pages를 shopee_app 디렉토리로 이동
mkdir -p shopee_app
mv pages shopee_app/
mv ui shopee_app/
mv ui_gen shopee_app/

# __init__.py 생성
touch shopee_app/__init__.py
```

### 2단계: ROS2 패키지 파일 생성

#### 2-1. `package.xml` 생성

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>shopee_app</name>
  <version>0.0.0</version>
  <description>Shopee GUI Application with ROS2 Integration</description>
  <maintainer email="your_email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>shopee_interfaces</depend>
  <depend>rcl_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

#### 2-2. `setup.py` 생성

```python
from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'shopee_app'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch 파일이 있는 경우
        # ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'PyQt6',
        'watchdog',  # dev.py에서 사용
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Shopee GUI Application with ROS2 Integration',
    license='Apache License 2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'shopee_app = shopee_app.launcher:main',
        ],
    },
)
```

#### 2-3. `setup.cfg` 생성

```ini
[develop]
script_dir=$base/lib/shopee_app
[install]
install_scripts=$base/lib/shopee_app
```

#### 2-4. `resource/shopee_app` 생성

```bash
mkdir -p resource
touch resource/shopee_app
```

### 3단계: ROS2 노드 통합

#### 3-1. `shopee_app/ros_node.py` 생성

Main Service와 통신하기 위한 ROS2 노드를 QThread로 구현합니다.

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Log
from shopee_interfaces.msg import OrderStatus, RobotStatus  # 필요한 메시지 타입
from PyQt6.QtCore import QThread, pyqtSignal

class RosNodeThread(QThread):
    '''ROS2 노드를 별도 스레드에서 실행하고 Signal로 GUI에 데이터 전달'''
    
    # GUI로 데이터를 전달하기 위한 Signal 정의
    log_received = pyqtSignal(str, str, str)  # (노드명, 레벨, 메시지)
    order_updated = pyqtSignal(dict)          # 주문 정보 업데이트
    robot_status_updated = pyqtSignal(str)    # 로봇 상태 업데이트
    
    def __init__(self):
        super().__init__()
        self.node = None
        self.timer = None
    
    def run(self):
        '''스레드 실행 시 ROS2 노드 초기화 및 spin'''
        rclpy.init()
        self.node = Node('shopee_app_node')
        
        # /rosout 토픽 구독 (모든 로그 메시지)
        self.node.create_subscription(
            Log,
            '/rosout',
            self.log_callback,
            10
        )
        
        # Main Service의 토픽 구독 예시
        # self.node.create_subscription(
        #     OrderStatus,
        #     '/main/order_status',
        #     self.order_status_callback,
        #     10
        # )
        
        # 주기적으로 시스템 정보 폴링
        self.timer = self.node.create_timer(1.0, self.poll_system_info)
        
        rclpy.spin(self.node)
        
        # 종료 시 정리
        self.node.destroy_node()
        rclpy.shutdown()
    
    def log_callback(self, msg: Log):
        '''로그 메시지 콜백'''
        level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}
        level = level_map.get(msg.level, 'UNKNOWN')
        self.log_received.emit(msg.name, level, msg.msg)
    
    def order_status_callback(self, msg):
        '''주문 상태 업데이트 콜백'''
        order_data = {
            'order_id': msg.order_id,
            'status': msg.status,
            # ... 필요한 필드 추가
        }
        self.order_updated.emit(order_data)
    
    def poll_system_info(self):
        '''주기적으로 ROS2 시스템 정보 수집'''
        if not self.node:
            return
        
        # 노드 목록 가져오기
        node_names = self.node.get_node_names_and_namespaces()
        # ... 필요한 처리
    
    def stop(self):
        '''노드 종료'''
        if self.node and rclpy.ok():
            rclpy.shutdown()
        self.wait()  # 스레드 종료 대기
```

#### 3-2. `shopee_app/launcher.py` 생성

```python
import sys
from PyQt6.QtWidgets import QApplication
from .pages.main_window import MainWindow
from .ros_node import RosNodeThread

def main():
    '''GUI 애플리케이션 진입점'''
    app = QApplication(sys.argv)
    
    # ROS2 노드 스레드 시작
    ros_thread = RosNodeThread()
    ros_thread.start()
    
    # 메인 윈도우 생성 및 ROS 노드와 연결
    window = MainWindow()
    
    # ROS 노드의 Signal을 GUI Slot에 연결
    ros_thread.log_received.connect(window.on_log_received)
    ros_thread.order_updated.connect(window.on_order_updated)
    ros_thread.robot_status_updated.connect(window.on_robot_status_updated)
    
    window.show()
    
    # 애플리케이션 종료 시 ROS 노드도 종료
    app.aboutToQuit.connect(ros_thread.stop)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

#### 3-3. `shopee_app/pages/main_window.py` 수정

```python
from PyQt6.QtWidgets import QMainWindow
# ... 기존 import

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 기존 초기화 코드
    
    # ROS2 노드에서 받은 데이터를 처리하는 Slot 추가
    def on_log_received(self, node_name: str, level: str, message: str):
        '''로그 메시지를 받았을 때 처리'''
        # 로그 위젯에 표시
        pass
    
    def on_order_updated(self, order_data: dict):
        '''주문 정보가 업데이트되었을 때 처리'''
        # UI 업데이트
        pass
    
    def on_robot_status_updated(self, status: str):
        '''로봇 상태가 업데이트되었을 때 처리'''
        # UI 업데이트
        pass
```

### 4단계: 빌드 및 실행

#### 4-1. 패키지 빌드

```bash
cd /path/to/shopee_ros2

# shopee_app 패키지만 빌드
colcon build --packages-select shopee_app

# 환경 변수 로드
source install/setup.bash
```

#### 4-2. 실행

```bash
# ROS2 명령으로 실행
ros2 run shopee_app shopee_app

# 또는 Launch 파일 사용 (있는 경우)
ros2 launch shopee_app app.launch.py
```

### 5단계: 개발 모드 사용 (선택사항)

개발 중에는 `dev.py`를 계속 사용할 수 있습니다.

#### 5-1. `dev.py` 수정

```python
# dev.py 상단에 추가
import os
import sys

# ROS2 환경 변수 로드
ROS2_INSTALL_PATH = os.path.join(os.path.dirname(__file__), '../../install/setup.bash')

# 기존 코드는 그대로 유지하되, app.py 대신 shopee_app.launcher를 실행
def start_app():
    return subprocess.Popen(
        [sys.executable, '-m', 'shopee_app.launcher'],
        cwd=str(ROOT)
    )
```

#### 5-2. 개발 모드 실행

```bash
cd shopee_ros2/src/shopee_app
source ../../install/setup.bash  # ROS2 환경 로드
python dev.py
```

---

## 📝 Main Service와의 통신 예시

### InterfaceSpecification 참고

`docs/InterfaceSpecification/App_vs_Main.md` 문서를 참고하여 필요한 토픽/서비스를 구독하고 호출합니다.

#### 예시 1: 주문 생성 요청

```python
# shopee_app/ros_node.py에 추가

from shopee_interfaces.srv import CreateOrder

class RosNodeThread(QThread):
    # ... 기존 코드
    
    def __init__(self):
        super().__init__()
        # ... 기존 코드
        self.create_order_client = None
    
    def run(self):
        # ... 기존 코드
        
        # 서비스 클라이언트 생성
        self.create_order_client = self.node.create_client(
            CreateOrder,
            '/main/create_order'
        )
    
    def call_create_order(self, product_ids, user_id):
        '''주문 생성 서비스 호출'''
        if not self.create_order_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error('Create order service not available')
            return None
        
        request = CreateOrder.Request()
        request.product_ids = product_ids
        request.user_id = user_id
        
        future = self.create_order_client.call_async(request)
        # 비동기 처리 또는 결과 대기
        return future
```

#### 예시 2: 로봇 상태 구독

```python
# shopee_app/ros_node.py에 추가

from shopee_interfaces.msg import RobotStatusList

class RosNodeThread(QThread):
    robot_list_updated = pyqtSignal(list)  # Signal 추가
    
    def run(self):
        # ... 기존 코드
        
        # 로봇 상태 토픽 구독
        self.node.create_subscription(
            RobotStatusList,
            '/main/robot_status_list',
            self.robot_status_callback,
            10
        )
    
    def robot_status_callback(self, msg):
        '''로봇 상태 리스트 콜백'''
        robot_data = []
        for robot in msg.robots:
            robot_data.append({
                'robot_id': robot.robot_id,
                'state': robot.state,
                'battery': robot.battery,
                # ... 필요한 필드
            })
        self.robot_list_updated.emit(robot_data)
```

---

## ⚠️ 주의사항

1. **PyQt6 vs PySide6**
   - 현재 `shopee_app`은 PyQt6 사용
   - `pickee_main/dashboard`는 PySide6 사용
   - 둘 중 하나로 통일하는 것을 권장 (라이센스 차이 확인 필요)

2. **의존성 관리**
   - `requirements.txt`의 패키지들은 `setup.py`의 `install_requires`에도 추가
   - ROS2 관련 패키지는 `package.xml`에 명시

3. **스레드 안전성**
   - ROS2 노드는 별도 스레드에서 실행
   - GUI 업데이트는 반드시 Signal/Slot을 통해 메인 스레드에서 수행

4. **개발 모드**
   - `dev.py`는 개발 편의를 위해 유지 가능
   - 실제 배포 시에는 ROS2 명령으로 실행

---

## 📚 참고 자료

1. **pickee_main/dashboard 패키지**
   - 위치: `shopee_ros2/src/pickee_main/pickee_main/dashboard/`
   - ROS2 + GUI 통합 예시로 참고

2. **설계 문서**
   - `docs/InterfaceSpecification/App_vs_Main.md`: App과 Main Service 간 인터페이스
   - `docs/DevelopmentPlan/App/AppPlan.md`: App 개발 계획

3. **ROS2 공식 문서**
   - [Creating a Python package](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)
   - [Using Python packages](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html)

---

## ❓ 질문 및 지원

구조 변경 중 문제가 발생하면:
1. `pickee_main/dashboard` 패키지 구조 참고
2. 빌드 에러는 `colcon build --packages-select shopee_app --event-handlers console_direct+` 로 상세 로그 확인
3. 팀원에게 문의

---

**작성일**: 2025-10-19  
**작성자**: 시스템 아키텍트  
**버전**: 1.0

