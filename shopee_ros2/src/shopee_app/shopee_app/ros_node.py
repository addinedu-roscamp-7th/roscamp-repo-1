import threading
from typing import Callable
from typing import Optional

from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSignal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# ROS 인터페이스 패키지 사용 가능 여부를 표시한다.
ROS_INTERFACES_AVAILABLE = True

try:
    from shopee_interfaces.msg import PickeeRobotStatus
except ModuleNotFoundError:
    from dataclasses import dataclass

    ROS_INTERFACES_AVAILABLE = False

    @dataclass
    class PickeeRobotStatus:
        robot_id: int = 0
        state: str = ''
        battery_level: float = 0.0
        current_order_id: int = 0
        position_x: float = 0.0
        position_y: float = 0.0
        orientation_z: float = 0.0


class ShopeeAppNode(Node):
    '''Pickee 로봇 상태를 구독해 UI에 전달하는 ROS2 노드.'''

    def __init__(self):
        '''필요한 ROS 엔티티를 초기화하고 상태 토픽을 구독한다.'''
        super().__init__("shopee_app_node")
        self._status_listeners: list[Callable[[PickeeRobotStatus], None]] = []

        self.create_subscription(
            PickeeRobotStatus, "/pickee/robot_status", self.on_pickee_status, 10
        )

    def on_pickee_status(self, msg: PickeeRobotStatus) -> None:
        '''수신한 로봇 상태 메시지를 등록된 리스너들에게 전달한다.'''
        for callback in list(self._status_listeners):
            try:
                callback(msg)
            except Exception:
                self.get_logger().exception("Pickee status listener 실패")

    def add_status_listener(
        self, callback: Callable[[PickeeRobotStatus], None]
    ) -> None:
        '''중복을 방지하며 상태 콜백을 등록한다.'''
        if callback not in self._status_listeners:
            self._status_listeners.append(callback)

    def remove_status_listener(
        self, callback: Callable[[PickeeRobotStatus], None]
    ) -> None:
        '''등록된 상태 콜백을 안전하게 제거한다.'''
        if callback in self._status_listeners:
            self._status_listeners.remove(callback)


class RosNodeThread(QThread):
    '''GUI와 분리된 스레드에서 ROS2 이벤트 루프를 실행한다.'''

    node_ready = pyqtSignal()
    node_error = pyqtSignal(str)
    pickee_status_received = pyqtSignal(object)

    def __init__(self):
        '''스레드 동기화를 위한 이벤트와 실행기를 준비한다.'''
        super().__init__()
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()
        self._executor: Optional[SingleThreadedExecutor] = None
        self._node: Optional[ShopeeAppNode] = None
        self._initialized = False

    @property
    def node(self) -> Optional[ShopeeAppNode]:
        return self._node

    def run(self):
        '''ROS 초기화 및 스핀 루프를 실행하고 오류를 UI에 전달한다.'''
        if not ROS_INTERFACES_AVAILABLE:
            self.node_error.emit(
                'ROS2 인터페이스 패키지(shopee_interfaces)가 설치되지 않았습니다. '
                'colcon 빌드 또는 의존성 설치를 먼저 수행해주세요.'
            )
            return
        try:
            try:
                rclpy.init(args=None)
                self._initialized = True
            except RuntimeError:
                self._initialized = False
            self._node = ShopeeAppNode()
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._node.add_status_listener(self._handle_pickee_status)
            self._ready_event.set()
            self.node_ready.emit()
            while not self._shutdown_event.is_set():
                try:
                    self._executor.spin_once(timeout_sec=0.1)
                except ExternalShutdownException:
                    break
        except Exception as exc:
            self.node_error.emit(str(exc))
        finally:
            self._ready_event.clear()
            if self._executor is not None and self._node is not None:
                self._executor.remove_node(self._node)
            if self._node is not None:
                self._node.remove_status_listener(self._handle_pickee_status)
                self._node.destroy_node()
                self._node = None
            if self._executor is not None:
                self._executor.shutdown()
                self._executor = None
            if self._initialized:
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
                self._initialized = False

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        '''노드 준비가 완료될 때까지 대기하거나 타임아웃을 반환한다.'''
        return self._ready_event.wait(timeout=timeout)

    def shutdown(self):
        '''외부에서 종료 신호를 받아 스핀 루프를 멈춘다.'''
        self._shutdown_event.set()

    def _handle_pickee_status(self, msg: PickeeRobotStatus) -> None:
        '''선택된 콜백 대신 PyQt 시그널로 상태 메시지를 전달한다.'''
        self.pickee_status_received.emit(msg)
