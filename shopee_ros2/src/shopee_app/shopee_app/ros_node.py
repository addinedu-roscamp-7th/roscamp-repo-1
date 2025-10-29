import threading
from typing import Optional

from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSignal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from shopee_interfaces.msg import PickeeRobotStatus


class ShopeeAppNode(Node):
    def __init__(self):
        super().__init__("shopee_app_node")
        # TODO: docs/InterfaceSpecification/App_vs_Main.md 에 정의된 통신 인터페이스를 구현한다.

        self._pickee_status_sub = self.create_subscription(
            PickeeRobotStatus, "/pickee/robot_status", self.on_pickee_status, 10
        )

    # Pickee 로봇 상태 토픽 콜백
    def on_pickee_status(self, msg: PickeeRobotStatus) -> None:
        print(f"Received PickeeRobotStatus: {msg}")


class RosNodeThread(QThread):
    node_ready = pyqtSignal()
    node_error = pyqtSignal(str)

    def __init__(self):
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
        try:
            try:
                rclpy.init(args=None)
                self._initialized = True
            except RuntimeError:
                self._initialized = False
            self._node = ShopeeAppNode()
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
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
        return self._ready_event.wait(timeout=timeout)

    def shutdown(self):
        self._shutdown_event.set()
