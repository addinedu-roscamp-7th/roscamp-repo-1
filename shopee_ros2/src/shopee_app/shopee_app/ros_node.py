import threading
from typing import Any
from typing import Callable
from typing import Optional

from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSignal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.timer import Timer

from shopee_interfaces.msg import PickeeVisionDetection
from shopee_interfaces.msg import PickeeRobotStatus

PRODUCT_CATALOG = {
    1: "고추냉이",
    2: "불닭캔",
    3: "버터캔",
    4: "리챔",
    5: "두유",
    6: "카프리썬",
    7: "홍사과",
    8: "청사과",
    9: "오렌지",
    10: "삼겹살",
    11: "닭",
    12: "생선",
    13: "전복",
    14: "이클립스",
    15: "아이비",
    16: "빼빼로",
    17: "오예스",
}


class ShopeeAppNode(Node):
    """Pickee 로봇 상태를 구독해 UI에 전달하는 ROS2 노드."""

    def __init__(self):
        """필요한 ROS 엔티티를 초기화하고 상태 토픽을 구독한다."""
        super().__init__('shopee_app_node')
        self._status_listeners: list[Callable[[PickeeRobotStatus], None]] = []
        self._detection_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._detection_subscription: Subscription | None = None
        self._status_subscription: Subscription | None = None
        self._detection_subscription_timer: Timer | None = None

        self._status_subscription = self.create_subscription(
            PickeeRobotStatus, '/pickee/robot_status', self.on_pickee_status, 10
        )

        self.get_logger().info('pickee/vision/detection_result 구독은 사용자 창 전환 시 활성화한다')

    def set_detection_subscription_enabled(self, enabled: bool) -> None:
        """사용자 화면 상태에 맞춰 비전 결과 구독을 동적으로 제어한다."""
        if enabled:
            if self._detection_subscription is not None:
                return
            self._detection_subscription = self.create_subscription(
                PickeeVisionDetection,
                'pickee/vision/detection_result',
                self.on_selection_bbox,
                10,
            )
            self.get_logger().info('pickee/vision/detection_result 구독을 시작했다')
            self._start_detection_subscription_timer()
            return

        if self._detection_subscription is None:
            return
        self.destroy_subscription(self._detection_subscription)
        self._detection_subscription = None
        self._stop_detection_subscription_timer()
        self.get_logger().info('pickee/vision/detection_result 구독을 중지했다')

    def on_pickee_status(self, msg: PickeeRobotStatus) -> None:
        """수신한 로봇 상태 메시지를 등록된 리스너들에게 전달한다."""
        for callback in list(self._status_listeners):
            try:
                callback(msg)
            except Exception:
                self.get_logger().exception("Pickee status listener 실패")

    def add_status_listener(
        self, callback: Callable[[PickeeRobotStatus], None]
    ) -> None:
        """중복을 방지하며 상태 콜백을 등록한다."""
        if callback not in self._status_listeners:
            self._status_listeners.append(callback)

    def remove_status_listener(
        self, callback: Callable[[PickeeRobotStatus], None]
    ) -> None:
        """등록된 상태 콜백을 안전하게 제거한다."""
        if callback in self._status_listeners:
            self._status_listeners.remove(callback)

    def add_detection_listener(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """비전 결과를 전달할 콜백을 등록한다."""
        if callback not in self._detection_listeners:
            self._detection_listeners.append(callback)

    def remove_detection_listener(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """등록된 비전 콜백을 안전하게 제거한다."""
        if callback in self._detection_listeners:
            self._detection_listeners.remove(callback)

    def on_selection_bbox(self, msg: PickeeVisionDetection) -> None:
        products_payload: list[dict[str, Any]] = []
        for product in msg.products:
            bbox = product.bbox
            bbox_number = product.bbox_number
            product_name = PRODUCT_CATALOG.get(product.product_id, '미등록상품')
            product_id = product.product_id

            self.get_logger().info(f'상품명: {product_name}, ID:{product_id}')
            self.get_logger().info(f'bbox번호: {bbox_number}')
            self.get_logger().info(
                f'BBox={bbox.x1}, {bbox.y1}, -> {bbox.x2}, {bbox.y2}'
            )
            try:
                bbox_data = {
                    'x1': int(bbox.x1),
                    'y1': int(bbox.y1),
                    'x2': int(bbox.x2),
                    'y2': int(bbox.y2),
                }
            except (TypeError, ValueError):
                continue
            payload = {
                'product_id': int(product_id),
                'product_name': product_name,
                'bbox_number': int(bbox_number),
                'bbox': bbox_data,
            }
            products_payload.append(payload)
        try:
            robot_id_value = int(msg.robot_id)
        except (TypeError, ValueError):
            robot_id_value = -1
        try:
            order_id_value = int(msg.order_id)
        except (TypeError, ValueError):
            order_id_value = -1
        detection_payload = {
            'robot_id': robot_id_value,
            'order_id': order_id_value,
            'products': products_payload,
        }
        self._notify_detection_listeners(detection_payload)

    def _notify_detection_listeners(self, payload: dict[str, Any]) -> None:
        """UI 쓰레드로 비전 결과를 안전하게 전달한다."""
        for callback in list(self._detection_listeners):
            try:
                callback(payload)
            except Exception:
                self.get_logger().exception('Pickee detection listener 실패')

    def _log_detection_subscription(self) -> None:
        """비전 토픽 구독이 활성화된 동안 상태 로그를 출력한다."""
        if self._detection_subscription is None:
            return
        self.get_logger().info('pickee/vision/detection_result 구독 유지 중')

    def _start_detection_subscription_timer(self) -> None:
        if self._detection_subscription_timer is not None:
            return
        self._detection_subscription_timer = self.create_timer(
            5.0, self._log_detection_subscription
        )

    def _stop_detection_subscription_timer(self) -> None:
        if self._detection_subscription_timer is None:
            return
        self.destroy_timer(self._detection_subscription_timer)
        self._detection_subscription_timer = None


class RosNodeThread(QThread):
    """GUI와 분리된 스레드에서 ROS2 이벤트 루프를 실행한다."""

    node_ready = pyqtSignal()
    node_error = pyqtSignal(str)
    pickee_status_received = pyqtSignal(object)
    pickee_detection_received = pyqtSignal(object)

    def __init__(self):
        """스레드 동기화를 위한 이벤트와 실행기를 준비한다."""
        super().__init__()
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()
        self._executor: Optional[SingleThreadedExecutor] = None
        self._node: Optional[ShopeeAppNode] = None
        self._initialized = False
        self._pending_detection_enabled = False

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
            self._node.add_status_listener(self._handle_pickee_status)
            self._node.add_detection_listener(self._handle_pickee_detection)
            self._apply_detection_subscription_state()
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
                self._node.remove_detection_listener(self._handle_pickee_detection)
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
        """노드 준비가 완료될 때까지 대기하거나 타임아웃을 반환한다."""
        return self._ready_event.wait(timeout=timeout)

    def shutdown(self):
        """외부에서 종료 신호를 받아 스핀 루프를 멈춘다."""
        self._shutdown_event.set()

    def set_detection_subscription_enabled(self, enabled: bool) -> None:
        """UI 스레드에서 호출해도 안전하게 비전 구독 상태를 전환한다."""
        self._pending_detection_enabled = enabled
        executor = self._executor
        if executor is None:
            return

        def _toggle() -> None:
            if self._node is None:
                return
            self._node.set_detection_subscription_enabled(
                self._pending_detection_enabled
            )

        executor.create_task(_toggle)

    def _apply_detection_subscription_state(self) -> None:
        if self._node is None:
            return
        self._node.set_detection_subscription_enabled(self._pending_detection_enabled)

    def _handle_pickee_detection(self, payload: dict[str, Any]) -> None:
        """비전 감지 결과를 PyQt 시그널로 전달한다."""
        self.pickee_detection_received.emit(payload)

    def _handle_pickee_status(self, msg: PickeeRobotStatus) -> None:
        """선택된 콜백 대신 PyQt 시그널로 상태 메시지를 전달한다."""
        self.pickee_status_received.emit(msg)
