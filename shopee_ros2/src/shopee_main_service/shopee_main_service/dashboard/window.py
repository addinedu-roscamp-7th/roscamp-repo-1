"""
대시보드 메인 윈도우

PyQt6 기반 메인 윈도우 및 레이아웃을 구성한다.
"""

from datetime import datetime
from typing import Any, Dict

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from .panels import EventLogPanel, OrderPanel, RobotPanel


class DashboardWindow(QMainWindow):
    """
    대시보드 메인 윈도우

    로봇 상태, 주문, 이벤트 로그를 실시간으로 표시한다.
    """

    def __init__(self, bridge):
        """
        Args:
            bridge: DashboardBridge 인스턴스
        """
        super().__init__()
        self._bridge = bridge
        self._init_ui()
        self._start_polling()

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('Shopee Main Service Dashboard')
        self.setGeometry(100, 100, 1200, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 상단 스플리터 (로봇 패널 | 주문 패널)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.robot_panel = RobotPanel()
        self.order_panel = OrderPanel()

        top_splitter.addWidget(self.robot_panel)
        top_splitter.addWidget(self.order_panel)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)

        # 하단 이벤트 로그
        self.event_log_panel = EventLogPanel()

        # 전체 레이아웃
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.event_log_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(main_splitter)

        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('상태: 연결 대기 중...')
        self.status_bar.addPermanentWidget(self.status_label)

    def _start_polling(self):
        """브릿지에서 데이터를 폴링하는 타이머 시작"""
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_bridge)
        self.poll_timer.start(100)  # 100ms마다 폴링

    def _poll_bridge(self):
        """
        브릿지에서 데이터를 가져와 UI를 업데이트한다.
        """
        payload_count = 0
        while True:
            payload = self._bridge.get_for_gui(timeout=0.0)
            if payload is None:
                break

            payload_count += 1
            payload_type = payload.get('type')

            if payload_type == 'snapshot':
                self._handle_snapshot(payload.get('data', {}))
            elif payload_type == 'event':
                self._handle_event(payload.get('data', {}))

        # 디버깅: 최초 1번만 로그 출력
        if not hasattr(self, '_poll_logged'):
            self._poll_logged = True
            if payload_count > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f'Dashboard polling: received {payload_count} payloads')

    def _handle_snapshot(self, snapshot: Dict[str, Any]):
        """
        스냅샷 데이터를 처리하여 패널을 업데이트한다.

        Args:
            snapshot: DashboardDataProvider.collect_snapshot() 결과
        """
        # 디버깅: 스냅샷 구조 확인
        if not hasattr(self, '_structure_logged'):
            self._structure_logged = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f'Snapshot keys: {list(snapshot.keys())}')
            logger.info(f'Orders data type: {type(snapshot.get("orders"))}')
            logger.info(f'Orders data: {snapshot.get("orders")}')

        # 로봇 상태 업데이트
        robots = snapshot.get('robots', [])
        self.robot_panel.update_data(robots)

        # 주문 목록 업데이트
        orders = snapshot.get('orders', {})
        self.order_panel.update_data(orders)

        # 상태바 갱신
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        robot_count = len(robots)
        order_count = orders.get('summary', {}).get('total_active', 0)
        self.status_label.setText(
            f'상태: 연결됨 | 로봇: {robot_count}대 | 진행 중 주문: {order_count}건 | 마지막 갱신: {now}'
        )

        # 디버깅: 최초 1번만 로그 출력
        if not hasattr(self, '_snapshot_logged'):
            self._snapshot_logged = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f'Dashboard snapshot: {robot_count} robots, {order_count} orders')

    def _handle_event(self, event_data: Dict[str, Any]):
        """
        이벤트 데이터를 처리하여 로그에 추가한다.

        Args:
            event_data: EventBus 이벤트 페이로드
        """
        self.event_log_panel.add_event(event_data)

    def closeEvent(self, event):
        """
        창 닫기 이벤트 처리

        사용자가 창을 닫을 때 타이머를 정리한다.
        메인 서비스는 계속 실행된다.
        """
        self.poll_timer.stop()
        event.accept()
