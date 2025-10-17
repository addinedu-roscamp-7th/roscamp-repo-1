"""
대시보드 메인 윈도우

PyQt6 기반 탭 구조 메인 윈도우 및 레이아웃을 구성한다.
"""

from datetime import datetime
from typing import Any, Dict

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .tabs import (
    EventLogTab,
    OrderManagementTab,
    OverviewTab,
    RobotStatusTab,
    SystemDiagnosticsTab,
)


class DashboardWindow(QMainWindow):
    """
    대시보드 메인 윈도우 (탭 구조)

    5개 탭으로 시스템 정보를 분류하여 표시한다.
    - 탭1: 개요 (성능 메트릭스, 요약)
    - 탭2: 로봇 상태
    - 탭3: 주문 관리
    - 탭4: 시스템 진단
    - 탭5: 이벤트 로그
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
        self.setWindowTitle('Shopee Main Service Dashboard v5.0')
        self.setGeometry(100, 100, 1400, 900)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 탭 위젯 생성
        self.tab_widget = QTabWidget()

        # 각 탭 추가
        self.overview_tab = OverviewTab()
        self.robot_tab = RobotStatusTab()
        self.order_tab = OrderManagementTab()
        self.diagnostics_tab = SystemDiagnosticsTab()
        self.log_tab = EventLogTab()

        self.tab_widget.addTab(self.overview_tab, '개요')
        self.tab_widget.addTab(self.robot_tab, '로봇 상태')
        self.tab_widget.addTab(self.order_tab, '주문 관리')
        self.tab_widget.addTab(self.diagnostics_tab, '시스템 진단')
        self.tab_widget.addTab(self.log_tab, '이벤트 로그')

        main_layout.addWidget(self.tab_widget)

        # 상태바 (확장)
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
        스냅샷 데이터를 처리하여 모든 탭을 업데이트한다.

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

        # 각 탭 업데이트
        robots = snapshot.get('robots', [])
        orders = snapshot.get('orders', {})

        # 탭1: 개요 - 전체 스냅샷 전달
        self.overview_tab.update_data(snapshot)

        # 탭2: 로봇 상태
        self.robot_tab.update_data(robots)

        # 탭3: 주문 관리
        self.order_tab.update_data(orders)

        # 탭4: 시스템 진단 - 전체 스냅샷 전달
        self.diagnostics_tab.update_data(snapshot)

        # 상태바 확장 갱신
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        robot_count = len(robots)
        order_count = orders.get('summary', {}).get('total_active', 0)
        metrics = snapshot.get('metrics', {})
        app_sessions = metrics.get('network', {}).get('app_sessions', 0)

        self.status_label.setText(
            f'상태: 연결됨 | App 세션: {app_sessions} | 로봇: {robot_count}대 | '
            f'진행 중 주문: {order_count}건 | 마지막 갱신: {now}'
        )

        # 디버깅: 최초 1번만 로그 출력
        if not hasattr(self, '_snapshot_logged'):
            self._snapshot_logged = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f'Dashboard snapshot: {robot_count} robots, {order_count} orders')

    def _handle_event(self, event_data: Dict[str, Any]):
        """
        이벤트 데이터를 처리하여 이벤트 로그 탭 및 개요 탭에 추가한다.

        Args:
            event_data: EventBus 이벤트 페이로드
        """
        # 이벤트 로그 탭에 추가
        self.log_tab.add_event(event_data)

        # 개요 탭의 최근 알림에 추가
        self.overview_tab.add_alert(event_data)

    def closeEvent(self, event):
        """
        창 닫기 이벤트 처리

        사용자가 창을 닫을 때 타이머를 정리한다.
        메인 서비스는 계속 실행된다.
        """
        self.poll_timer.stop()
        event.accept()
