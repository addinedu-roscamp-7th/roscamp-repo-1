"""
대시보드 메인 윈도우

모듈화된 UI와 탭 구조를 로드하고 관리하는 메인 컨테이너.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

import rclpy
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QLabel
from PyQt6.QtGui import QCloseEvent

from .ui_gen.dashboard_window_ui import Ui_DashboardWindow
from .tabs.overview_tab import OverviewTab
from .tabs.robot_status_tab import RobotStatusTab
from .tabs.order_management_tab import OrderManagementTab
from .tabs.event_log_tab import EventLogTab
from .tabs.topic_monitor_tab import TopicMonitorTab
from .tabs.service_monitor_tab import ServiceMonitorTab
from .tabs.db_admin_tab import DBAdminTab
from .tabs.video_monitor_tab import VideoMonitorTab

logger = logging.getLogger(__name__)


class DashboardWindow(QMainWindow, Ui_DashboardWindow):
    """
    대시보드 메인 윈도우

    - dashboard_window_ui.py의 Ui_DashboardWindow를 상속받아 UI 뼈대 구성
    - 각 탭 위젯을 생성하고 QTabWidget에 추가
    - DashboardBridge로부터 데이터를 받아 각 탭에 분배
    """

    def __init__(self, bridge, ros_node, db_manager=None, streaming_service=None):
        super().__init__()
        self._bridge = bridge
        self._ros_node = ros_node
        self._db_manager = db_manager
        self._streaming_service = streaming_service

        # 메인 윈도우 UI 로드
        self.setupUi(self)
        self._init_tabs()
        self._configure_ui()

        # 데이터 폴링 시작
        self._start_polling()

        # ROS2 스핀 타이머 시작
        self._start_ros_spin()

    def _init_tabs(self):
        """각 탭 위젯을 생성하고 메인 탭 위젯에 추가한다."""
        self.overview_tab = OverviewTab()
        self.robot_tab = RobotStatusTab()
        self.order_tab = OrderManagementTab()
        # self.log_tab = EventLogTab()
        self.topic_monitor_tab = TopicMonitorTab()
        self.service_monitor_tab = ServiceMonitorTab()

        # DB 관리 탭은 db_manager가 있을 때만 추가
        self.db_admin_tab = None
        if self._db_manager:
            self.db_admin_tab = DBAdminTab(self._db_manager)

        # 영상 모니터링 탭은 streaming_service가 있을 때만 추가
        self.video_monitor_tab = None
        if self._streaming_service:
            self.video_monitor_tab = VideoMonitorTab(self._streaming_service)

        self.tab_widget.addTab(self.overview_tab, '개요')
        self.tab_widget.addTab(self.robot_tab, '로봇 상태')
        self.tab_widget.addTab(self.order_tab, '주문 관리')
        # self.tab_widget.addTab(self.log_tab, '이벤트 로그')
        self.tab_widget.addTab(self.topic_monitor_tab, 'ROS2 토픽 모니터')
        self.tab_widget.addTab(self.service_monitor_tab, 'ROS2 서비스 모니터')

        # 영상 모니터링 탭 추가 (있는 경우)
        if self.video_monitor_tab:
            self.tab_widget.addTab(self.video_monitor_tab, '영상 모니터링')

        # DB 관리 탭 추가 (있는 경우)
        if self.db_admin_tab:
            self.tab_widget.addTab(self.db_admin_tab, 'DB 관리')

    def _configure_ui(self):
        """UI의 추가적인 속성을 설정한다."""
        self.status_label = QLabel('상태: 연결 대기 중...')
        self.statusbar.addPermanentWidget(self.status_label)
        
        # 전체화면으로 시작
        self.showFullScreen()

    def closeEvent(self, event: QCloseEvent):
        """윈도우 종료 이벤트 처리"""
        # DB 관리 탭 정리
        if hasattr(self, 'db_admin_tab') and self.db_admin_tab:
            self.db_admin_tab.cleanup()
        event.accept()

    def _start_polling(self):
        """브릿지에서 데이터를 폴링하는 타이머 시작"""
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_bridge)
        self.poll_timer.start(100)  # 100ms마다 폴링

    def _start_ros_spin(self):
        """ROS2 노드를 주기적으로 스핀하는 타이머 시작"""
        self.ros_spin_timer = QTimer()
        self.ros_spin_timer.timeout.connect(self._spin_ros_node)
        self.ros_spin_timer.start(10)  # 10ms마다 ROS2 메시지 처리

    def _spin_ros_node(self):
        """ROS2 노드를 한 번 스핀하여 메시지를 처리한다."""
        try:
            if self._ros_node and rclpy.ok():
                rclpy.spin_once(self._ros_node, timeout_sec=0)
        except Exception as e:
            logger.error(f'ROS2 스핀 오류: {e}')

    def _poll_bridge(self):
        """브릿지에서 데이터를 가져와 UI를 업데이트한다."""
        from ..constants import GUI_QUEUE_TIMEOUT

        count = 0
        while True:
            payload = self._bridge.get_for_gui(timeout=GUI_QUEUE_TIMEOUT)
            if payload is None:
                break

            count += 1
            payload_type = payload.get('type')
            if payload_type == 'snapshot':
                self._handle_snapshot(payload.get('data', {}))
                # 스냅샷 로그 제거 (너무 많음)
            elif payload_type == 'event':
                self._handle_event(payload.get('data', {}))
                logger.info(f'Dashboard received event: {payload.get("data", {}).get("type", "unknown")}')
            elif payload_type == 'ros_topic':
                self._handle_ros_topic(payload.get('data', {}))
            elif payload_type == 'ros_service':
                self._handle_ros_service(payload.get('data', {}))

    def _handle_snapshot(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터를 처리하여 모든 탭을 업데이트한다."""
        self.overview_tab.update_data(snapshot)
        self.robot_tab.update_data(snapshot.get('robots', []))
        self.order_tab.update_data(snapshot.get('orders', {}))
        self._update_statusbar(snapshot)

    def _handle_event(self, event_data: Dict[str, Any]):
        """이벤트 데이터를 처리하여 로그 및 알림을 업데이트한다."""
        # self.log_tab.add_event(event_data)
        self.overview_tab.add_alert(event_data)

    def _handle_ros_topic(self, event_data: Dict[str, Any]):
        """ROS 토픽 수신 이벤트를 처리하여 토픽 모니터 탭에 추가한다."""
        self.topic_monitor_tab.add_ros_topic_event(event_data)
        
        # ROS 토픽을 이벤트 로그에도 추가 (이벤트 토픽만)
        if not event_data.get('is_periodic', False):
            self.overview_tab.add_ros_topic_event(event_data)

    def _handle_ros_service(self, event_data: Dict[str, Any]):
        """ROS 서비스 수신 이벤트를 처리하여 서비스 모니터 탭에 추가한다."""
        self.service_monitor_tab.handle_service_event(event_data)

    def _update_statusbar(self, snapshot: Dict[str, Any]):
        """상태바 업데이트"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        robot_count = len(snapshot.get('robots', []))
        order_count = snapshot.get('orders', {}).get('summary', {}).get('total_active', 0)
        app_sessions = snapshot.get('metrics', {}).get('network', {}).get('app_sessions', 0)
        
        # status_label을 직접 업데이트
        status_text = (
            f'상태: 연결됨 | App 세션: {app_sessions} | 로봇: {robot_count}대 | '
            f'진행 중 주문: {order_count}건 | 마지막 갱신: {now}'
        )
        self.status_label.setText(status_text)

    def closeEvent(self, event):
        """창 닫기 이벤트 처리"""
        # 타이머 중지
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        if hasattr(self, 'ros_spin_timer'):
            self.ros_spin_timer.stop()

        # 탭 리소스 정리
        if hasattr(self, 'db_admin_tab') and self.db_admin_tab:
            self.db_admin_tab.cleanup()
        if hasattr(self, 'video_monitor_tab') and self.video_monitor_tab:
            self.video_monitor_tab.cleanup()

        # ROS2 종료
        if rclpy.ok():
            logger.info('ROS2 종료 신호 전송')
            rclpy.shutdown()

        event.accept()
