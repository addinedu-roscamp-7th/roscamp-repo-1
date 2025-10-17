"""
대시보드 탭 위젯

각 탭별 UI 컴포넌트를 정의한다.
"""

from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .panels import EventLogPanel, OrderPanel, RobotPanel


class OverviewTab(QWidget):
    """
    탭 1: 개요 (Overview)

    시스템 전체 성능과 핵심 정보를 한눈에 표시한다.
    """

    def __init__(self):
        super().__init__()
        self._recent_alerts: List[Dict[str, Any]] = []  # 최근 알림 저장 (최대 5건)
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 시스템 성능 메트릭스 카드
        self.metrics_group = QGroupBox('시스템 성능 메트릭스')
        metrics_layout = QGridLayout()

        # 메트릭 레이블 초기화
        self.avg_time_label = QLabel('평균 처리 시간: -')
        self.throughput_label = QLabel('시간당 처리량: -')
        self.success_rate_label = QLabel('성공률: -')
        self.robot_util_label = QLabel('로봇 활용률: -')
        self.system_load_label = QLabel('시스템 부하: -')

        metrics_layout.addWidget(self.avg_time_label, 0, 0)
        metrics_layout.addWidget(self.throughput_label, 0, 1)
        metrics_layout.addWidget(self.success_rate_label, 0, 2)
        metrics_layout.addWidget(self.robot_util_label, 1, 0)
        metrics_layout.addWidget(self.system_load_label, 1, 1)

        self.metrics_group.setLayout(metrics_layout)
        layout.addWidget(self.metrics_group)

        # 하단 스플리터 (로봇 요약 | 주문 요약)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 활성 로봇 요약
        self.robot_summary_group = QGroupBox('활성 로봇 요약')
        robot_summary_layout = QVBoxLayout()
        self.robot_summary_label = QLabel('로딩 중...')
        robot_summary_layout.addWidget(self.robot_summary_label)
        self.robot_summary_group.setLayout(robot_summary_layout)

        # 진행 중 주문 요약
        self.order_summary_group = QGroupBox('진행 중 주문 요약')
        order_summary_layout = QVBoxLayout()
        self.order_summary_label = QLabel('로딩 중...')
        order_summary_layout.addWidget(self.order_summary_label)
        self.order_summary_group.setLayout(order_summary_layout)

        bottom_splitter.addWidget(self.robot_summary_group)
        bottom_splitter.addWidget(self.order_summary_group)

        layout.addWidget(bottom_splitter)

        # 최근 알림
        self.alerts_group = QGroupBox('최근 알림 (최신 5건)')
        alerts_layout = QVBoxLayout()
        self.alerts_label = QLabel('알림 없음')
        alerts_layout.addWidget(self.alerts_label)
        self.alerts_group.setLayout(alerts_layout)

        layout.addWidget(self.alerts_group)

        # 비율 설정
        layout.setStretchFactor(self.metrics_group, 1)
        layout.setStretchFactor(bottom_splitter, 2)
        layout.setStretchFactor(self.alerts_group, 1)

    def update_data(self, snapshot: Dict[str, Any]):
        """
        스냅샷 데이터로 개요 탭을 업데이트한다.

        Args:
            snapshot: 전체 시스템 스냅샷
        """
        metrics = snapshot.get('metrics', {})

        # 성능 메트릭스 업데이트
        avg_time = metrics.get('avg_processing_time', 0)
        throughput = metrics.get('hourly_throughput', 0)
        success_rate = metrics.get('success_rate', 0)
        robot_util = metrics.get('robot_utilization', 0)
        system_load = metrics.get('system_load', 0)

        self.avg_time_label.setText(f'평균 처리 시간: {avg_time:.1f}s')
        self.throughput_label.setText(f'시간당 처리량: {throughput}건')
        self.success_rate_label.setText(f'성공률: {success_rate:.1f}%')
        self.robot_util_label.setText(f'로봇 활용률: {robot_util:.1f}%')
        self.system_load_label.setText(f'시스템 부하: {system_load:.1f}%')

        # 로봇 요약 업데이트
        robots = snapshot.get('robots', [])

        # robots가 RobotState 객체 리스트인지 딕셔너리 리스트인지 확인
        if robots and hasattr(robots[0], 'robot_type'):
            # RobotState 객체인 경우
            pickee_list = [r for r in robots if r.robot_type == 'PICKEE']
            packee_list = [r for r in robots if r.robot_type == 'PACKEE']

            pickee_working = sum(1 for r in pickee_list if r.status == 'WORKING')
            pickee_idle = sum(1 for r in pickee_list if r.status == 'IDLE')
            pickee_error = sum(1 for r in pickee_list if r.status == 'ERROR')

            packee_working = sum(1 for r in packee_list if r.status == 'WORKING')
            packee_idle = sum(1 for r in packee_list if r.status == 'IDLE')
            packee_offline = sum(1 for r in packee_list if r.status == 'OFFLINE')
        else:
            # 딕셔너리인 경우 (현재 구현)
            pickee_list = [r for r in robots if r.get('robot_type') == 'PICKEE']
            packee_list = [r for r in robots if r.get('robot_type') == 'PACKEE']

            pickee_working = sum(1 for r in pickee_list if r.get('status') == 'WORKING')
            pickee_idle = sum(1 for r in pickee_list if r.get('status') == 'IDLE')
            pickee_error = sum(1 for r in pickee_list if r.get('status') == 'ERROR')

            packee_working = sum(1 for r in packee_list if r.get('status') == 'WORKING')
            packee_idle = sum(1 for r in packee_list if r.get('status') == 'IDLE')
            packee_offline = sum(1 for r in packee_list if r.get('status') == 'OFFLINE')

        robot_summary_text = f"""
Pickee: {len(pickee_list)}대
├─ WORKING: {pickee_working}
├─ IDLE: {pickee_idle}
└─ ERROR: {pickee_error}

Packee: {len(packee_list)}대
├─ WORKING: {packee_working}
├─ IDLE: {packee_idle}
└─ OFFLINE: {packee_offline}
        """.strip()

        self.robot_summary_label.setText(robot_summary_text)

        # 주문 요약 업데이트
        orders = snapshot.get('orders', {})
        order_summary = orders.get('summary', {})

        order_summary_text = f"""
진행 중 주문: {order_summary.get('total_active', 0)}건
평균 진행률: {order_summary.get('avg_progress', 0):.0f}%

최근 1시간 완료: {metrics.get('hourly_throughput', 0)}건
실패: {order_summary.get('failed_count', 0)}건
        """.strip()

        self.order_summary_label.setText(order_summary_text)

        # 최근 알림 업데이트
        self._update_alerts_display()

    def add_alert(self, event_data: Dict[str, Any]):
        """
        새로운 알림을 추가한다.

        Args:
            event_data: 이벤트 데이터
        """
        # 중요한 이벤트만 알림으로 표시
        event_type = event_data.get('type', '')

        # 알림으로 표시할 이벤트 타입 필터링
        alert_types = {
            'robot_failure': '오류',
            'robot_timeout_notification': '오류',
            'robot_reassignment_notification': '경고',
            'reservation_timeout': '경고',
            'packing_info_notification': '정보',
            'cart_update_notification': '정보',
        }

        if event_type not in alert_types:
            return

        # 알림 데이터 구성
        now = datetime.now()
        alert_level = alert_types[event_type]
        message = event_data.get('message', event_type)

        # 레벨별 아이콘
        icon_map = {
            '오류': '\U0001f534',  # 🔴
            '경고': '\U0001f7e1',  # 🟡
            '정보': '\U0001f7e2',  # 🟢
        }
        icon = icon_map.get(alert_level, '\u2139')  # ℹ

        alert = {
            'timestamp': now,
            'level': alert_level,
            'icon': icon,
            'message': message,
            'event_type': event_type,
        }

        # 최근 알림 목록에 추가 (최대 5건)
        self._recent_alerts.insert(0, alert)
        if len(self._recent_alerts) > 5:
            self._recent_alerts = self._recent_alerts[:5]

        # 화면 업데이트
        self._update_alerts_display()

    def _update_alerts_display(self):
        """알림 목록을 화면에 표시한다."""
        if not self._recent_alerts:
            self.alerts_label.setText('알림 없음')
            return

        # 알림 텍스트 생성
        alert_lines = []
        for alert in self._recent_alerts:
            timestamp_str = alert['timestamp'].strftime('%H:%M:%S')
            icon = alert['icon']
            message = alert['message']
            # 메시지가 너무 길면 줄임
            if len(message) > 60:
                message = message[:57] + '...'
            alert_lines.append(f'{icon} [{timestamp_str}] {message}')

        self.alerts_label.setText('\n'.join(alert_lines))


class RobotStatusTab(QWidget):
    """
    탭 2: 로봇 상태 (Robot Status)

    전체 로봇의 상세 상태를 표시한다.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 기존 RobotPanel 재사용
        self.robot_panel = RobotPanel()
        layout.addWidget(self.robot_panel)

    def update_data(self, robots: List[Any]):
        """
        로봇 데이터로 탭을 업데이트한다.

        Args:
            robots: 로봇 상태 목록
        """
        self.robot_panel.update_data(robots)


class OrderManagementTab(QWidget):
    """
    탭 3: 주문 관리 (Order Management)

    진행 중인 주문의 상세 정보를 표시한다.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 기존 OrderPanel 재사용
        self.order_panel = OrderPanel()
        layout.addWidget(self.order_panel)

    def update_data(self, orders: Dict[str, Any]):
        """
        주문 데이터로 탭을 업데이트한다.

        Args:
            orders: 주문 데이터 딕셔너리
        """
        self.order_panel.update_data(orders)


class SystemDiagnosticsTab(QWidget):
    """
    탭 4: 시스템 진단 (System Diagnostics)

    에러 추적 및 네트워크 상태를 표시한다.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 에러 및 장애 추적
        self.error_group = QGroupBox('에러 및 장애 추적')
        error_layout = QVBoxLayout()
        self.error_label = QLabel('로딩 중...')
        error_layout.addWidget(self.error_label)
        self.error_group.setLayout(error_layout)

        # 네트워크 및 연결 상태
        self.network_group = QGroupBox('네트워크 및 연결 상태')
        network_layout = QVBoxLayout()
        self.network_label = QLabel('로딩 중...')
        network_layout.addWidget(self.network_label)
        self.network_group.setLayout(network_layout)

        layout.addWidget(self.error_group)
        layout.addWidget(self.network_group)

    def update_data(self, snapshot: Dict[str, Any]):
        """
        스냅샷 데이터로 진단 탭을 업데이트한다.

        Args:
            snapshot: 전체 시스템 스냅샷
        """
        metrics = snapshot.get('metrics', {})

        # 에러 정보 업데이트
        failed_by_reason = metrics.get('failed_orders_by_reason', {})
        failed_orders = metrics.get('failed_orders', [])
        error_robots = metrics.get('error_robots', [])
        offline_robots = metrics.get('offline_robots', [])
        llm_stats = metrics.get('llm_stats', {})
        ros_retry_count = metrics.get('ros_retry_count', 0)

        error_lines: List[str] = ['최근 실패 주문 (30분 이내)']
        if failed_by_reason:
            for reason, count in failed_by_reason.items():
                error_lines.append(f'  {reason}: {count}건')
        else:
            error_lines.append('  실패 없음')

        if failed_orders:
            latest = failed_orders[0]
            latest_reason = latest.get('failure_reason') or 'UNKNOWN'
            error_lines.append(f'  최신 실패: #{latest.get("order_id")} ({latest_reason})')

        error_lines.append('')
        if error_robots:
            error_lines.append('로봇 오류 상태')
            for robot in error_robots:
                error_lines.append(
                    f"  #{robot.get('robot_id')} ({robot.get('robot_type')}) - {robot.get('status')}"
                )
        else:
            error_lines.append('로봇 오류 상태: 없음')

        if offline_robots:
            error_lines.append('')
            error_lines.append('오프라인 로봇')
            for robot in offline_robots:
                error_lines.append(
                    f"  #{robot.get('robot_id')} ({robot.get('robot_type')}) 마지막 갱신: {robot.get('last_update', '-')}"
                )
        else:
            error_lines.append('')
            error_lines.append('오프라인 로봇: 없음')

        llm_success = llm_stats.get('success_rate', 0.0)
        llm_latency = llm_stats.get('avg_response_time', 0.0)
        fallback_count = llm_stats.get('fallback_count', 0)
        failure_count = llm_stats.get('failure_count', 0)
        error_lines.append('')
        error_lines.append(
            f'LLM 성공률: {llm_success:.1f}% (응답 {llm_latency:.1f}ms, 폴백 {fallback_count}회, 실패 {failure_count}회)'
        )
        error_lines.append(f'ROS 서비스 재시도: {ros_retry_count}회')

        self.error_label.setText('\n'.join(error_lines))

        # 네트워크 정보 업데이트
        network = metrics.get('network', {})

        network_lines = [
            f'App 세션: {network.get("app_sessions", 0)} / {network.get("app_sessions_max", 200)}'
        ]

        topic_health = network.get('ros_topic_health', {})
        if topic_health:
            unhealthy_topics = [name for name, healthy in topic_health.items() if not healthy]
            if unhealthy_topics:
                problem_topics = ', '.join(unhealthy_topics)
                network_lines.append(
                    f'ROS 토픽: 오류 ({problem_topics})'
                )
            else:
                receive_rate = network.get('topic_receive_rate', 0.0)
                network_lines.append(f'ROS 토픽: 정상 ({receive_rate:.1f}% 수신)')
        else:
            network_lines.append('ROS 토픽: 정보 없음')

        event_activity = network.get('event_topic_activity', {})
        if event_activity:
            overdue_topics = [name for name, info in event_activity.items() if info.get('overdue')]
            if overdue_topics:
                joined_overdue = ', '.join(overdue_topics)
                network_lines.append(
                    f'이벤트 토픽 지연: {joined_overdue}'
                )
            event_summaries = []
            for name, info in event_activity.items():
                seconds_since_last = info.get('seconds_since_last')
                if seconds_since_last is None:
                    event_summaries.append(f'{name}=미수신')
                else:
                    event_summaries.append(f'{name}={seconds_since_last:.1f}초')
            if event_summaries:
                network_lines.append('이벤트 토픽 최신 수신: ' + ', '.join(event_summaries))

        network_lines.append(f'LLM 응답시간: {network.get("llm_response_time", 0.0):.1f}ms')
        network_lines.append(
            f'DB 커넥션: {network.get("db_connections", 0)} / {network.get("db_connections_max", 10)}'
        )

        self.network_label.setText('\n'.join(network_lines))


class EventLogTab(QWidget):
    """
    탭 5: 이벤트 로그 (Event Log)

    전체 이벤트 히스토리 및 검색 기능을 제공한다.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 기존 EventLogPanel 재사용
        self.event_log_panel = EventLogPanel()
        layout.addWidget(self.event_log_panel)

    def add_event(self, event_data: Dict[str, Any]):
        """
        이벤트를 로그에 추가한다.

        Args:
            event_data: 이벤트 데이터
        """
        self.event_log_panel.add_event(event_data)
