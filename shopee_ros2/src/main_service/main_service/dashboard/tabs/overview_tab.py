"""'개요' 탭의 UI 로직"""
from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QLabel, QTreeWidgetItem

from ..ui_gen.tab_overview_ui import Ui_OverviewTab
from .base_tab import BaseTab


class OverviewTab(BaseTab, Ui_OverviewTab):
    """'개요' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # 이벤트 테이블 모델 초기화
        self.event_model = QStandardItemModel(self)
        self.event_model.setHorizontalHeaderLabels(["시간", "유형", "메시지"])
        self.event_table_view.setModel(self.event_model)
        self.event_table_view.setColumnWidth(0, 100) # 시간
        self.event_table_view.setColumnWidth(1, 80)  # 유형
        self.event_table_view.setColumnWidth(2, 500) # 메시지

        # 상태 트리 위젯 초기화
        self.status_tree_widget.setHeaderLabels(["항목", "값"])
        self.status_tree_widget.setColumnWidth(0, 180)
        self.robot_status_item = QTreeWidgetItem(self.status_tree_widget, ["🤖 로봇 현황"])
        self.order_status_item = QTreeWidgetItem(self.status_tree_widget, ["📦 주문 현황"])
        self.diagnostics_item = QTreeWidgetItem(self.status_tree_widget, ["🩺 시스템 진단"])
        self.robot_status_item.setExpanded(True)
        self.order_status_item.setExpanded(True)
        self.diagnostics_item.setExpanded(True)

    def _normalize_robot_type(self, robot_type) -> str:
        """
        로봇 타입을 정규화한다.

        RobotType Enum이나 문자열 모두 처리하여 소문자 문자열로 반환한다.
        """
        if robot_type is None:
            return 'unknown'

        # Enum 객체인 경우
        if hasattr(robot_type, 'value'):
            return str(robot_type.value).lower()

        # 문자열인 경우
        return str(robot_type).lower()

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 개요 탭을 업데이트한다."""
        metrics = snapshot.get('metrics', {})
        
        # 상단 메트릭스 업데이트 (프로그레스 바 및 라벨)
        avg_time = metrics.get('avg_processing_time', 0)
        self.avg_processing_time_label.setText(f"평균 처리 시간: {avg_time:.1f}초")
        self._apply_color_threshold(self.avg_processing_time_label, avg_time, {
            'good': (0, 2.0), 'warning': (2.0, 5.0), 'critical': (5.0, float('inf'))
        })
        
        throughput = metrics.get('hourly_throughput', 0)
        self.hourly_throughput_label.setText(f"시간당 처리량: {throughput}건")
        self.hourly_throughput_label.setStyleSheet("")

        success_rate = metrics.get('success_rate', 0)
        self.success_rate_progress.setValue(int(success_rate))
        self.success_rate_progress.setFormat(f'{success_rate:.1f}%')
        self._apply_progress_bar_color(self.success_rate_progress, success_rate, {
            'good': (95, 101), 'warning': (90, 95), 'critical': (0, 90)
        })

        robot_util = metrics.get('robot_utilization', 0)
        self.robot_utilization_progress.setValue(int(robot_util))
        self.robot_utilization_progress.setFormat(f'{robot_util:.1f}%')
        self._apply_progress_bar_color(self.robot_utilization_progress, robot_util, {
            'good': (60, 101), 'warning': (30, 60), 'critical': (0, 30)
        })

        system_load = metrics.get('system_load', 0)
        self.system_load_progress.setValue(int(system_load))
        self.system_load_progress.setFormat(f'{system_load:.1f}%')
        self._apply_progress_bar_color(self.system_load_progress, system_load, {
            'good': (0, 70), 'warning': (70, 90), 'critical': (90, 101)
        })
        
        active_orders = metrics.get('active_orders', 0)
        self.active_orders_label.setText(f"활성 주문: {active_orders}건")

        # --- 시스템 상태 요약 트리 업데이트 ---
        # 로봇 현황 업데이트
        robots = snapshot.get('robots', [])
        pickee_list = [r for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'pickee']
        packee_list = [r for r in robots if self._normalize_robot_type(r.get('robot_type')) == 'packee']
        
        # 업데이트 전, 확장 상태 저장
        expansion_states = {}
        for i in range(self.robot_status_item.childCount()):
            child = self.robot_status_item.child(i)
            key = child.text(0).split(' ')[0]
            expansion_states[key] = child.isExpanded()

        self.robot_status_item.takeChildren() # 기존 자식 아이템 삭제
        self._update_robot_tree_item(self.robot_status_item, "Pickee", pickee_list)
        self._update_robot_tree_item(self.robot_status_item, "Packee", packee_list)

        # 업데이트 후, 확장 상태 복원
        for i in range(self.robot_status_item.childCount()):
            child = self.robot_status_item.child(i)
            key = child.text(0).split(' ')[0]
            if key in expansion_states:
                child.setExpanded(expansion_states[key])

        # 주문 현황 업데이트
        orders = snapshot.get('orders', {})
        order_summary = orders.get('summary', {})
        self.order_status_item.takeChildren() # 기존 자식 아이템 삭제
        QTreeWidgetItem(self.order_status_item, ["진행 중 주문", f"{order_summary.get('total_active', 0)}건"])
        QTreeWidgetItem(self.order_status_item, ["평균 진행률", f"{order_summary.get('avg_progress', 0):.0f}%"])
        QTreeWidgetItem(self.order_status_item, ["최근 1시간 완료", f"{throughput}건"])
        QTreeWidgetItem(self.order_status_item, ["실패", f"{order_summary.get('failed_count', 0)}건"])

        # 시스템 진단 업데이트
        self._update_diagnostics_tree(metrics)

    def _update_robot_tree_item(self, parent_item: QTreeWidgetItem, robot_type_name: str, robot_list: list):
        """로봇 현황 트리 아이템을 업데이트하는 헬퍼 함수"""
        if not robot_list:
            QTreeWidgetItem(parent_item, [f"{robot_type_name} (전체 0대)"])
            return

        active_robots = [r for r in robot_list if r.get('status') != 'OFFLINE']
        error_count = sum(1 for r in active_robots if r.get('status') == 'ERROR')

        type_item = QTreeWidgetItem(parent_item, [f"{robot_type_name} (활성 {len(active_robots)}/{len(robot_list)}대)"])
        QTreeWidgetItem(type_item, ["WORKING", f"{sum(1 for r in active_robots if r.get('status') == 'WORKING')}대"])
        QTreeWidgetItem(type_item, ["IDLE", f"{sum(1 for r in active_robots if r.get('status') == 'IDLE')}대"])
        error_item = QTreeWidgetItem(type_item, ["ERROR", f"{error_count}대"])
        QTreeWidgetItem(type_item, ["OFFLINE", f"{len(robot_list) - len(active_robots)}대"])

        if error_count > 0:
            error_item.setForeground(1, Qt.GlobalColor.red)
            type_item.setForeground(0, Qt.GlobalColor.red)

    def _update_diagnostics_tree(self, metrics: Dict[str, Any]):
        """시스템 진단 트리 아이템을 업데이트한다."""
        self.diagnostics_item.takeChildren()

        # 실패/오류 섹션
        failures_item = QTreeWidgetItem(self.diagnostics_item, ["실패/오류"])
        failed_by_reason = metrics.get('failed_orders_by_reason', {})
        if failed_by_reason:
            reason_text = ', '.join(f'{reason}: {count}건' for reason, count in failed_by_reason.items())
        else:
            reason_text = '없음'
        QTreeWidgetItem(failures_item, ["최근 실패 주문(60분)", reason_text])

        failed_orders = metrics.get('failed_orders', [])
        if failed_orders:
            for order in failed_orders:
                ended_at = order.get('ended_at')
                ended_text = ended_at[11:19] if isinstance(ended_at, str) and 'T' in ended_at else '-'
                amount = order.get('total_price')
                amount_text = f'₩{int(amount):,}' if amount else '-'
                order_line = f"#{order.get('order_id', '-')} / 사유={order.get('failure_reason', 'UNKNOWN')} / 금액={amount_text} / 종료={ended_text}"
                QTreeWidgetItem(failures_item, [order_line])

        # LLM 상태를 트리 구조로 변경
        llm_stats = metrics.get('llm_stats', {})
        llm_item = QTreeWidgetItem(failures_item, ["LLM 상태"])
        QTreeWidgetItem(llm_item, ["성공률", f"{llm_stats.get('success_rate', 0.0):.1f}%"])
        QTreeWidgetItem(llm_item, ["평균 응답 시간", f"{llm_stats.get('avg_response_time', 0.0):.1f}ms"])
        QTreeWidgetItem(llm_item, ["폴백 횟수", f"{llm_stats.get('fallback_count', 0)}회"])
        QTreeWidgetItem(llm_item, ["실패 횟수", f"{llm_stats.get('failure_count', 0)}회"])
        llm_item.setExpanded(True)

        QTreeWidgetItem(failures_item, ["ROS 서비스 재시도", f"{metrics.get('ros_retry_count', 0)}회"])

        # 로봇 장애 섹션
        error_robots_item = QTreeWidgetItem(self.diagnostics_item, ["로봇 장애"])
        error_robots = metrics.get('error_robots', [])
        if not error_robots:
            QTreeWidgetItem(error_robots_item, ["없음"])
        else:
            for robot in error_robots:
                line = self._format_robot_line(robot, '오류')
                QTreeWidgetItem(error_robots_item, [line])

        # 네트워크/연결 섹션
        network_item = QTreeWidgetItem(self.diagnostics_item, ["네트워크/연결"])
        network = metrics.get('network', {})
        app_sessions = network.get('app_sessions', 0)
        app_max = network.get('app_sessions_max', 200)
        QTreeWidgetItem(network_item, ["App 세션", f"{app_sessions} / {app_max}"])
        
        llm_response = network.get('llm_response_time', 0)
        QTreeWidgetItem(network_item, ["LLM 응답 시간", f"{llm_response:.0f}ms"])

        failures_item.setExpanded(True)
        error_robots_item.setExpanded(True)
        network_item.setExpanded(True)

    @staticmethod
    def _format_robot_line(robot: Dict[str, Any], label: str) -> str:
        """로봇 장애 정보를 문자열로 변환한다."""
        robot_id = robot.get('robot_id', '-')
        robot_type = robot.get('robot_type', '-')
        status = robot.get('status', '-')
        last_update = robot.get('last_update')
        if isinstance(last_update, str) and 'T' in last_update:
            last_seen = last_update[11:19]
        else:
            last_seen = '-'
        return f"#{robot_id} ({robot_type}) [{label}] 상태={status} / 마지막 갱신={last_seen}"

    def add_alert(self, event_data: Dict[str, Any]):
        """최근 알림을 테이블에 추가"""
        from datetime import datetime
        from PyQt6.QtGui import QIcon

        alert_types = {'robot_failure': '오류', 'robot_timeout_notification': '타임아웃'}
        event_type = event_data.get('type', '')
        # if event_type not in alert_types:
        #     return

        icon_map = {'오류': '🔴', '타임아웃': '🟠'} # 아이콘 대신 텍스트 아이콘 사용
        type_str = alert_types.get(event_type, '정보')
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        message = event_data.get('message', '')

        # QStandardItem 리스트 생성
        time_item = QStandardItem(timestamp)
        type_item = QStandardItem(f"{icon_map.get(type_str, '🔵')} {type_str}")
        message_item = QStandardItem(message)

        # 모델에 행 추가
        self.event_model.insertRow(0, [time_item, type_item, message_item])

        # 모델의 행 수가 100개를 초과하면 가장 오래된 행 삭제
        if self.event_model.rowCount() > 100:
            self.event_model.removeRow(100)

    def add_ros_topic_event(self, event_data: Dict[str, Any]):
        """ROS 토픽 이벤트를 로그 테이블에 추가한다."""
        from datetime import datetime

        timestamp = datetime.fromtimestamp(event_data.get('timestamp', datetime.now().timestamp()))
        topic_name = event_data.get('topic_name', 'unknown')
        msg_data = event_data.get('msg', {})
        
        event_type = topic_name.strip('/').replace('/', ' ')
        
        message_parts = []
        for key, value in msg_data.items():
            message_parts.append(f'{key}={value}')
        message = ', '.join(message_parts)

        level = 'INFO'
        if 'fail' in event_type.lower() or 'error' in event_type.lower():
            level = 'ERROR'
        elif 'warn' in event_type.lower():
            level = 'WARN'

        time_item = QStandardItem(timestamp.strftime('%H:%M:%S'))
        type_item = QStandardItem(f"🔵 {level}")
        message_item = QStandardItem(f"[{event_type}] {message}")

        if level == 'ERROR':
            type_item.setText(f"🔴 {level}")
        elif level == 'WARN':
            type_item.setText(f"🟠 {level}")

        self.event_model.insertRow(0, [time_item, type_item, message_item])
        if self.event_model.rowCount() > 100:
            self.event_model.removeRow(100)

    def _apply_color_threshold(self, label: QLabel, value: float, thresholds: Dict[str, tuple]) -> None:
        """
        임계값에 따라 라벨 색상을 적용한다.
        
        Args:
            label: 색상을 적용할 QLabel
            value: 현재 값
            thresholds: {'good': (min, max), 'warning': (min, max), 'critical': (min, max)}
        """
        color = 'black'  # 기본 색상
        
        if 'good' in thresholds:
            min_val, max_val = thresholds['good']
            if min_val <= value < max_val:
                color = 'green'
        
        if 'warning' in thresholds:
            min_val, max_val = thresholds['warning']
            if min_val <= value < max_val:
                color = 'orange'
        
        if 'critical' in thresholds:
            min_val, max_val = thresholds['critical']
            if min_val <= value < max_val:
                color = 'red'
        
        # 스타일시트 적용
        label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _apply_progress_bar_color(self, progress_bar: QLabel, value: float, thresholds: Dict[str, tuple]) -> None:
        """
        임계값에 따라 프로그레스 바 색상을 적용한다.
        
        Args:
            progress_bar: 색상을 적용할 QProgressBar
            value: 현재 값
            thresholds: {'good': (min, max), 'warning': (min, max), 'critical': (min, max)}
        """
        color = '#4caf50'  # 기본 색상 (녹색)
        
        if 'good' in thresholds:
            min_val, max_val = thresholds['good']
            if min_val <= value < max_val:
                color = '#4caf50' # 녹색
        
        if 'warning' in thresholds:
            min_val, max_val = thresholds['warning']
            if min_val <= value < max_val:
                color = '#ff9800' # 주황
        
        if 'critical' in thresholds:
            min_val, max_val = thresholds['critical']
            if min_val <= value < max_val:
                color = '#f44336' # 빨강

        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)
