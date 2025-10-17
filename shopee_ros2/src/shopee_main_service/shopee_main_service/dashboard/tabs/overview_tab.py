"""'개요' 탭의 UI 로직"""
from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from ..ui_gen.tab_overview_ui import Ui_OverviewTab
from .base_tab import BaseTab


class OverviewTab(BaseTab, Ui_OverviewTab):
    """'개요' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._recent_alerts = []

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 개요 탭을 업데이트한다."""
        metrics = snapshot.get('metrics', {})
        
        # 평균 처리 시간 (목표 2초 이하: 녹색, 2~5초: 노랑, 5초 이상: 빨강)
        avg_time = metrics.get('avg_processing_time', 0)
        self.avg_processing_time_label.setText(f"평균 처리 시간: {avg_time:.1f}초")
        self._apply_color_threshold(self.avg_processing_time_label, avg_time, {
            'good': (0, 2.0), 'warning': (2.0, 5.0), 'critical': (5.0, float('inf'))
        })
        
        # 시간당 처리량 (색상 없음)
        throughput = metrics.get('hourly_throughput', 0)
        self.hourly_throughput_label.setText(f"시간당 처리량: {throughput}건")
        self.hourly_throughput_label.setStyleSheet("")  # 기본 색상
        
        # 성공률 (95% 이상: 녹색, 90~95%: 노랑, 90% 미만: 빨강)
        success_rate = metrics.get('success_rate', 0)
        self.success_rate_label.setText(f"성공률: {success_rate:.1f}%")
        self._apply_color_threshold(self.success_rate_label, success_rate, {
            'good': (95, 100), 'warning': (90, 95), 'critical': (0, 90)
        })
        
        # 로봇 활용률 (60% 이상: 녹색, 30~60%: 노랑, 30% 미만: 빨강)
        robot_util = metrics.get('robot_utilization', 0)
        self.robot_utilization_label.setText(f"로봇 가동률: {robot_util:.1f}%")
        self._apply_color_threshold(self.robot_utilization_label, robot_util, {
            'good': (60, 100), 'warning': (30, 60), 'critical': (0, 30)
        })
        
        # 시스템 부하 (70% 미만: 녹색, 70~90%: 노랑, 90% 이상: 빨강)
        system_load = metrics.get('system_load', 0)
        self.system_load_label.setText(f"시스템 부하: {system_load:.1f}%")
        self._apply_color_threshold(self.system_load_label, system_load, {
            'good': (0, 70), 'warning': (70, 90), 'critical': (90, 100)
        })
        
        # 활성 주문 수
        active_orders = metrics.get('active_orders', 0)
        self.active_orders_label.setText(f"활성 주문: {active_orders}건")

        # 알림 영역에 시스템 상태 정보 표시
        robots = snapshot.get('robots', [])
        pickee_list = [r for r in robots if r.get('robot_type') == 'PICKEE']
        packee_list = [r for r in robots if r.get('robot_type') == 'PACKEE']
        
        # 활성 로봇만 필터링 (OFFLINE 제외)
        active_pickee = [r for r in pickee_list if r.get('status') != 'OFFLINE']
        active_packee = [r for r in packee_list if r.get('status') != 'OFFLINE']
        
        # OFFLINE 로봇 수 계산
        offline_pickee = sum(1 for r in pickee_list if r.get('status') == 'OFFLINE')
        offline_packee = sum(1 for r in packee_list if r.get('status') == 'OFFLINE')
        
        orders = snapshot.get('orders', {})
        order_summary = orders.get('summary', {})
        
        status_text = f"""=== 시스템 상태 개요 ===

📊 성능 메트릭스:
• 평균 처리 시간: {avg_time:.1f}초
• 시간당 처리량: {throughput}건
• 성공률: {success_rate:.1f}%
• 로봇 가동률: {robot_util:.1f}%
• 시스템 부하: {system_load:.1f}%

🤖 로봇 현황:
Pickee: {len(active_pickee)}/{len(pickee_list)}대 활성
├─ WORKING: {sum(1 for r in active_pickee if r.get('status') == 'WORKING')}대
├─ IDLE: {sum(1 for r in active_pickee if r.get('status') == 'IDLE')}대
├─ ERROR: {sum(1 for r in active_pickee if r.get('status') == 'ERROR')}대
└─ OFFLINE: {offline_pickee}대 (비활성)

Packee: {len(active_packee)}/{len(packee_list)}대 활성
├─ WORKING: {sum(1 for r in active_packee if r.get('status') == 'WORKING')}대
├─ IDLE: {sum(1 for r in active_packee if r.get('status') == 'IDLE')}대
├─ ERROR: {sum(1 for r in active_packee if r.get('status') == 'ERROR')}대
└─ OFFLINE: {offline_packee}대 (비활성)

📦 주문 현황:
• 진행 중 주문: {order_summary.get('total_active', 0)}건
• 평균 진행률: {order_summary.get('avg_progress', 0):.0f}%
• 최근 1시간 완료: {throughput}건
• 실패: {order_summary.get('failed_count', 0)}건

최근 알림: {len(self._recent_alerts)}건
        """.strip()
        
        self.alerts_text.setPlainText(status_text)

    def add_alert(self, event_data: Dict[str, Any]):
        """최근 알림에 추가"""
        from datetime import datetime
        alert_types = {'robot_failure': '오류', 'robot_timeout_notification': '오류'}
        event_type = event_data.get('type', '')
        if event_type not in alert_types: 
            return

        icon_map = {'오류': '🔴', '경고': '🟡', '정보': '🟢'}
        alert = {
            'timestamp': datetime.now(), 
            'icon': icon_map.get(alert_types[event_type], 'ℹ'), 
            'message': event_data.get('message', '')
        }
        self._recent_alerts.insert(0, alert)
        self._recent_alerts = self._recent_alerts[:10]  # 최근 10개 알림 유지

        # 알림을 텍스트 영역에 추가 (기존 내용 유지하면서)
        current_text = self.alerts_text.toPlainText()
        alert_text = f"🚨 [{alert['timestamp'].strftime('%H:%M:%S')}] {alert['message']}"
        
        # 새 알림을 맨 위에 추가
        if current_text:
            new_text = alert_text + "\n" + current_text
        else:
            new_text = alert_text
            
        self.alerts_text.setPlainText(new_text)

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
