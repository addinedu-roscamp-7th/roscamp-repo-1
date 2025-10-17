"""'개요' 탭의 UI 로직"""
from typing import Any, Dict

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
        self.avg_time_label.setText(f"평균 처리 시간: {metrics.get('avg_processing_time', 0):.1f}s")
        self.throughput_label.setText(f"시간당 처리량: {metrics.get('hourly_throughput', 0)}건")
        self.success_rate_label.setText(f"성공률: {metrics.get('success_rate', 0):.1f}%")
        self.robot_util_label.setText(f"로봇 활용률: {metrics.get('robot_utilization', 0):.1f}%")
        self.system_load_label.setText(f"시스템 부하: {metrics.get('system_load', 0):.1f}%")

        robots = snapshot.get('robots', [])
        pickee_list = [r for r in robots if r.get('robot_type') == 'PICKEE']
        packee_list = [r for r in robots if r.get('robot_type') == 'PACKEE']
        robot_summary_text = f"""
Pickee: {len(pickee_list)}대
├─ WORKING: {sum(1 for r in pickee_list if r.get('status') == 'WORKING')}
├─ IDLE: {sum(1 for r in pickee_list if r.get('status') == 'IDLE')}
└─ ERROR: {sum(1 for r in pickee_list if r.get('status') == 'ERROR')}

Packee: {len(packee_list)}대
├─ WORKING: {sum(1 for r in packee_list if r.get('status') == 'WORKING')}
├─ IDLE: {sum(1 for r in packee_list if r.get('status') == 'IDLE')}
└─ OFFLINE: {sum(1 for r in packee_list if r.get('status') == 'OFFLINE')}
        """.strip()
        self.robot_summary_label.setText(robot_summary_text)

        orders = snapshot.get('orders', {})
        order_summary = orders.get('summary', {})
        order_summary_text = f"""
진행 중 주문: {order_summary.get('total_active', 0)}건
평균 진행률: {order_summary.get('avg_progress', 0):.0f}%

최근 1시간 완료: {metrics.get('hourly_throughput', 0)}건
실패: {order_summary.get('failed_count', 0)}건
        """.strip()
        self.order_summary_label.setText(order_summary_text)

    def add_alert(self, event_data: Dict[str, Any]):
        """최근 알림에 추가"""
        from datetime import datetime
        alert_types = {'robot_failure': '오류', 'robot_timeout_notification': '오류'}
        event_type = event_data.get('type', '')
        if event_type not in alert_types: return

        icon_map = {'오류': '🔴', '경고': '🟡', '정보': '🟢'}
        alert = {'timestamp': datetime.now(), 'icon': icon_map.get(alert_types[event_type], 'ℹ'), 'message': event_data.get('message', '')}
        self._recent_alerts.insert(0, alert)
        self._recent_alerts = self._recent_alerts[:5]

        alert_lines = [f"{a['icon']} [{a['timestamp'].strftime('%H:%M:%S')}] {a['message'][:60]}" for a in self._recent_alerts]
        self.alerts_label.setText('\n'.join(alert_lines) if alert_lines else '알림 없음')
