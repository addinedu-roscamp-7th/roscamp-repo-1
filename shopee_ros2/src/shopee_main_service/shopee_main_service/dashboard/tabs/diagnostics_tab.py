'''시스템 진단' 탭의 UI 로직'''
from typing import Any, Dict

from ..ui_gen.tab_diagnostics_ui import Ui_SystemDiagnosticsTab
from .base_tab import BaseTab


class SystemDiagnosticsTab(BaseTab, Ui_SystemDiagnosticsTab):
    """'시스템 진단' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def update_data(self, snapshot: Dict[str, Any]):
        """스냅샷 데이터로 진단 탭을 업데이트한다."""
        metrics = snapshot.get('metrics', {})
        failed_by_reason = metrics.get('failed_orders_by_reason', {})
        error_lines = ['최근 실패 주문 (30분 이내): ' + (', '.join(f'{r}: {c}건' for r, c in failed_by_reason.items()) if failed_by_reason else '없음')]
        self.error_label.setText('\n'.join(error_lines))

        network = metrics.get('network', {})
        network_lines = [f"App 세션: {network.get('app_sessions', 0)} / {network.get('app_sessions_max', 200)}"]
        self.network_label.setText('\n'.join(network_lines))

