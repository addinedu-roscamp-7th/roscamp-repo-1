"""'ROS2 서비스 모니터' 탭의 UI 로직"""
import json
from datetime import datetime
from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem

from ..ui_gen.tab_service_monitor_ui import Ui_ServiceMonitorTab
from .base_tab import BaseTab


class ServiceMonitorTab(BaseTab, Ui_ServiceMonitorTab):
    """'ROS2 서비스 모니터' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.call_id_to_row: Dict[str, int] = {}
        self.service_table.setColumnWidth(1, 300) # Service Name
        self.service_table.setColumnWidth(5, 250) # Request
        self.service_table.setColumnWidth(6, 250) # Response

    def handle_service_event(self, event_data: Dict[str, Any]):
        """서비스 호출 또는 응답 이벤트를 처리한다."""
        call_id = event_data.get('call_id')
        if not call_id:
            return

        if call_id not in self.call_id_to_row:
            # 새로운 서비스 호출
            self._add_new_service_call(event_data)
        else:
            # 기존 서비스 호출에 대한 응답
            self._update_service_response(event_data)

    def _add_new_service_call(self, event_data: Dict[str, Any]):
        """테이블에 새로운 서비스 호출 행을 추가한다."""
        row = self.service_table.rowCount()
        self.service_table.insertRow(row)
        self.call_id_to_row[event_data['call_id']] = row

        timestamp = event_data.get('timestamp', 0)
        dt_object = datetime.fromtimestamp(timestamp)
        time_str = dt_object.strftime('%H:%M:%S.%f')[:-3]

        service_name = event_data.get('service_name', '?')
        msg_str = json.dumps(event_data.get('msg', {}), ensure_ascii=False)

        self.service_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.service_table.setItem(row, 1, QTableWidgetItem(service_name))
        self.service_table.setItem(row, 2, QTableWidgetItem("Request ->"))
        self.service_table.setItem(row, 3, QTableWidgetItem("Pending..."))
        self.service_table.setItem(row, 5, QTableWidgetItem(msg_str))

        self.service_table.scrollToBottom()

    def _update_service_response(self, event_data: Dict[str, Any]):
        """기존 행을 서비스 응답 내용으로 업데이트한다."""
        row = self.call_id_to_row[event_data['call_id']]

        success = event_data.get('success', False)
        duration_ms = event_data.get('duration_ms', 0.0)
        msg_str = json.dumps(event_data.get('msg', {}), ensure_ascii=False)

        status_item = QTableWidgetItem("Success" if success else "Failed")
        status_item.setForeground(Qt.GlobalColor.blue if success else Qt.GlobalColor.red)

        self.service_table.setItem(row, 2, QTableWidgetItem("<- Response"))
        self.service_table.setItem(row, 3, status_item)
        self.service_table.setItem(row, 4, QTableWidgetItem(f"{duration_ms:.2f}"))
        self.service_table.setItem(row, 6, QTableWidgetItem(msg_str))

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않고 이벤트만 받습니다."""
        pass
