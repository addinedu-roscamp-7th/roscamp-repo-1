"""'로봇 상태' 탭의 UI 로직"""
from typing import Any, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem

from ..ui_gen.tab_robot_status_ui import Ui_RobotStatusTab
from .base_tab import BaseTab


class RobotStatusTab(BaseTab, Ui_RobotStatusTab):
    """'로봇 상태' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def update_data(self, robots: List[Dict[str, Any]]):
        """로봇 상태 데이터로 테이블을 업데이트한다."""
        self.robot_table.setRowCount(len(robots))
        for row, robot in enumerate(robots):
            self.robot_table.setItem(row, 0, QTableWidgetItem(str(robot.get('robot_id', ''))))
            robot_type = robot.get('robot_type', '')
            type_text = getattr(robot_type, 'value', str(robot_type))
            self.robot_table.setItem(row, 1, QTableWidgetItem(type_text))
            status = robot.get('status', '')
            status_item = QTableWidgetItem(str(status))
            if status == 'OFFLINE': status_item.setForeground(Qt.GlobalColor.red)
            elif status == 'ERROR': status_item.setForeground(Qt.GlobalColor.magenta)
            self.robot_table.setItem(row, 2, status_item)
            battery = robot.get('battery_level')
            battery_item = QTableWidgetItem(f'{battery:.1f}' if battery is not None else '-')
            if battery is not None:
                if battery < 20: battery_item.setForeground(Qt.GlobalColor.red)
                elif battery < 50: battery_item.setForeground(Qt.GlobalColor.yellow)
            self.robot_table.setItem(row, 3, battery_item)
            reserved_text = '예약됨' if robot.get('reserved', False) else '-'
            self.robot_table.setItem(row, 4, QTableWidgetItem(reserved_text))
            order_id = robot.get('active_order_id')
            self.robot_table.setItem(row, 5, QTableWidgetItem(str(order_id) if order_id else '-'))
            last_update = robot.get('last_update')
            update_text = last_update.strftime('%H:%M:%S') if hasattr(last_update, 'strftime') else '-'
            self.robot_table.setItem(row, 6, QTableWidgetItem(update_text))
