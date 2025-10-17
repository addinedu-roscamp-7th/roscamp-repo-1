"""'로봇 상태' 탭의 UI 로직"""
from typing import Any, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView

from ..ui_gen.tab_robot_status_ui import Ui_RobotStatusTab
from .base_tab import BaseTab


class RobotStatusTab(BaseTab, Ui_RobotStatusTab):
    """'로봇 상태' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._setup_table_columns()

    def _setup_table_columns(self):
        """테이블 컬럼 너비와 리사이즈 정책을 설정한다."""
        header = self.robot_table.horizontalHeader()
        
        # 각 컬럼의 너비를 설정 (픽셀 단위)
        self.robot_table.setColumnWidth(0, 80)   # Robot ID
        self.robot_table.setColumnWidth(1, 80)   # Type
        self.robot_table.setColumnWidth(2, 100)  # Status
        self.robot_table.setColumnWidth(3, 90)   # Battery(%)
        self.robot_table.setColumnWidth(4, 120)  # Location
        self.robot_table.setColumnWidth(5, 80)   # Cart
        self.robot_table.setColumnWidth(6, 90)   # Reserved
        self.robot_table.setColumnWidth(7, 90)   # Order ID
        self.robot_table.setColumnWidth(8, 100)  # Offline Time
        
        # 마지막 컬럼(Last Update)은 남은 공간을 모두 차지하도록 설정
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        
        # 나머지 컬럼들은 고정 크기로 설정
        for i in range(9):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)

    def update_data(self, robots: List[Dict[str, Any]]):
        """로봇 상태 데이터로 테이블을 업데이트한다."""
        from datetime import datetime
        
        self.robot_table.setRowCount(len(robots))
        for row, robot in enumerate(robots):
            # 컬럼 0: Robot ID
            self.robot_table.setItem(row, 0, QTableWidgetItem(str(robot.get('robot_id', ''))))
            
            # 컬럼 1: Type
            robot_type = robot.get('robot_type', '')
            type_text = getattr(robot_type, 'value', str(robot_type))
            self.robot_table.setItem(row, 1, QTableWidgetItem(type_text))
            
            # 컬럼 2: Status
            status = robot.get('status', '')
            status_item = QTableWidgetItem(str(status))
            if status == 'OFFLINE': 
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == 'ERROR': 
                status_item.setForeground(Qt.GlobalColor.magenta)
            self.robot_table.setItem(row, 2, status_item)
            
            # 컬럼 3: Battery(%)
            battery = robot.get('battery_level')
            battery_item = QTableWidgetItem(f'{battery:.1f}%' if battery is not None else '-')
            if battery is not None:
                if battery < 20: 
                    battery_item.setForeground(Qt.GlobalColor.red)
                elif battery < 50: 
                    battery_item.setForeground(Qt.GlobalColor.yellow)
                else:
                    battery_item.setForeground(Qt.GlobalColor.green)
            self.robot_table.setItem(row, 3, battery_item)
            
            # 컬럼 4: Location (NEW)
            location = robot.get('current_location', 'UNKNOWN')
            self.robot_table.setItem(row, 4, QTableWidgetItem(str(location)))
            
            # 컬럼 5: Cart (NEW)
            cart_status = robot.get('cart_status', 'UNKNOWN')
            cart_text = 'Full' if cart_status == 'FULL' else 'Empty' if cart_status == 'EMPTY' else str(cart_status)
            self.robot_table.setItem(row, 5, QTableWidgetItem(cart_text))
            
            # 컬럼 6: Reserved
            reserved_text = '예약됨' if robot.get('reserved', False) else '-'
            self.robot_table.setItem(row, 6, QTableWidgetItem(reserved_text))
            
            # 컬럼 7: Order ID
            order_id = robot.get('active_order_id')
            self.robot_table.setItem(row, 7, QTableWidgetItem(str(order_id) if order_id else '-'))
            
            # 컬럼 8: Offline Time (NEW)
            last_update = robot.get('last_update')
            offline_time_text = '-'
            if last_update and status == 'OFFLINE':
                try:
                    if isinstance(last_update, str):
                        # ISO 형식 문자열을 datetime으로 변환
                        if 'T' in last_update:
                            last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        else:
                            last_update_dt = datetime.fromisoformat(last_update)
                    else:
                        last_update_dt = last_update
                    
                    elapsed = datetime.now() - last_update_dt.replace(tzinfo=None)
                    total_seconds = int(elapsed.total_seconds())
                    if total_seconds >= 60:
                        minutes = total_seconds // 60
                        offline_time_text = f'{minutes}분'
                    else:
                        offline_time_text = f'{total_seconds}초'
                except Exception:
                    offline_time_text = '?'
            
            offline_item = QTableWidgetItem(offline_time_text)
            if status == 'OFFLINE' and offline_time_text not in ['-', '?']:
                offline_item.setForeground(Qt.GlobalColor.red)
            self.robot_table.setItem(row, 8, offline_item)
            
            # 컬럼 9: Last Update
            update_text = '-'
            if last_update:
                try:
                    if isinstance(last_update, str):
                        if 'T' in last_update:
                            dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        else:
                            dt = datetime.fromisoformat(last_update)
                        update_text = dt.strftime('%H:%M:%S')
                    elif hasattr(last_update, 'strftime'):
                        update_text = last_update.strftime('%H:%M:%S')
                except Exception:
                    update_text = str(last_update)[:8] if last_update else '-'
            
            self.robot_table.setItem(row, 9, QTableWidgetItem(update_text))
