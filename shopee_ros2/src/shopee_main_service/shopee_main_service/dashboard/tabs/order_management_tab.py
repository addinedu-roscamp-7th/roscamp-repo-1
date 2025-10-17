"""'주문 관리' 탭의 UI 로직"""
from typing import Any, Dict

from PyQt6.QtWidgets import QTableWidgetItem

from ..ui_gen.tab_order_management_ui import Ui_OrderManagementTab
from .base_tab import BaseTab


class OrderManagementTab(BaseTab, Ui_OrderManagementTab):
    """'주문 관리' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def update_data(self, orders_snapshot: Dict[str, Any]):
        """주문 데이터로 테이블을 업데이트한다."""
        orders = orders_snapshot.get('orders', [])
        self.order_table.setRowCount(len(orders))
        for row, order in enumerate(orders):
            self.order_table.setItem(row, 0, QTableWidgetItem(str(order.get('order_id', ''))))
            self.order_table.setItem(row, 1, QTableWidgetItem(str(order.get('customer_id', ''))))
            self.order_table.setItem(row, 2, QTableWidgetItem(str(order.get('status', ''))))
            self.order_table.setItem(row, 3, QTableWidgetItem(str(order.get('total_items', 0))))
            total_price = order.get('total_price')
            amount_text = f'₩{int(total_price):,}' if total_price is not None else '-'
            self.order_table.setItem(row, 4, QTableWidgetItem(amount_text))
            self.order_table.setItem(row, 5, QTableWidgetItem(f"{order.get('progress', 0)}"))
            started = order.get('started_at', '')
            started_text = started.split('T')[1].split('.')[0] if 'T' in started else '-'
            self.order_table.setItem(row, 6, QTableWidgetItem(started_text))
            elapsed_sec = order.get('elapsed_seconds')
            elapsed_text = f'{int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s' if elapsed_sec is not None else '-'
            self.order_table.setItem(row, 7, QTableWidgetItem(elapsed_text))
            self.order_table.setItem(row, 8, QTableWidgetItem(str(order.get('pickee_robot_id', '-'))))
            self.order_table.setItem(row, 9, QTableWidgetItem(str(order.get('packee_robot_id', '-'))))
