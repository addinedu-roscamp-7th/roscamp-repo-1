"""'주문 관리' 탭의 UI 로직"""
from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QProgressBar, QWidget, QHBoxLayout, QHeaderView

from ..ui_gen.tab_order_management_ui import Ui_OrderManagementTab
from .base_tab import BaseTab


class OrderManagementTab(BaseTab, Ui_OrderManagementTab):
    """'주문 관리' 탭의 UI 및 로직"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._setup_table_columns()

    def _setup_table_columns(self):
        """테이블 컬럼 너비와 리사이즈 정책을 설정한다."""
        header = self.order_table.horizontalHeader()
        
        # 각 컬럼의 너비를 설정 (픽셀 단위)
        self.order_table.setColumnWidth(0, 80)   # Order ID
        self.order_table.setColumnWidth(1, 100)  # Customer
        self.order_table.setColumnWidth(2, 100)  # Status
        self.order_table.setColumnWidth(3, 60)   # Items
        self.order_table.setColumnWidth(4, 100)  # Amount
        self.order_table.setColumnWidth(5, 120)  # Progress(%)
        self.order_table.setColumnWidth(6, 80)   # Started
        self.order_table.setColumnWidth(7, 80)   # Elapsed
        self.order_table.setColumnWidth(8, 70)   # Pickee
        
        # 마지막 컬럼(Packee)은 남은 공간을 모두 차지하도록 설정
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        
        # 나머지 컬럼들은 고정 크기로 설정
        for i in range(9):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)

    def update_data(self, orders_snapshot: Dict[str, Any]):
        """주문 데이터로 테이블을 업데이트한다."""
        orders = orders_snapshot.get('orders', [])
        self.order_table.setRowCount(len(orders))
        
        for row, order in enumerate(orders):
            # 컬럼 0: Order ID
            self.order_table.setItem(row, 0, QTableWidgetItem(str(order.get('order_id', ''))))
            
            # 컬럼 1: Customer
            self.order_table.setItem(row, 1, QTableWidgetItem(str(order.get('customer_id', ''))))
            
            # 컬럼 2: Status
            status = str(order.get('status', ''))
            status_item = QTableWidgetItem(status)
            # 상태별 색상 적용
            if 'FAIL' in status:
                status_item.setForeground(Qt.GlobalColor.red)
            elif status in ['PACKED', 'DELIVERED']:
                status_item.setForeground(Qt.GlobalColor.green)
            elif status in ['PICKING', 'PACKING', 'MOVING']:
                status_item.setForeground(Qt.GlobalColor.blue)
            self.order_table.setItem(row, 2, status_item)
            
            # 컬럼 3: Items
            self.order_table.setItem(row, 3, QTableWidgetItem(str(order.get('total_items', 0))))
            
            # 컬럼 4: Amount
            total_price = order.get('total_price')
            amount_text = f'₩{int(total_price):,}' if total_price is not None else '-'
            self.order_table.setItem(row, 4, QTableWidgetItem(amount_text))
            
            # 컬럼 5: Progress (프로그레스 바로 표시)
            progress = order.get('progress', 0)
            progress_widget = self._create_progress_widget(progress)
            self.order_table.setCellWidget(row, 5, progress_widget)
            
            # 컬럼 6: Started
            started = order.get('started_at', '')
            started_text = started.split('T')[1].split('.')[0] if 'T' in started else '-'
            self.order_table.setItem(row, 6, QTableWidgetItem(started_text))
            
            # 컬럼 7: Elapsed
            elapsed_sec = order.get('elapsed_seconds')
            elapsed_text = f'{int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s' if elapsed_sec is not None else '-'
            elapsed_item = QTableWidgetItem(elapsed_text)
            # 30초 이상 경과 시 경고 색상
            if elapsed_sec and elapsed_sec > 30:
                elapsed_item.setForeground(Qt.GlobalColor.red)
            elif elapsed_sec and elapsed_sec > 20:
                elapsed_item.setForeground(Qt.GlobalColor.darkYellow)
            self.order_table.setItem(row, 7, elapsed_item)
            
            # 컬럼 8: Pickee
            pickee_id = order.get('pickee_robot_id', '-')
            self.order_table.setItem(row, 8, QTableWidgetItem(str(pickee_id)))
            
            # 컬럼 9: Packee
            packee_id = order.get('packee_robot_id', '-')
            self.order_table.setItem(row, 9, QTableWidgetItem(str(packee_id)))

    def _create_progress_widget(self, progress: int) -> QWidget:
        """진행률을 표시하는 프로그레스 바 위젯을 생성한다."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(int(progress))
        progress_bar.setTextVisible(True)
        progress_bar.setFormat(f'{progress}%')
        
        # 프로그레스 바 크기 설정
        progress_bar.setMinimumHeight(20)
        progress_bar.setMaximumHeight(25)
        
        # 진행률에 따른 색상 설정 (더 명확한 스타일)
        if progress >= 80:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;  /* 녹색 */
                    border-radius: 2px;
                }
            """)
        elif progress >= 50:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #FF9800;  /* 주황색 */
                    border-radius: 2px;
                }
            """)
        else:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;  /* 파란색 */
                    border-radius: 2px;
                }
            """)
        
        layout.addWidget(progress_bar)
        return widget
