"""
대시보드 UI 패널 컴포넌트

PyQt6 기반 테이블 및 로그 위젯을 제공한다.
"""

from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class RobotPanel(QWidget):
    """
    로봇 상태를 표시하는 패널

    Pickee/Packee 로봇의 상태, 배터리, 예약 정보, 활성 주문 등을 테이블로 표시한다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 그룹박스
        group_box = QGroupBox('로봇 상태')
        group_layout = QVBoxLayout()

        # 테이블 생성
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            'Robot ID',
            'Type',
            'Status',
            'Battery(%)',
            'Reserved',
            'Order ID',
            'Last Update',
        ])

        # 테이블 설정
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        group_layout.addWidget(self.table)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

    def update_data(self, robots: List[Dict[str, Any]]):
        """
        로봇 상태 데이터를 업데이트한다.

        Args:
            robots: 로봇 상태 딕셔너리 리스트
        """
        self.table.setRowCount(len(robots))

        for row, robot in enumerate(robots):
            # Robot ID
            self.table.setItem(row, 0, QTableWidgetItem(str(robot.get('robot_id', ''))))

            # Type
            robot_type = robot.get('robot_type', '')
            if isinstance(robot_type, str):
                type_text = robot_type
            else:
                type_text = getattr(robot_type, 'value', str(robot_type))
            self.table.setItem(row, 1, QTableWidgetItem(type_text))

            # Status
            status = robot.get('status', '')
            status_item = QTableWidgetItem(str(status))
            if status == 'OFFLINE':
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == 'ERROR':
                status_item.setForeground(Qt.GlobalColor.magenta)
            self.table.setItem(row, 2, status_item)

            # Battery
            battery = robot.get('battery_level')
            if battery is not None:
                battery_item = QTableWidgetItem(f'{battery:.1f}')
                if battery < 20:
                    battery_item.setForeground(Qt.GlobalColor.red)
                elif battery < 50:
                    battery_item.setForeground(Qt.GlobalColor.yellow)
            else:
                battery_item = QTableWidgetItem('-')
            self.table.setItem(row, 3, battery_item)

            # Reserved
            reserved = robot.get('reserved', False)
            reserved_text = '예약됨' if reserved else '-'
            self.table.setItem(row, 4, QTableWidgetItem(reserved_text))

            # Order ID
            order_id = robot.get('active_order_id')
            order_text = str(order_id) if order_id else '-'
            self.table.setItem(row, 5, QTableWidgetItem(order_text))

            # Last Update
            last_update = robot.get('last_update')
            if last_update:
                if isinstance(last_update, str):
                    update_text = last_update.split('.')[0]  # 마이크로초 제거
                else:
                    update_text = last_update.strftime('%H:%M:%S')
            else:
                update_text = '-'
            self.table.setItem(row, 6, QTableWidgetItem(update_text))


class OrderPanel(QWidget):
    """
    진행 중 주문을 표시하는 패널

    현재 처리 중인 주문의 상태, 진행률, 할당 로봇, 경과 시간 등을 표시한다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 그룹박스
        group_box = QGroupBox('진행 중 주문')
        group_layout = QVBoxLayout()

        # 테이블 생성
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            'Order ID',
            'Customer',
            'Status',
            'Items',
            'Amount',
            'Progress(%)',
            'Started',
            'Elapsed',
            'Pickee',
            'Packee',
        ])

        # 테이블 설정
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        group_layout.addWidget(self.table)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

    def update_data(self, orders_snapshot: Dict[str, Any]):
        """
        주문 데이터를 업데이트한다.

        Args:
            orders_snapshot: OrderService.get_active_orders_snapshot() 결과
        """
        orders = orders_snapshot.get('orders', [])
        self.table.setRowCount(len(orders))

        for row, order in enumerate(orders):
            # Order ID
            self.table.setItem(row, 0, QTableWidgetItem(str(order.get('order_id', ''))))

            # Customer
            customer_id = order.get('customer_id', '')
            self.table.setItem(row, 1, QTableWidgetItem(str(customer_id)))

            # Status
            status = order.get('status', '')
            self.table.setItem(row, 2, QTableWidgetItem(str(status)))

            # Items
            total_items = order.get('total_items', 0)
            self.table.setItem(row, 3, QTableWidgetItem(str(total_items)))

            # Amount
            total_price = order.get('total_price')
            if total_price is not None:
                try:
                    amount_value = int(total_price)
                    amount_text = f'₩{amount_value:,}'
                except (TypeError, ValueError):
                    amount_text = str(total_price)
            else:
                amount_text = '-'
            self.table.setItem(row, 4, QTableWidgetItem(amount_text))

            # Progress
            progress = order.get('progress', 0)
            self.table.setItem(row, 5, QTableWidgetItem(f'{progress}'))

            # Started
            started = order.get('started_at', '')
            if started:
                started_text = started.split('T')[1].split('.')[0] if 'T' in started else started
            else:
                started_text = '-'
            self.table.setItem(row, 6, QTableWidgetItem(started_text))

            # Elapsed
            elapsed_sec = order.get('elapsed_seconds')
            if elapsed_sec is not None:
                elapsed_min = int(elapsed_sec // 60)
                elapsed_s = int(elapsed_sec % 60)
                elapsed_text = f'{elapsed_min}m {elapsed_s}s'
            else:
                elapsed_text = '-'
            self.table.setItem(row, 7, QTableWidgetItem(elapsed_text))

            # Pickee
            pickee_id = order.get('pickee_robot_id')
            pickee_text = str(pickee_id) if pickee_id else '-'
            self.table.setItem(row, 8, QTableWidgetItem(pickee_text))

            # Packee
            packee_id = order.get('packee_robot_id')
            packee_text = str(packee_id) if packee_id else '-'
            self.table.setItem(row, 9, QTableWidgetItem(packee_text))


class EventLogPanel(QWidget):
    """
    이벤트 로그를 표시하는 패널

    EventBus에서 발생한 이벤트를 시간순으로 표시한다.
    """

    MAX_LOG_ENTRIES = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_entries: List[str] = []
        self._init_ui()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 그룹박스
        group_box = QGroupBox('이벤트 로그')
        group_layout = QVBoxLayout()

        # 텍스트 에디터 (읽기 전용)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumHeight(200)

        group_layout.addWidget(self.text_edit)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

    def add_event(self, event_data: Dict[str, Any]):
        """
        이벤트를 로그에 추가한다.

        Args:
            event_data: 이벤트 페이로드 딕셔너리
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        event_type = event_data.get('type', 'unknown')
        message = event_data.get('message', '')

        # 추가 정보 포맷팅
        data = event_data.get('data', {})
        details = []
        if 'order_id' in data:
            details.append(f"Order={data['order_id']}")
        if 'robot_id' in data:
            details.append(f"Robot={data['robot_id']}")

        detail_str = ', '.join(details) if details else ''
        log_line = f'[{timestamp}] {event_type}: {message}'
        if detail_str:
            log_line += f' ({detail_str})'

        # 로그 추가
        self._log_entries.append(log_line)

        # 최대 개수 제한
        if len(self._log_entries) > self.MAX_LOG_ENTRIES:
            self._log_entries = self._log_entries[-self.MAX_LOG_ENTRIES:]

        # 화면 갱신
        self.text_edit.setPlainText('\n'.join(self._log_entries))

        # 자동 스크롤
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_logs(self):
        """로그를 모두 지운다."""
        self._log_entries.clear()
        self.text_edit.clear()
