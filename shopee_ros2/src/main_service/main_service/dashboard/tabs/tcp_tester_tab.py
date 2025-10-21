"""
'TCP 테스터' 탭의 UI 로직

App↔Main Service 간 TCP 통신을 테스트하기 위한 도구입니다.
실제 App 없이 Main Service의 TCP 메시지 처리 기능을 테스트할 수 있습니다.

지원 메시지:
- 요청-응답 API (14개)
- 이벤트 알림 (7개)
"""
import json
import logging
import socket
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTreeWidgetItem,
    QTextEdit,
)

from ..ui_gen.tab_tcp_tester_ui import Ui_TcpTesterTab
from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class TcpClient(QObject):
    """TCP 클라이언트 (비동기 수신용)"""

    message_received = pyqtSignal(dict)
    connection_lost = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.receive_thread: Optional[threading.Thread] = None

    def connect(self, host: str, port: int) -> bool:
        """서버에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((host, port))
            self.running = True

            # 수신 스레드 시작
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()

            return True
        except Exception as e:
            logger.error(f'TCP 연결 실패: {e}')
            self.socket = None
            return False

    def disconnect(self):
        """서버 연결 해제"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
            self.receive_thread = None

    def send(self, message: dict) -> bool:
        """메시지 전송"""
        if not self.socket:
            return False

        try:
            json_str = json.dumps(message, ensure_ascii=False) + '\n'
            self.socket.sendall(json_str.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f'TCP 전송 실패: {e}')
            return False

    def _receive_loop(self):
        """수신 루프 (별도 스레드)"""
        buffer = b''

        while self.running and self.socket:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    # 서버가 연결을 끊음
                    self.connection_lost.emit()
                    break

                buffer += chunk

                # JSON 객체 파싱 시도
                try:
                    message = json.loads(buffer.decode('utf-8'))
                    self.message_received.emit(message)
                    buffer = b''
                except json.JSONDecodeError:
                    # 아직 완전한 JSON이 아님, 더 수신 대기
                    if len(buffer) > 1024 * 1024:  # 1MB 초과 시 버퍼 초기화
                        logger.warning('버퍼 오버플로우, 초기화')
                        buffer = b''
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f'TCP 수신 오류: {e}')
                self.connection_lost.emit()
                break


class TcpTesterTab(BaseTab, Ui_TcpTesterTab):
    """
    'TCP 테스터' 탭의 UI 및 로직

    App→Main 방향의 TCP 메시지를 전송하여 Main Service의 TCP 처리 기능을 테스트합니다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self._tcp_client = TcpClient()
        self._current_message_type = None
        self._field_widgets = {}
        self._message_definitions = self._build_message_definitions()

        self._init_logic()

    def _init_logic(self):
        """로직 및 시그널-슬롯 연결 초기화"""
        self._populate_message_tree()

        # UI 시그널 연결
        self.message_tree.itemClicked.connect(self._on_message_selected)
        self.connect_button.clicked.connect(self._connect_to_server)
        self.disconnect_button.clicked.connect(self._disconnect_from_server)
        self.send_button.clicked.connect(self._send_message)
        self.reset_button.clicked.connect(self._reset_fields)
        self.clear_log_button.clicked.connect(self.result_log.clear)

        # TCP 클라이언트 시그널 연결
        self._tcp_client.message_received.connect(self._on_message_received)
        self._tcp_client.connection_lost.connect(self._on_connection_lost)

    def _build_message_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        지원하는 모든 TCP 메시지 정의

        Returns:
            Dict: 메시지 타입을 키로, 메시지 정의를 값으로 하는 딕셔너리
        """
        return {
            # 요청-응답 API
            'user_login': {
                'category': '요청-응답 API',
                'name': '사용자 로그인',
                'description': '사용자 로그인 요청',
                'fields': {
                    'user_id': {'type': 'str', 'default': 'customer001', 'description': '사용자 ID'},
                    'password': {'type': 'str', 'default': 'hunter2', 'description': '비밀번호'},
                },
            },
            'total_product': {
                'category': '요청-응답 API',
                'name': '전체 상품 요청',
                'description': '전체 상품 목록 요청',
                'fields': {
                    'user_id': {'type': 'str', 'default': 'customer001', 'description': '사용자 ID'},
                },
            },
            'product_search': {
                'category': '요청-응답 API',
                'name': '상품 검색',
                'description': '상품 검색 요청',
                'fields': {
                    'user_id': {'type': 'str', 'default': 'customer001', 'description': '사용자 ID'},
                    'query': {'type': 'str', 'default': '사과', 'description': '검색어'},
                    'filter': {'type': 'json', 'default': {}, 'description': '필터 (JSON)'},
                },
            },
            'order_create': {
                'category': '요청-응답 API',
                'name': '주문 생성',
                'description': '새 주문 생성 요청',
                'fields': {
                    'user_id': {'type': 'str', 'default': 'customer001', 'description': '사용자 ID'},
                    'cart_items': {'type': 'json', 'default': [], 'description': '장바구니 아이템 (JSON)'},
                    'payment_method': {'type': 'str', 'default': 'card', 'description': '결제 방법'},
                    'total_amount': {'type': 'int', 'default': 16200, 'description': '총 금액'},
                },
            },
            'product_selection': {
                'category': '요청-응답 API',
                'name': '상품 선택 (BBox)',
                'description': 'BBox 번호로 상품 선택',
                'fields': {
                    'order_id': {'type': 'int', 'default': 15, 'description': '주문 ID'},
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'bbox_number': {'type': 'int', 'default': 2, 'description': 'BBox 번호'},
                    'product_id': {'type': 'int', 'default': 45, 'description': '상품 ID'},
                },
            },
            'product_selection_by_text': {
                'category': '요청-응답 API',
                'name': '상품 선택 (텍스트)',
                'description': '음성 텍스트로 상품 선택',
                'fields': {
                    'order_id': {'type': 'int', 'default': 15, 'description': '주문 ID'},
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'speech': {'type': 'str', 'default': '사과 두 개', 'description': '음성 텍스트'},
                },
            },
            'shopping_end': {
                'category': '요청-응답 API',
                'name': '쇼핑 종료',
                'description': '쇼핑 종료 요청',
                'fields': {
                    'user_id': {'type': 'str', 'default': 'customer001', 'description': '사용자 ID'},
                    'order_id': {'type': 'int', 'default': 15, 'description': '주문 ID'},
                },
            },
            'video_stream_start': {
                'category': '요청-응답 API',
                'name': '영상 스트림 시작',
                'description': '영상 스트리밍 시작 요청',
                'fields': {
                    'user_type': {'type': 'str', 'default': 'admin', 'description': '사용자 타입'},
                    'user_id': {'type': 'str', 'default': 'admin01', 'description': '사용자 ID'},
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                },
            },
            'video_stream_stop': {
                'category': '요청-응답 API',
                'name': '영상 스트림 중지',
                'description': '영상 스트리밍 중지 요청',
                'fields': {
                    'user_type': {'type': 'str', 'default': 'admin', 'description': '사용자 타입'},
                    'user_id': {'type': 'str', 'default': 'admin01', 'description': '사용자 ID'},
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                },
            },
            'inventory_search': {
                'category': '요청-응답 API',
                'name': '재고 조회',
                'description': '재고 검색 요청',
                'fields': {
                    'product_id': {'type': 'int?', 'default': None, 'description': '상품 ID (선택)'},
                    'name': {'type': 'str?', 'default': '사과', 'description': '상품명 (선택)'},
                    'category': {'type': 'str?', 'default': 'fruit', 'description': '카테고리 (선택)'},
                },
            },
            'inventory_create': {
                'category': '요청-응답 API',
                'name': '재고 추가',
                'description': '재고 추가 요청',
                'fields': {
                    'product_id': {'type': 'int', 'default': 278, 'description': '상품 ID'},
                    'barcode': {'type': 'str', 'default': '8800000001055', 'description': '바코드'},
                    'name': {'type': 'str', 'default': '그릭요거트', 'description': '상품명'},
                    'quantity': {'type': 'int', 'default': 12, 'description': '수량'},
                    'price': {'type': 'int', 'default': 4900, 'description': '가격'},
                    'section_id': {'type': 'int', 'default': 205, 'description': '섹션 ID'},
                    'category': {'type': 'str', 'default': 'dairy', 'description': '카테고리'},
                    'allergy_info_id': {'type': 'int', 'default': 18, 'description': '알러지 정보 ID'},
                    'is_vegan_friendly': {'type': 'bool', 'default': False, 'description': '비건 친화'},
                },
            },
            'inventory_update': {
                'category': '요청-응답 API',
                'name': '재고 수정',
                'description': '재고 수정 요청',
                'fields': {
                    'product_id': {'type': 'int', 'default': 20, 'description': '상품 ID'},
                    'barcode': {'type': 'str', 'default': '8800000000012', 'description': '바코드'},
                    'name': {'type': 'str', 'default': '청사과', 'description': '상품명'},
                    'quantity': {'type': 'int', 'default': 30, 'description': '수량'},
                    'price': {'type': 'int', 'default': 3200, 'description': '가격'},
                    'section_id': {'type': 'int', 'default': 101, 'description': '섹션 ID'},
                    'category': {'type': 'str', 'default': 'fruit', 'description': '카테고리'},
                    'allergy_info_id': {'type': 'int', 'default': 12, 'description': '알러지 정보 ID'},
                    'is_vegan_friendly': {'type': 'bool', 'default': True, 'description': '비건 친화'},
                },
            },
            'inventory_delete': {
                'category': '요청-응답 API',
                'name': '재고 삭제',
                'description': '재고 삭제 요청',
                'fields': {
                    'product_id': {'type': 'int', 'default': 20, 'description': '상품 ID'},
                },
            },
            'robot_history_search': {
                'category': '요청-응답 API',
                'name': '작업 이력 조회',
                'description': '로봇 작업 이력 조회',
                'fields': {
                    'robot_id': {'type': 'int?', 'default': 1, 'description': '로봇 ID (선택)'},
                    'is_complete': {'type': 'bool?', 'default': None, 'description': '완료 여부 (선택)'},
                },
            },
        }

    def _populate_message_tree(self):
        """메시지 트리를 채웁니다"""
        self.message_tree.clear()
        categories: Dict[str, QTreeWidgetItem] = {}

        for msg_type, msg_def in self._message_definitions.items():
            category = msg_def['category']
            name = msg_def['name']

            if category not in categories:
                category_item = QTreeWidgetItem(self.message_tree, [category])
                category_item.setExpanded(True)
                categories[category] = category_item

            message_item = QTreeWidgetItem(categories[category], [name])
            message_item.setData(0, Qt.ItemDataRole.UserRole, msg_type)

    def _on_message_selected(self, item: QTreeWidgetItem, column: int):
        """메시지 선택 시 필드 입력 폼 생성"""
        msg_type = item.data(0, Qt.ItemDataRole.UserRole)
        if not msg_type:
            return

        self._current_message_type = msg_type
        msg_def = self._message_definitions.get(msg_type)
        if not msg_def:
            return

        desc = msg_def.get('description', '')
        self.message_desc_label.setText(f'<b>{msg_def["name"]}</b><br>{desc}<br><i>type: {msg_type}</i>')

        # 기존 폼 위젯 제거
        while self.fields_form_layout.rowCount() > 0:
            self.fields_form_layout.removeRow(0)
        self._field_widgets = {}

        fields = msg_def.get('fields', {})

        if not fields:
            label = QLabel('이 메시지는 필드가 없습니다.')
            label.setStyleSheet('color: #999; font-style: italic;')
            self.fields_form_layout.addRow(label)
            self.send_button.setEnabled(self._tcp_client.socket is not None)
            self.reset_button.setEnabled(False)
            return

        for field_name, field_info in fields.items():
            field_type = field_info.get('type', 'str')
            default_value = field_info.get('default', '')
            description = field_info.get('description', '')

            # 필드 타입에 따라 위젯 생성
            if field_type == 'int' or field_type == 'int?':
                widget = QSpinBox()
                widget.setRange(-2147483648, 2147483647)
                if default_value is not None:
                    widget.setValue(int(default_value))
                widget.setSpecialValueText('null' if field_type == 'int?' else '')
            elif field_type == 'float' or field_type == 'float?':
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(2)
                if default_value is not None:
                    widget.setValue(float(default_value))
            elif field_type == 'bool' or field_type == 'bool?':
                widget = QCheckBox()
                if default_value is not None:
                    widget.setChecked(bool(default_value))
            elif field_type == 'json':
                widget = QTextEdit()
                widget.setMaximumHeight(100)
                widget.setPlainText(json.dumps(default_value, ensure_ascii=False, indent=2))
            else:  # str, str?
                widget = QLineEdit()
                if default_value is not None:
                    widget.setText(str(default_value))

            self._field_widgets[field_name] = widget

            # 레이블에 설명 추가
            label_text = f'{field_name}'
            if description:
                label_text = f'<b>{field_name}</b><br><small>{description}</small>'

            label = QLabel(label_text)
            label.setTextFormat(Qt.TextFormat.RichText)
            self.fields_form_layout.addRow(label, widget)

        self.send_button.setEnabled(self._tcp_client.socket is not None)
        self.reset_button.setEnabled(True)

    def _reset_fields(self):
        """필드를 기본값으로 초기화"""
        if not self._current_message_type:
            return

        msg_def = self._message_definitions.get(self._current_message_type)
        if not msg_def:
            return

        fields = msg_def.get('fields', {})

        for field_name, widget in self._field_widgets.items():
            field_info = fields.get(field_name, {})
            default_value = field_info.get('default', '')
            field_type = field_info.get('type', 'str')

            if isinstance(widget, QSpinBox):
                if default_value is not None:
                    widget.setValue(int(default_value))
            elif isinstance(widget, QDoubleSpinBox):
                if default_value is not None:
                    widget.setValue(float(default_value))
            elif isinstance(widget, QCheckBox):
                if default_value is not None:
                    widget.setChecked(bool(default_value))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(json.dumps(default_value, ensure_ascii=False, indent=2))
            elif isinstance(widget, QLineEdit):
                if default_value is not None:
                    widget.setText(str(default_value))

    def _connect_to_server(self):
        """TCP 서버에 연결"""
        host = self.host_input.text().strip()
        port_text = self.port_input.text().strip()

        try:
            port = int(port_text)
        except ValueError:
            self._log_message(f'에러: 유효하지 않은 포트 번호: {port_text}', 'error')
            return

        self._log_message(f'연결 시도: {host}:{port}...', 'info')

        if self._tcp_client.connect(host, port):
            self._log_message(f'성공: {host}:{port}에 연결됨', 'success')
            self.connection_status_label.setText('🟢 연결됨')
            self.connection_status_label.setStyleSheet('color: #4ec9b0;')
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)

            if self._current_message_type:
                self.send_button.setEnabled(True)
        else:
            self._log_message(f'에러: 연결 실패', 'error')

    def _disconnect_from_server(self):
        """TCP 서버 연결 해제"""
        self._tcp_client.disconnect()
        self._log_message('연결 해제됨', 'info')
        self.connection_status_label.setText('⚪ 연결 안됨')
        self.connection_status_label.setStyleSheet('')
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.send_button.setEnabled(False)

    def _send_message(self):
        """메시지 전송"""
        if not self._current_message_type:
            self._log_message('에러: 메시지 타입이 선택되지 않았습니다.', 'error')
            return

        if not self._tcp_client.socket:
            self._log_message('에러: TCP 연결이 없습니다.', 'error')
            return

        msg_def = self._message_definitions.get(self._current_message_type)
        if not msg_def:
            return

        fields = msg_def.get('fields', {})
        data = {}

        try:
            for field_name, widget in self._field_widgets.items():
                field_info = fields.get(field_name, {})
                field_type = field_info.get('type', 'str')

                value = None
                if isinstance(widget, QSpinBox):
                    value = widget.value()
                    if field_type == 'int?' and value == widget.minimum():
                        value = None
                elif isinstance(widget, QDoubleSpinBox):
                    value = widget.value()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QTextEdit):
                    text = widget.toPlainText().strip()
                    try:
                        value = json.loads(text) if text else {}
                    except json.JSONDecodeError as e:
                        self._log_message(f'에러: {field_name} 필드의 JSON 형식이 잘못되었습니다: {e}', 'error')
                        return
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if field_type == 'int':
                        value = int(text) if text else 0
                    elif field_type == 'float':
                        value = float(text) if text else 0.0
                    else:  # str, str?
                        value = text if text else None

                if value is not None or '?' in field_type:
                    data[field_name] = value

        except Exception as e:
            self._log_message(f'에러: 필드 값 처리 실패: {e}', 'error')
            return

        message = {
            'type': self._current_message_type,
            'data': data,
        }

        if self._tcp_client.send(message):
            self._log_message(f'📤 전송: {self._current_message_type}', 'info')
            self._log_message(json.dumps(message, indent=2, ensure_ascii=False), 'debug')
        else:
            self._log_message(f'에러: 메시지 전송 실패', 'error')

    def _on_message_received(self, message: dict):
        """메시지 수신 시 처리"""
        msg_type = message.get('type', 'unknown')
        self._log_message(f'📥 수신: {msg_type}', 'success')
        self._log_message(json.dumps(message, indent=2, ensure_ascii=False), 'debug')

    def _on_connection_lost(self):
        """연결 끊김 시 처리"""
        self._log_message('에러: 서버 연결이 끊어졌습니다', 'error')
        self.connection_status_label.setText('🔴 연결 끊김')
        self.connection_status_label.setStyleSheet('color: #f48771;')
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.send_button.setEnabled(False)

    def _log_message(self, message: str, level: str = 'info'):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        color_map = {
            'info': '#d4d4d4',
            'success': '#4ec9b0',
            'error': '#f48771',
            'debug': '#9cdcfe',
        }
        color = color_map.get(level, '#d4d4d4')
        html = f'<span style="color: #808080;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.result_log.append(html)

        # 자동 스크롤
        scrollbar = self.result_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def cleanup(self):
        """정리 작업"""
        self._tcp_client.disconnect()
