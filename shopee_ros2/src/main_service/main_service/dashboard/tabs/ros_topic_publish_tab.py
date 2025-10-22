"""
'ROS2 토픽 시뮬레이터' 탭의 UI 로직

이 탭은 Main Service를 테스트하기 위해 로봇→Main 방향의 토픽을 시뮬레이션합니다.
실제 로봇 없이 Main Service의 토픽 수신 기능을 테스트할 수 있습니다.

지원 토픽:
- Pickee 로봇 상태/이벤트 토픽 (7개)
- Packee 로봇 상태/이벤트 토픽 (3개)
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTreeWidgetItem,
)

# .ui 파일로부터 생성된 클래스 import
from ..ui_gen.tab_ros_topic_publish_ui import Ui_RosTopicPublishTab
from .base_tab import BaseTab

# Shopee 인터페이스 메시지 타입
from shopee_interfaces.msg import (
    PickeeMoveStatus,
    PickeeArrival,
    PickeeProductDetection,
    PickeeCartHandover,
    PickeeRobotStatus,
    PickeeProductSelection,
    PickeeProductLoaded,
    PackeePackingComplete,
    PackeeRobotStatus,
    PackeeAvailability,
)

logger = logging.getLogger(__name__)


class RosTopicPublishTab(BaseTab, Ui_RosTopicPublishTab):
    """
    'ROS2 토픽 시뮬레이터' 탭의 UI 및 로직

    로봇→Main 방향의 토픽을 발행하여 Main Service의 토픽 수신 기능을 테스트합니다.
    """

    def __init__(self, ros_node=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)  # .ui 파일에 정의된 UI 설정

        self._ros_node = ros_node
        self._publishers = {}  # 생성된 Publisher 캐시
        self._current_topic = None
        self._param_widgets = {}
        self._topic_definitions = self._build_topic_definitions()

        self._init_logic()

    def _init_logic(self):
        """로직 및 시그널-슬롯 연결 초기화"""
        self._populate_topic_tree()
        self.topic_tree.itemClicked.connect(self._on_topic_selected)
        self.publish_button.clicked.connect(self._publish_message)

    def _build_topic_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        시뮬레이션 가능한 모든 ROS2 토픽 정의 (로봇→Main 방향)

        Returns:
            Dict: 토픽 경로를 키로, 토픽 정의를 값으로 하는 딕셔너리
        """
        return {
            # Pickee 토픽
            '/pickee/moving_status': {
                'category': 'Pickee 상태',
                'name': '이동 시작 알림',
                'msg_type': PickeeMoveStatus,
                'description': 'Pickee 로봇의 이동 시작을 알립니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'location_id': {'type': 'int', 'default': 1, 'description': '목적지 위치 ID'},
                },
            },
            '/pickee/arrival_notice': {
                'category': 'Pickee 상태',
                'name': '도착 보고',
                'msg_type': PickeeArrival,
                'description': 'Pickee 로봇의 목적지 도착을 보고합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'location_id': {'type': 'int', 'default': 1, 'description': '도착 위치 ID'},
                    'section_id': {'type': 'int', 'default': 1, 'description': '섹션 ID (섹션 아니면 -1)'},
                },
            },
            '/pickee/robot_status': {
                'category': 'Pickee 상태',
                'name': '로봇 상태 전송',
                'msg_type': PickeeRobotStatus,
                'description': 'Pickee 로봇의 현재 상태를 전송합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'state': {'type': 'str', 'default': 'PK_S10', 'description': 'Pickee 상태 코드'},
                    'battery_level': {'type': 'float', 'default': 85.0, 'description': '배터리 레벨 (%)'},
                    'current_order_id': {'type': 'int', 'default': 0, 'description': '현재 주문 ID'},
                    'position_x': {'type': 'float', 'default': 0.0, 'description': 'X 좌표'},
                    'position_y': {'type': 'float', 'default': 0.0, 'description': 'Y 좌표'},
                    'orientation_z': {'type': 'float', 'default': 0.0, 'description': 'Z 방향'},
                },
            },
            '/pickee/product_detected': {
                'category': 'Pickee 상품',
                'name': '상품 위치 인식 완료',
                'msg_type': PickeeProductDetection,
                'description': 'Pickee 로봇이 상품 위치 인식을 완료했습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'products': {'type': 'list', 'default': [], 'description': 'DetectedProduct[] (JSON)'},
                },
            },
            '/pickee/cart_handover_complete': {
                'category': 'Pickee 상품',
                'name': '장바구니 교체 완료',
                'msg_type': PickeeCartHandover,
                'description': 'Pickee 로봇의 장바구니 교체가 완료되었습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                },
            },
            '/pickee/product/selection_result': {
                'category': 'Pickee 상품',
                'name': '담기 완료 보고',
                'msg_type': PickeeProductSelection,
                'description': '사용자가 선택한 상품을 장바구니에 담았습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'product_id': {'type': 'int', 'default': 1, 'description': '상품 ID'},
                    'success': {'type': 'bool', 'default': True, 'description': '성공 여부'},
                    'quantity': {'type': 'int', 'default': 1, 'description': '수량'},
                    'message': {'type': 'str', 'default': 'Success', 'description': '메시지'},
                },
            },
            '/pickee/product/loaded': {
                'category': 'Pickee 상품',
                'name': '창고 물품 적재 완료 보고',
                'msg_type': PickeeProductLoaded,
                'description': '창고에서 물품 적재가 완료되었습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'description': '로봇 ID'},
                    'product_id': {'type': 'int', 'default': 1, 'description': '상품 ID'},
                    'quantity': {'type': 'int', 'default': 1, 'description': '수량'},
                    'success': {'type': 'bool', 'default': True, 'description': '성공 여부'},
                    'message': {'type': 'str', 'default': 'Loaded', 'description': '메시지'},
                },
            },
            # Packee 토픽
            '/packee/packing_complete': {
                'category': 'Packee',
                'name': '포장 완료 알림',
                'msg_type': PackeePackingComplete,
                'description': 'Packee 로봇의 포장 작업이 완료되었습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 2, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'success': {'type': 'bool', 'default': True, 'description': '성공 여부'},
                    'packed_items': {'type': 'int', 'default': 5, 'description': '포장한 아이템 수'},
                    'message': {'type': 'str', 'default': 'Packing completed', 'description': '메시지'},
                },
            },
            '/packee/robot_status': {
                'category': 'Packee',
                'name': '로봇 상태 전송',
                'msg_type': PackeeRobotStatus,
                'description': 'Packee 로봇의 현재 상태를 전송합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 2, 'description': '로봇 ID'},
                    'state': {'type': 'str', 'default': 'idle', 'description': '로봇 상태'},
                    'current_order_id': {'type': 'int', 'default': 0, 'description': '현재 주문 ID'},
                    'items_in_cart': {'type': 'int', 'default': 0, 'description': '카트 내 아이템 수'},
                },
            },
            '/packee/availability_result': {
                'category': 'Packee',
                'name': '작업 가능 확인 완료',
                'msg_type': PackeeAvailability,
                'description': 'Packee 로봇의 작업 가능 여부 확인이 완료되었습니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 2, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'description': '주문 ID'},
                    'available': {'type': 'bool', 'default': True, 'description': '작업 가능 여부'},
                    'cart_detected': {'type': 'bool', 'default': True, 'description': '카트 감지 여부'},
                    'message': {'type': 'str', 'default': 'Ready for packing', 'description': '메시지'},
                },
            },
        }

    def _populate_topic_tree(self):
        """토픽 트리를 채웁니다"""
        self.topic_tree.clear()
        categories: Dict[str, QTreeWidgetItem] = {}

        for topic_path, topic_def in self._topic_definitions.items():
            category = topic_def['category']
            name = topic_def['name']

            if category not in categories:
                category_item = QTreeWidgetItem(self.topic_tree, [category])
                category_item.setExpanded(True)
                categories[category] = category_item

            topic_item = QTreeWidgetItem(categories[category], [name])
            topic_item.setData(0, Qt.ItemDataRole.UserRole, topic_path)

    def _on_topic_selected(self, item: QTreeWidgetItem, column: int):
        """토픽 선택 시 메시지 입력 폼 생성"""
        topic_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not topic_path:
            return

        self._current_topic = topic_path
        topic_def = self._topic_definitions.get(topic_path)
        if not topic_def:
            return

        desc = topic_def.get('description', '')
        self.topic_desc_label.setText(f'<b>{topic_def["name"]}</b><br>{desc}<br><i>{topic_path}</i>')

        # 기존 폼 위젯 제거
        while self.param_form_layout.rowCount() > 0:
            self.param_form_layout.removeRow(0)
        self._param_widgets = {}

        # fields 정의가 있으면 사용, 없으면 메시지 타입에서 자동 추출
        fields = topic_def.get('fields', {})

        if not fields:
            label = QLabel('이 토픽은 필드가 없습니다.')
            label.setStyleSheet('color: #999; font-style: italic;')
            self.param_form_layout.addRow(label)
            self.publish_button.setEnabled(True)
            return

        for field_name, field_info in fields.items():
            field_type = field_info.get('type', 'str')
            default_value = field_info.get('default', '')
            description = field_info.get('description', '')

            # 필드 타입에 따라 위젯 생성
            if field_type == 'int':
                widget = QSpinBox()
                widget.setRange(-2147483648, 2147483647)
                widget.setValue(int(default_value))
            elif field_type == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(2)
                widget.setValue(float(default_value))
            elif field_type == 'bool':
                widget = QCheckBox()
                widget.setChecked(bool(default_value))
            elif field_type == 'list':
                widget = QLineEdit()
                widget.setText('[]')
                widget.setPlaceholderText('JSON 형식 리스트, 예: [{"key": "value"}]')
            else:  # str
                widget = QLineEdit()
                widget.setText(str(default_value))

            self._param_widgets[field_name] = widget

            # 레이블에 설명 추가
            label_text = f'{field_name}'
            if description:
                label_text = f'<b>{field_name}</b><br><small>{description}</small>'

            label = QLabel(label_text)
            label.setTextFormat(Qt.TextFormat.RichText)
            self.param_form_layout.addRow(label, widget)

        self.publish_button.setEnabled(True)

    def _publish_message(self):
        """메시지 발행"""
        if not self._current_topic or not self._ros_node:
            self._log_message('에러: 토픽이 선택되지 않았거나 ROS 노드가 없습니다.', 'error')
            return

        topic_def = self._topic_definitions.get(self._current_topic)
        if not topic_def:
            self._log_message(f'에러: 토픽 정의를 찾을 수 없습니다: {self._current_topic}', 'error')
            return

        topic_name = self._current_topic
        msg_type = topic_def['msg_type']

        if topic_name not in self._publishers:
            try:
                self._publishers[topic_name] = self._ros_node.create_publisher(msg_type, topic_name, 10)
                self._log_message(f'알림: Publisher for {topic_name} 생성됨.', 'info')
            except Exception as e:
                self._log_message(f'에러: Publisher 생성 실패: {e}', 'error')
                return

        publisher = self._publishers[topic_name]
        msg = msg_type()

        fields = topic_def.get('fields', {})

        try:
            for field_name, widget in self._param_widgets.items():
                field_info = fields.get(field_name, {})
                field_type = field_info.get('type', 'str')

                value = None
                if isinstance(widget, QSpinBox):
                    value = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    value = widget.value()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if field_type == 'list':
                        try:
                            value = json.loads(text) if text else []
                        except json.JSONDecodeError:
                            self._log_message(f'에러: {field_name} 필드의 JSON 형식이 잘못되었습니다.', 'error')
                            return
                    elif field_type == 'int':
                        value = int(text) if text else 0
                    elif field_type == 'float':
                        value = float(text) if text else 0.0
                    else:  # str
                        value = text

                # 메시지 필드에 값 설정
                if value is not None and hasattr(msg, field_name):
                    setattr(msg, field_name, value)

        except Exception as e:
            self._log_message(f'에러: 메시지 필드 값 설정 실패: {e}', 'error')
            return

        try:
            publisher.publish(msg)
            self._log_message(f'성공: {topic_name}에 메시지 발행 완료.', 'success')
            self._log_message(f'내용: {json.dumps(self._msg_to_dict(msg), indent=2, ensure_ascii=False)}', 'debug')
        except Exception as e:
            self._log_message(f'에러: 메시지 발행 실패: {e}', 'error')

    def _msg_to_dict(self, msg) -> Dict[str, Any]:
        """ROS 메시지를 딕셔너리로 변환"""
        result = {}
        if hasattr(msg, 'get_fields_and_field_types'):
            fields = msg.get_fields_and_field_types().keys()
            for field in fields:
                value = getattr(msg, field)
                if isinstance(value, list):
                    result[field] = [self._msg_to_dict(item) if hasattr(item, 'get_fields_and_field_types') else item for item in value]
                elif hasattr(value, 'get_fields_and_field_types'):
                    result[field] = self._msg_to_dict(value)
                else:
                    result[field] = value
        return result

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
        if not self._ros_node:
            return
        for publisher in self._publishers.values():
            self._ros_node.destroy_publisher(publisher)
        self._publishers.clear()
