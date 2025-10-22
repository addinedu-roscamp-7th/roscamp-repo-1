"""'ROS2 제어 패널' 탭의 UI 로직"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTreeWidgetItem,
)

from shopee_interfaces.srv import (
    PickeeWorkflowStartTask,
    PickeeWorkflowMoveToSection,
    PickeeWorkflowMoveToPackaging,
    PickeeWorkflowReturnToBase,
    PickeeWorkflowReturnToStaff,
    PickeeProductDetect,
    PickeeProductProcessSelection,
    PickeeWorkflowEndShopping,
    PickeeMainVideoStreamStart,
    PickeeMainVideoStreamStop,
    PackeePackingCheckAvailability,
    PackeePackingStart,
)

# .ui 파일로부터 생성된 클래스 import
from ..ui_gen.tab_ros_service_request_ui import Ui_RosServiceRequestTab
from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class RosServiceRequestTab(BaseTab, Ui_RosServiceRequestTab):
    """'ROS2 서비스 요청' 탭의 UI 및 로직"""

    def __init__(self, robot_coordinator=None, loop=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)  # .ui 파일에 정의된 UI 설정

        self._robot_coordinator = robot_coordinator
        self._loop = loop  # asyncio 이벤트 루프 저장
        self._current_service = None
        self._param_widgets = {}
        self._service_definitions = self._build_service_definitions()

        self._init_logic()

    def _init_logic(self):
        """로직 및 시그널-슬롯 연결 초기화"""
        self._populate_service_tree()
        self.service_tree.itemClicked.connect(self._on_service_selected)
        self.search_input.textChanged.connect(self._filter_services)
        self.reset_button.clicked.connect(self._reset_parameters)
        self.call_button.clicked.connect(self._call_service)
        self.clear_log_button.clicked.connect(self._clear_log)

    def _build_service_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Main Service가 호출할 수 있는 모든 ROS2 서비스 정의

        Returns:
            서비스 이름을 키로, 서비스 정의를 값으로 하는 딕셔너리
        """
        return {
            # Pickee 워크플로우 서비스
            '/pickee/workflow/start_task': {
                'category': 'Pickee 워크플로우',
                'name': '작업 시작',
                'service_type': PickeeWorkflowStartTask,
                'description': 'Pickee 로봇에게 피킹 작업을 시작하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'user_id': {'type': 'str', 'default': 'test_user', 'required': True, 'description': '사용자 ID'},
                    'product_list': {'type': 'list', 'default': [], 'required': False, 'description': '상품 위치 리스트 (비워두면 빈 리스트)'},
                },
            },
            '/pickee/workflow/move_to_section': {
                'category': 'Pickee 워크플로우',
                'name': '섹션 이동',
                'service_type': PickeeWorkflowMoveToSection,
                'description': 'Pickee 로봇이 특정 섹션으로 이동하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'location_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '위치 ID'},
                    'section_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '섹션 ID'},
                },
            },
            '/pickee/workflow/move_to_packaging': {
                'category': 'Pickee 워크플로우',
                'name': '포장대 이동',
                'service_type': PickeeWorkflowMoveToPackaging,
                'description': 'Pickee 로봇이 포장대로 이동하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'location_id': {'type': 'int', 'default': 10, 'required': True, 'min': 1, 'description': '포장대 위치 ID'},
                },
            },
            '/pickee/workflow/return_to_base': {
                'category': 'Pickee 워크플로우',
                'name': '복귀 (홈)',
                'service_type': PickeeWorkflowReturnToBase,
                'description': 'Pickee 로봇이 홈 위치로 복귀하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'location_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '홈 위치 ID'},
                },
            },
            '/pickee/workflow/return_to_staff': {
                'category': 'Pickee 워크플로우',
                'name': '복귀 (직원)',
                'service_type': PickeeWorkflowReturnToStaff,
                'description': 'Pickee 로봇이 마지막 직원 위치로 복귀하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                },
            },
            '/pickee/workflow/end_shopping': {
                'category': 'Pickee 워크플로우',
                'name': '쇼핑 종료',
                'service_type': PickeeWorkflowEndShopping,
                'description': 'Pickee 로봇의 쇼핑을 종료하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                },
            },
            # Pickee 상품 관련 서비스
            '/pickee/product/detect': {
                'category': 'Pickee 상품',
                'name': '상품 인식',
                'service_type': PickeeProductDetect,
                'description': 'Pickee 로봇이 상품을 인식하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'product_ids': {'type': 'list_int', 'default': '1,2,3', 'required': True, 'description': '상품 ID 리스트 (쉼표로 구분)'},
                },
            },
            '/pickee/product/process_selection': {
                'category': 'Pickee 상품',
                'name': '상품 선택 처리',
                'service_type': PickeeProductProcessSelection,
                'description': '사용자가 선택한 상품을 처리하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'product_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '상품 ID'},
                    'bbox_number': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': 'BBox 번호'},
                },
            },
            # Pickee 영상 스트리밍 서비스
            '/pickee/video_stream/start': {
                'category': 'Pickee 영상',
                'name': '영상 스트림 시작',
                'service_type': PickeeMainVideoStreamStart,
                'description': 'Pickee 로봇의 영상 스트리밍을 시작합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                },
            },
            '/pickee/video_stream/stop': {
                'category': 'Pickee 영상',
                'name': '영상 스트림 중지',
                'service_type': PickeeMainVideoStreamStop,
                'description': 'Pickee 로봇의 영상 스트리밍을 중지합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 1, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                },
            },
            # Packee 서비스
            '/packee/packing/check_availability': {
                'category': 'Packee',
                'name': '작업 가능 여부 확인',
                'service_type': PackeePackingCheckAvailability,
                'description': 'Packee 로봇의 작업 가능 여부를 확인합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 2, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                },
            },
            '/packee/packing/start': {
                'category': 'Packee',
                'name': '포장 시작',
                'service_type': PackeePackingStart,
                'description': 'Packee 로봇에게 포장 작업을 시작하도록 명령합니다.',
                'fields': {
                    'robot_id': {'type': 'int', 'default': 2, 'required': True, 'min': 1, 'description': '로봇 ID'},
                    'order_id': {'type': 'int', 'default': 0, 'required': True, 'min': 0, 'description': '주문 ID'},
                    'products': {'type': 'list', 'default': [], 'required': False, 'description': '상품 정보 리스트 (비워두면 빈 리스트)'},
                },
            },
        }

    def _populate_service_tree(self):
        """서비스 트리를 채웁니다"""
        self.service_tree.clear()
        categories: Dict[str, QTreeWidgetItem] = {}

        for service_path, service_def in self._service_definitions.items():
            category = service_def['category']
            name = service_def['name']

            if category not in categories:
                category_item = QTreeWidgetItem(self.service_tree, [category])
                category_item.setExpanded(True)
                categories[category] = category_item

            service_item = QTreeWidgetItem(categories[category], [name])
            service_item.setData(0, Qt.ItemDataRole.UserRole, service_path)

    def _filter_services(self, text: str):
        """서비스 검색 필터링"""
        text = text.lower()

        for i in range(self.service_tree.topLevelItemCount()):
            category_item = self.service_tree.topLevelItem(i)
            category_visible = False

            for j in range(category_item.childCount()):
                service_item = category_item.child(j)
                service_path = service_item.data(0, Qt.ItemDataRole.UserRole)
                service_def = self._service_definitions.get(service_path, {})
                service_name = service_def.get('name', '').lower()

                visible = text in service_name or text in service_path.lower()
                service_item.setHidden(not visible)

                if visible:
                    category_visible = True

            category_item.setHidden(not category_visible)

    def _on_service_selected(self, item: QTreeWidgetItem, column: int):
        """서비스 선택 시 호출"""
        service_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not service_path:
            return

        self._current_service = service_path
        service_def = self._service_definitions.get(service_path)
        if not service_def:
            return

        # 서비스 설명 업데이트
        desc = service_def.get('description', '')
        self.service_desc_label.setText(f'<b>{service_def["name"]}</b><br>{desc}<br><i>{service_path}</i>')

        # 파라미터 폼 생성
        self._build_parameter_form(service_def)

        # 버튼 활성화
        self.reset_button.setEnabled(True)
        self.call_button.setEnabled(True)

    def _build_parameter_form(self, service_def: Dict[str, Any]):
        """파라미터 입력 폼을 생성합니다"""
        # 기존 폼 제거
        while self.param_form_layout.rowCount() > 0:
            self.param_form_layout.removeRow(0)

        self._param_widgets = {}
        fields = service_def.get('fields', {})

        if not fields:
            label = QLabel('이 서비스는 파라미터가 없습니다.')
            label.setStyleSheet('color: #999; font-style: italic;')
            self.param_form_layout.addRow(label)
            return

        for field_name, field_info in fields.items():
            field_type = field_info.get('type', 'str')
            default_value = field_info.get('default', '')
            description = field_info.get('description', '')

            # 필드 타입에 따라 위젯 생성
            if field_type == 'int':
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(int(default_value))
            elif field_type == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setValue(float(default_value))
            elif field_type == 'bool':
                widget = QCheckBox()
                widget.setChecked(bool(default_value))
            elif field_type == 'list_int':
                widget = QLineEdit()
                widget.setText(str(default_value))
                widget.setPlaceholderText('예: 1,2,3')
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

    def _reset_parameters(self):
        """파라미터를 기본값으로 초기화"""
        if not self._current_service:
            return

        service_def = self._service_definitions.get(self._current_service)
        if not service_def:
            return

        fields = service_def.get('fields', {})
        for field_name, widget in self._param_widgets.items():
            field_info = fields.get(field_name, {})
            default_value = field_info.get('default', '')
            field_type = field_info.get('type', 'str')

            if isinstance(widget, QSpinBox):
                widget.setValue(int(default_value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(default_value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(default_value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(default_value))

        self._log_message('파라미터를 기본값으로 초기화했습니다.', 'info')

    def _call_service(self):
        """서비스 호출"""
        if not self._current_service:
            self._log_message('서비스가 선택되지 않았습니다.', 'error')
            return

        service_def = self._service_definitions.get(self._current_service)
        if not service_def:
            self._log_message(f'서비스 정의를 찾을 수 없습니다: {self._current_service}', 'error')
            return

        params = {}
        fields = service_def.get('fields', {})

        # 1. 파라미터 검증 및 수집
        for field_name, field_info in fields.items():
            widget = self._param_widgets.get(field_name)
            if not widget:
                continue

            field_type = field_info.get('type', 'str')
            is_required = field_info.get('required', False)
            min_value = field_info.get('min')

            try:
                value = None
                if isinstance(widget, QSpinBox):
                    value = widget.value()
                    if min_value is not None and value < min_value:
                        self._log_message(f'파라미터 값 오류: {field_name}은(는) {min_value} 이상이어야 합니다.', 'error')
                        return
                    params[field_name] = value

                elif isinstance(widget, QDoubleSpinBox):
                    value = widget.value()
                    if min_value is not None and value < min_value:
                        self._log_message(f'파라미터 값 오류: {field_name}은(는) {min_value} 이상이어야 합니다.', 'error')
                        return
                    params[field_name] = value

                elif isinstance(widget, QCheckBox):
                    params[field_name] = widget.isChecked()

                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if is_required and not text:
                        self._log_message(f'필수 파라미터 누락: {field_name}', 'error')
                        return

                    if not text:
                        # 필수 아닌데 비어있는 경우, 타입에 따라 기본값 설정
                        if field_type == 'list_int' or field_type == 'list':
                            params[field_name] = []
                        else:
                            params[field_name] = ''
                        continue

                    if field_type == 'list_int':
                        value = [int(x.strip()) for x in text.split(',')]
                        params[field_name] = value
                    elif field_type == 'list':
                        try:
                            params[field_name] = json.loads(text)
                        except json.JSONDecodeError:
                            self._log_message(f'파라미터 형식 오류 ({field_name}): 유효한 JSON 리스트가 아닙니다.', 'error')
                            return
                    elif field_type == 'int':
                        value = int(text)
                        if min_value is not None and value < min_value:
                            self._log_message(f'파라미터 값 오류: {field_name}은(는) {min_value} 이상이어야 합니다.', 'error')
                            return
                        params[field_name] = value
                    elif field_type == 'float':
                        value = float(text)
                        if min_value is not None and value < min_value:
                             self._log_message(f'파라미터 값 오류: {field_name}은(는) {min_value} 이상이어야 합니다.', 'error')
                             return
                        params[field_name] = value
                    else:  # str
                        params[field_name] = text

            except (ValueError, TypeError) as e:
                self._log_message(f'파라미터 파싱 오류 ({field_name}): {e}', 'error')
                return

        # 2. 로그 기록
        self._log_message(f'서비스 호출: {self._current_service}', 'info')
        self._log_message(f'파라미터: {json.dumps(params, indent=2, ensure_ascii=False)}', 'debug')

        # 3. 실제 서비스 호출 (비동기-스레드 안전)
        if self._robot_coordinator and self._loop:
            try:
                coro = self._async_call_service(service_def, params)
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                future.add_done_callback(self._handle_future_completion)
            except Exception as e:
                self._log_message(f'서비스 호출 스케줄링 실패: {e}', 'error')
        elif not self._robot_coordinator:
            self._log_message('RobotCoordinator가 설정되지 않았습니다.', 'error')
        else:
            self._log_message('이벤트 루프가 설정되지 않았습니다.', 'error')

    def _handle_future_completion(self, future):
        """비동기 Future 완료 시 예외를 확인하고 로그를 남깁니다."""
        try:
            future.result()
        except asyncio.CancelledError:
            logger.warning('서비스 호출 Future가 취소되었습니다.')
        except Exception as e:
            logger.exception(f'서비스 호출 Future 실행 중 예외 발생: {e}')

    async def _async_call_service(self, service_def: Dict[str, Any], params: Dict[str, Any]):
        """비동기로 서비스 호출"""
        service_name = self._current_service
        service_type = service_def['service_type']

        try:
            # Request 객체 생성
            request = service_type.Request()
            for field_name, value in params.items():
                if hasattr(request, field_name):
                    setattr(request, field_name, value)

            # 서비스 이름에 따라 적절한 메서드 호출
            response = None
            if service_name == '/pickee/workflow/start_task':
                response = await self._robot_coordinator.dispatch_pick_task(request)
            elif service_name == '/pickee/workflow/move_to_section':
                response = await self._robot_coordinator.dispatch_move_to_section(request)
            elif service_name == '/pickee/workflow/move_to_packaging':
                response = await self._robot_coordinator.dispatch_move_to_packaging(request)
            elif service_name == '/pickee/workflow/return_to_base':
                response = await self._robot_coordinator.dispatch_return_to_base(request)
            elif service_name == '/pickee/workflow/return_to_staff':
                response = await self._robot_coordinator.dispatch_return_to_staff(request)
            elif service_name == '/pickee/workflow/end_shopping':
                response = await self._robot_coordinator.dispatch_shopping_end(request)
            elif service_name == '/pickee/product/detect':
                response = await self._robot_coordinator.dispatch_product_detect(request)
            elif service_name == '/pickee/product/process_selection':
                response = await self._robot_coordinator.dispatch_pick_process(request)
            elif service_name == '/pickee/video_stream/start':
                response = await self._robot_coordinator.dispatch_video_stream_start(request)
            elif service_name == '/pickee/video_stream/stop':
                response = await self._robot_coordinator.dispatch_video_stream_stop(request)
            elif service_name == '/packee/packing/check_availability':
                response = await self._robot_coordinator.check_packee_availability(request)
            elif service_name == '/packee/packing/start':
                response = await self._robot_coordinator.dispatch_pack_task(request)
            else:
                self._log_message(f'지원하지 않는 서비스입니다: {service_name}', 'error')
                return

            # 응답 로그
            if response:
                response_dict = {}
                if hasattr(response, 'get_fields_and_field_types'):
                    fields = response.get_fields_and_field_types().keys()
                    for field in fields:
                        response_dict[field] = str(getattr(response, field))
                else:
                    response_dict = {'response': str(response)}

                success = response_dict.get('success', 'unknown')
                if success == 'True' or success is True:
                    self._log_message('서비스 호출 성공 ✓', 'success')
                else:
                    self._log_message('서비스 호출 실패 ✗', 'error')

                self._log_message(f'응답: {json.dumps(response_dict, indent=2, ensure_ascii=False)}', 'debug')
            else:
                self._log_message('응답이 없습니다.', 'warning')

        except Exception as e:
            logger.exception(f'서비스 호출 중 예외 발생: {e}')
            self._log_message(f'예외 발생: {e}', 'error')

    def _clear_log(self):
        """로그 지우기"""
        self.result_log.clear()

    def _log_message(self, message: str, level: str = 'info'):
        """
        로그 메시지 추가

        Args:
            message: 로그 메시지
            level: 로그 레벨 ('info', 'success', 'warning', 'error', 'debug')
        """
        timestamp = datetime.now().strftime('%H:%M:%S')

        color_map = {
            'info': '#d4d4d4',
            'success': '#4ec9b0',
            'warning': '#dcdcaa',
            'error': '#f48771',
            'debug': '#9cdcfe',
        }
        color = color_map.get(level, '#d4d4d4')

        html = f'<span style="color: #808080;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.result_log.append(html)

        # 자동 스크롤
        scrollbar = self.result_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_data(self, data):
        """이 탭은 스냅샷 데이터를 사용하지 않습니다."""
        pass
