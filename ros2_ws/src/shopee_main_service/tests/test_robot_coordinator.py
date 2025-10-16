"""
RobotCoordinator 통합 테스트

ROS2 서비스/토픽 통신을 Mock으로 테스트합니다.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import rclpy

from shopee_interfaces.msg import (
    PackeeAvailability,
    PackeePackingComplete,
    PackeeRobotStatus,
    PickeeArrival,
    PickeeCartHandover,
    PickeeMoveStatus,
    PickeeProductDetection,
    PickeeProductLoaded,
    PickeeProductSelection,
    PickeeRobotStatus,
)
from shopee_interfaces.srv import (
    PackeePackingCheckAvailability,
    PackeePackingStart,
    PickeeProductDetect,
    PickeeProductProcessSelection,
    PickeeWorkflowEndShopping,
    PickeeWorkflowMoveToPackaging,
    PickeeWorkflowMoveToSection,
    PickeeWorkflowReturnToBase,
    PickeeWorkflowReturnToStaff,
    PickeeWorkflowStartTask,
    PickeeMainVideoStreamStart,
    PickeeMainVideoStreamStop,
)

from shopee_main_service.robot_coordinator import RobotCoordinator
from shopee_main_service.constants import RobotType, RobotStatus

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope='session', autouse=True)
def init_rclpy():
    """ROS2 초기화 (세션 전체에 한 번만 실행)"""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def mock_state_store() -> MagicMock:
    """로봇 상태 스토어 Mock"""
    store = MagicMock()
    store.upsert_state = AsyncMock()
    store.release = AsyncMock()
    store.mark_offline = AsyncMock()
    store.get_state = AsyncMock()
    store.list_available = AsyncMock()
    return store


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """이벤트 버스 Mock"""
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def robot_coordinator(mock_state_store: MagicMock, mock_event_bus: MagicMock) -> RobotCoordinator:
    """RobotCoordinator 인스턴스 생성"""
    with patch('shopee_main_service.robot_coordinator.settings') as mock_settings:
        mock_settings.ROS_SERVICE_RETRY_ATTEMPTS = 1
        mock_settings.ROS_SERVICE_RETRY_BASE_DELAY = 0.0
        mock_settings.ROS_SERVICE_TIMEOUT = 1.0
        mock_settings.ROS_STATUS_HEALTH_TIMEOUT = 0  # 헬스 타이머 비활성화
        coordinator = RobotCoordinator(state_store=mock_state_store, event_bus=mock_event_bus)
    return coordinator


class TestRobotCoordinatorServiceCalls:
    """ROS2 서비스 호출 테스트"""

    async def test_dispatch_pick_task_success(self, robot_coordinator: RobotCoordinator):
        """Pickee 작업 시작 명령 성공"""
        # Arrange
        request = PickeeWorkflowStartTask.Request(
            robot_id=1,
            order_id=100,
            user_id='test_user',
            product_list=[42, 43]
        )
        mock_response = PickeeWorkflowStartTask.Response(success=True, message='Task started')

        # Mock ROS2 서비스 클라이언트
        with patch.object(robot_coordinator._pickee_start_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_start_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_pick_task(request)

        # Assert
        assert response.success is True
        assert response.message == 'Task started'
        mock_call.assert_called_once_with(request)

    async def test_dispatch_pick_process_success(self, robot_coordinator: RobotCoordinator):
        """상품 선택 처리 명령 성공"""
        # Arrange
        request = PickeeProductProcessSelection.Request(
            robot_id=1,
            order_id=100,
            action='add'
        )
        mock_response = PickeeProductProcessSelection.Response(success=True)

        with patch.object(robot_coordinator._pickee_process_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_process_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_pick_process(request)

        # Assert
        assert response.success is True

    async def test_dispatch_shopping_end_success(self, robot_coordinator: RobotCoordinator):
        """쇼핑 종료 명령 성공"""
        # Arrange
        request = PickeeWorkflowEndShopping.Request(robot_id=1, order_id=100)
        mock_response = PickeeWorkflowEndShopping.Response(success=True)

        with patch.object(robot_coordinator._pickee_end_shopping_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_end_shopping_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_shopping_end(request)

        # Assert
        assert response.success is True

    async def test_dispatch_move_to_section_success(self, robot_coordinator: RobotCoordinator):
        """섹션 이동 명령 성공"""
        # Arrange
        request = PickeeWorkflowMoveToSection.Request(robot_id=1, order_id=100, section_id=5)
        mock_response = PickeeWorkflowMoveToSection.Response(success=True)

        with patch.object(robot_coordinator._pickee_move_section_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_move_section_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_move_to_section(request)

        # Assert
        assert response.success is True

    async def test_dispatch_move_to_packaging_success(self, robot_coordinator: RobotCoordinator):
        """포장대 이동 명령 성공"""
        # Arrange
        request = PickeeWorkflowMoveToPackaging.Request(robot_id=1, order_id=100)
        mock_response = PickeeWorkflowMoveToPackaging.Response(success=True)

        with patch.object(robot_coordinator._pickee_move_packaging_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_move_packaging_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_move_to_packaging(request)

        # Assert
        assert response.success is True

    async def test_dispatch_return_to_base_success(self, robot_coordinator: RobotCoordinator):
        """복귀 명령 성공"""
        # Arrange
        request = PickeeWorkflowReturnToBase.Request(robot_id=1)
        mock_response = PickeeWorkflowReturnToBase.Response(success=True)

        with patch.object(robot_coordinator._pickee_return_base_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_return_base_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_return_to_base(request)

        # Assert
        assert response.success is True

    async def test_dispatch_return_to_staff_success(self, robot_coordinator: RobotCoordinator):
        """직원 복귀 명령 성공"""
        # Arrange
        request = PickeeWorkflowReturnToStaff.Request(robot_id=1)
        mock_response = PickeeWorkflowReturnToStaff.Response(success=True)

        with patch.object(robot_coordinator._pickee_return_staff_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_return_staff_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_return_to_staff(request)

        # Assert
        assert response.success is True

    async def test_dispatch_product_detect_success(self, robot_coordinator: RobotCoordinator):
        """상품 인식 명령 성공"""
        # Arrange
        request = PickeeProductDetect.Request(robot_id=1, order_id=100, product_ids=[42, 43])
        mock_response = PickeeProductDetect.Response(success=True)

        with patch.object(robot_coordinator._pickee_product_detect_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_product_detect_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_product_detect(request)

        # Assert
        assert response.success is True

    async def test_dispatch_video_stream_start_success(self, robot_coordinator: RobotCoordinator):
        """영상 스트림 시작 성공"""
        # Arrange
        request = PickeeMainVideoStreamStart.Request(robot_id=1, user_id='admin', user_type='admin')
        mock_response = PickeeMainVideoStreamStart.Response(success=True)

        with patch.object(robot_coordinator._pickee_video_start_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_video_start_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_video_stream_start(request)

        # Assert
        assert response.success is True

    async def test_dispatch_video_stream_stop_success(self, robot_coordinator: RobotCoordinator):
        """영상 스트림 중지 성공"""
        # Arrange
        request = PickeeMainVideoStreamStop.Request(robot_id=1, user_id='admin', user_type='admin')
        mock_response = PickeeMainVideoStreamStop.Response(success=True)

        with patch.object(robot_coordinator._pickee_video_stop_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_video_stop_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_video_stream_stop(request)

        # Assert
        assert response.success is True

    async def test_check_packee_availability_success(self, robot_coordinator: RobotCoordinator):
        """Packee 가용성 확인 성공"""
        # Arrange
        request = PackeePackingCheckAvailability.Request(robot_id=2, order_id=100)
        mock_response = PackeePackingCheckAvailability.Response(success=True, message='Available')

        with patch.object(robot_coordinator._packee_check_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._packee_check_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.check_packee_availability(request)

        # Assert
        assert response.success is True
        assert response.message == 'Available'

    async def test_dispatch_pack_task_success(self, robot_coordinator: RobotCoordinator):
        """Packee 포장 작업 시작 성공"""
        # Arrange
        request = PackeePackingStart.Request(robot_id=2, order_id=100)
        mock_response = PackeePackingStart.Response(success=True)

        with patch.object(robot_coordinator._packee_start_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._packee_start_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            mock_call.return_value.set_result(mock_response)

            # Act
            response = await robot_coordinator.dispatch_pack_task(request)

        # Assert
        assert response.success is True

    async def test_service_call_timeout(self, robot_coordinator: RobotCoordinator):
        """서비스 호출 타임아웃 처리"""
        # Arrange
        request = PickeeWorkflowStartTask.Request(robot_id=1, order_id=100, user_id='test', product_list=[])

        with patch.object(robot_coordinator._pickee_start_cli, 'wait_for_service', return_value=True), \
             patch.object(robot_coordinator._pickee_start_cli, 'call_async', return_value=asyncio.Future()) as mock_call:
            # 타임아웃 시뮬레이션
            future = asyncio.Future()
            mock_call.return_value = future
            # future는 완료되지 않음

            # Act & Assert
            with pytest.raises(asyncio.TimeoutError):
                await robot_coordinator.dispatch_pick_task(request)

    async def test_service_unavailable(self, robot_coordinator: RobotCoordinator):
        """서비스 사용 불가 처리"""
        # Arrange
        request = PickeeWorkflowStartTask.Request(robot_id=1, order_id=100, user_id='test', product_list=[])

        with patch.object(robot_coordinator._pickee_start_cli, 'wait_for_service', return_value=False):
            # Act & Assert
            with pytest.raises(RuntimeError, match='unavailable'):
                await robot_coordinator.dispatch_pick_task(request)


class TestRobotCoordinatorTopicCallbacks:
    """ROS2 토픽 구독 콜백 테스트"""

    def test_on_pickee_status_updates_state_store(
        self, robot_coordinator: RobotCoordinator, mock_state_store: MagicMock
    ):
        """Pickee 상태 토픽 수신 시 상태 스토어 업데이트"""
        # Arrange
        loop = asyncio.new_event_loop()
        robot_coordinator.set_asyncio_loop(loop)

        msg = PickeeRobotStatus(
            robot_id=1,
            state='IDLE',
            battery_level=85.5,
            current_order_id=0
        )

        # Act
        robot_coordinator._on_pickee_status(msg)

        # asyncio 태스크 실행
        loop.run_until_complete(asyncio.sleep(0.01))
        loop.close()

        # Assert
        mock_state_store.upsert_state.assert_called_once()
        call_args = mock_state_store.upsert_state.call_args[0][0]
        assert call_args.robot_id == 1
        assert call_args.robot_type == RobotType.PICKEE
        assert call_args.status == RobotStatus.IDLE.value
        assert call_args.battery_level == 85.5

    def test_on_pickee_status_detects_error(
        self, robot_coordinator: RobotCoordinator, mock_event_bus: MagicMock
    ):
        """Pickee ERROR 상태 감지 시 이벤트 발행"""
        # Arrange
        loop = asyncio.new_event_loop()
        robot_coordinator.set_asyncio_loop(loop)

        msg = PickeeRobotStatus(
            robot_id=1,
            state='ERROR',
            battery_level=50.0,
            current_order_id=100
        )

        # Act
        robot_coordinator._on_pickee_status(msg)

        # asyncio 태스크 실행
        loop.run_until_complete(asyncio.sleep(0.01))
        loop.close()

        # Assert
        mock_event_bus.publish.assert_called()
        call_args = mock_event_bus.publish.call_args[0]
        assert call_args[0] == 'robot_failure'
        assert call_args[1]['robot_id'] == 1
        assert call_args[1]['robot_type'] == RobotType.PICKEE.value
        assert call_args[1]['status'] == RobotStatus.ERROR.value

    def test_on_pickee_move_calls_callback(self, robot_coordinator: RobotCoordinator):
        """Pickee 이동 상태 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_move_cb=callback_mock)

        msg = PickeeMoveStatus(robot_id=1, order_id=100, location_id=5)

        # Act
        robot_coordinator._on_pickee_move(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_pickee_arrival_calls_callback(self, robot_coordinator: RobotCoordinator):
        """Pickee 도착 알림 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_arrival_cb=callback_mock)

        msg = PickeeArrival(robot_id=1, order_id=100, location_id=5, section_id=10)

        # Act
        robot_coordinator._on_pickee_arrival(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_pickee_handover_calls_callback(self, robot_coordinator: RobotCoordinator):
        """Pickee 장바구니 전달 완료 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_handover_cb=callback_mock)

        msg = PickeeCartHandover(order_id=100, robot_id=1)

        # Act
        robot_coordinator._on_pickee_handover(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_product_detected_calls_callback(self, robot_coordinator: RobotCoordinator):
        """상품 인식 완료 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_product_detected_cb=callback_mock)

        msg = PickeeProductDetection(robot_id=1, order_id=100, product_ids=[42, 43], bbox_numbers=[1, 2])

        # Act
        robot_coordinator._on_product_detected(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_pickee_selection_calls_callback(self, robot_coordinator: RobotCoordinator):
        """상품 선택 결과 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_selection_cb=callback_mock)

        msg = PickeeProductSelection(
            robot_id=1,
            order_id=100,
            product_id=42,
            bbox_number=1,
            success=True
        )

        # Act
        robot_coordinator._on_pickee_selection(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_product_loaded_calls_callback(self, robot_coordinator: RobotCoordinator):
        """창고 물품 적재 완료 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(pickee_product_loaded_cb=callback_mock)

        msg = PickeeProductLoaded(robot_id=1, product_id=42, success=True, message='Loaded')

        # Act
        robot_coordinator._on_product_loaded(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_packee_status_updates_state_store(
        self, robot_coordinator: RobotCoordinator, mock_state_store: MagicMock
    ):
        """Packee 상태 토픽 수신 시 상태 스토어 업데이트"""
        # Arrange
        loop = asyncio.new_event_loop()
        robot_coordinator.set_asyncio_loop(loop)

        msg = PackeeRobotStatus(
            robot_id=2,
            state='BUSY',
            current_order_id=100
        )

        # Act
        robot_coordinator._on_packee_status(msg)

        # asyncio 태스크 실행
        loop.run_until_complete(asyncio.sleep(0.01))
        loop.close()

        # Assert
        mock_state_store.upsert_state.assert_called_once()
        call_args = mock_state_store.upsert_state.call_args[0][0]
        assert call_args.robot_id == 2
        assert call_args.robot_type == RobotType.PACKEE

    def test_on_packee_availability_calls_callback(self, robot_coordinator: RobotCoordinator):
        """Packee 가용성 결과 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(packee_availability_cb=callback_mock)

        msg = PackeeAvailability(available=True, robot_id=2, message='Available')

        # Act
        robot_coordinator._on_packee_availability(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)

    def test_on_packee_complete_calls_callback(self, robot_coordinator: RobotCoordinator):
        """Packee 포장 완료 콜백 호출"""
        # Arrange
        callback_mock = MagicMock()
        robot_coordinator.set_status_callbacks(packee_complete_cb=callback_mock)

        msg = PackeePackingComplete(order_id=100, robot_id=2, success=True, message='Done')

        # Act
        robot_coordinator._on_packee_complete(msg)

        # Assert
        callback_mock.assert_called_once_with(msg)


class TestRobotHealthMonitor:
    """로봇 헬스 모니터링 테스트"""

    async def test_health_timeout_marks_offline(
        self, mock_state_store: MagicMock, mock_event_bus: MagicMock
    ):
        """헬스 타임아웃 시 로봇을 OFFLINE으로 마킹"""
        # Arrange
        with patch('shopee_main_service.robot_coordinator.settings') as mock_settings:
            mock_settings.ROS_SERVICE_RETRY_ATTEMPTS = 1
            mock_settings.ROS_SERVICE_RETRY_BASE_DELAY = 0.0
            mock_settings.ROS_SERVICE_TIMEOUT = 1.0
            mock_settings.ROS_STATUS_HEALTH_TIMEOUT = 0.1  # 100ms 타임아웃

            coordinator = RobotCoordinator(state_store=mock_state_store, event_bus=mock_event_bus)
            # 현재 실행 중인 이벤트 루프 사용
            coordinator.set_asyncio_loop(asyncio.get_running_loop())

            # 초기 상태 기록
            msg = PickeeRobotStatus(robot_id=1, state='IDLE', battery_level=90.0, current_order_id=0)
            coordinator._on_pickee_status(msg)

            # 타임아웃 대기
            await asyncio.sleep(0.15)

            # 헬스 체크 수동 실행
            coordinator._check_robot_health()

            # asyncio 태스크 실행을 위한 대기
            await asyncio.sleep(0.05)

        # Assert
        mock_state_store.mark_offline.assert_called_once_with(1)
        mock_event_bus.publish.assert_called()


class TestRobotCoordinatorStateNormalization:
    """로봇 상태 정규화 테스트"""

    def test_normalize_status_idle(self, robot_coordinator: RobotCoordinator):
        """IDLE 상태 정규화"""
        assert robot_coordinator._normalize_status('idle') == RobotStatus.IDLE.value
        assert robot_coordinator._normalize_status('IDLE') == RobotStatus.IDLE.value
        assert robot_coordinator._normalize_status('  idle  ') == RobotStatus.IDLE.value

    def test_normalize_status_working(self, robot_coordinator: RobotCoordinator):
        """WORKING 상태 정규화"""
        assert robot_coordinator._normalize_status('working') == RobotStatus.WORKING.value
        assert robot_coordinator._normalize_status('WORKING') == RobotStatus.WORKING.value

    def test_normalize_status_error(self, robot_coordinator: RobotCoordinator):
        """ERROR 상태 정규화"""
        assert robot_coordinator._normalize_status('error') == RobotStatus.ERROR.value
        assert robot_coordinator._normalize_status('ERROR') == RobotStatus.ERROR.value

    def test_normalize_status_offline(self, robot_coordinator: RobotCoordinator):
        """OFFLINE 상태 정규화"""
        assert robot_coordinator._normalize_status('offline') == RobotStatus.OFFLINE.value
        assert robot_coordinator._normalize_status('OFFLINE') == RobotStatus.OFFLINE.value

    def test_normalize_status_default(self, robot_coordinator: RobotCoordinator):
        """알 수 없는 상태는 기본값(IDLE) 반환"""
        assert robot_coordinator._normalize_status('unknown') == RobotStatus.IDLE.value
        assert robot_coordinator._normalize_status('') == RobotStatus.IDLE.value
