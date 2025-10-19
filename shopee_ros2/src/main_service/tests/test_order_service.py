"""
Unit tests for the OrderService.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shopee_interfaces.msg import PickeeArrival, PickeeCartHandover, PickeeMoveStatus, PackeePackingComplete
from shopee_interfaces.srv import PickeeProductDetect, PickeeWorkflowStartTask

from main_service.database_models import Customer, Order, OrderItem, Product, Section, Shelf, Location
from main_service.order_service import OrderService

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


def find_push_payload(mock_event_bus: AsyncMock, message_type: str):
    """Utility to find a recorded app_push payload by type."""
    for args in mock_event_bus.publish.await_args_list:
        if args.args and args.args[0] == "app_push":
            payload = args.args[1]
            if payload.get("type") == message_type:
                return payload
    return None


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Fixture to create a mock DatabaseManager."""
    db_manager = MagicMock()
    # We need to mock the context manager enter/exit
    mock_session = MagicMock()
    db_manager.session_scope.return_value.__enter__.return_value = mock_session
    db_manager.session_scope.return_value.__exit__.return_value = None
    return db_manager


@pytest.fixture
def mock_robot_coordinator() -> MagicMock:
    """Fixture to create a mock RobotCoordinator."""
    robot = MagicMock()
    robot.dispatch_pick_task = AsyncMock()
    robot.dispatch_move_to_section = AsyncMock()
    robot.dispatch_pick_process = AsyncMock()
    robot.dispatch_shopping_end = AsyncMock()
    robot.dispatch_move_to_packaging = AsyncMock()
    robot.dispatch_product_detect = AsyncMock()
    robot.check_packee_availability = AsyncMock()
    robot.dispatch_pack_task = AsyncMock()
    robot.dispatch_return_to_base = AsyncMock()
    return robot


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Fixture to create a mock EventBus."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus

@pytest.fixture
def mock_allocator() -> MagicMock:
    """Fixture to create a mock RobotAllocator."""
    allocator = MagicMock()
    allocator.reserve_robot = AsyncMock()
    allocator.release_robot = AsyncMock()
    allocator.reserve_robot.return_value = MagicMock(robot_id=1)
    return allocator


@pytest.fixture
def order_service(
    mock_db_manager: MagicMock, 
    mock_robot_coordinator: AsyncMock, 
    mock_event_bus: AsyncMock,
    mock_allocator: MagicMock,
) -> OrderService:
    """Fixture to create an OrderService instance with mock dependencies."""
    return OrderService(
        db=mock_db_manager, 
        robot_coordinator=mock_robot_coordinator, 
        event_bus=mock_event_bus,
        allocator=mock_allocator,
    )


class TestOrderServiceCreateOrder:
    """Test suite for the OrderService.create_order method."""

    def _setup_mocks(self, mock_session):
        """Helper to set up common mock objects."""
        mock_customer = Customer(customer_id=1, id="testuser")
        mock_location = Location(location_id=10)
        mock_shelf = Shelf(shelf_id=100, location_id=10, location=mock_location)
        mock_section = Section(section_id=1000, shelf_id=100, shelf=mock_shelf)
        mock_product = Product(
            product_id=42, name="Test Product", section=mock_section, 
            section_id=1000, warehouse_id=1
        )

        def query_side_effect(model):
            if model == Customer:
                return MagicMock(filter_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_customer))))
            if model == Product:
                return MagicMock(filter_by=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_product))))
            return MagicMock()
        
        mock_session.query.side_effect = query_side_effect
        
        # Mock the flush behavior to assign an ID to new_order
        def flush_side_effect():
            mock_session.new_order.order_id = 99
        mock_session.flush = flush_side_effect

        # Store the order object on the session to be accessible by flush_side_effect
        def add_side_effect(instance):
            if isinstance(instance, Order):
                mock_session.new_order = instance
        mock_session.add.side_effect = add_side_effect


    async def test_create_order_success(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
        mock_robot_coordinator: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Test successful order creation and dispatch."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        self._setup_mocks(mock_session)
        
        mock_robot_coordinator.dispatch_pick_task.return_value = MagicMock(success=True)
        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=5)
        
        user_id = "testuser"
        items = [{"product_id": 42, "quantity": 2}]

        # Act
        result = await order_service.create_order(user_id, items)

        # Assert
        assert result is not None
        order_id, robot_id = result
        assert order_id == 99
        assert robot_id == 5
        mock_robot_coordinator.dispatch_pick_task.assert_awaited_once()
        mock_session.commit.assert_called_once()
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_allocator.release_robot.assert_not_awaited()
        assert order_service._assignment_manager.get_pickee(99) == 5

    async def test_create_order_user_not_found(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
        mock_robot_coordinator: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Test order creation failure when the user is not found."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None # Simulate user not found

        # Act
        result = await order_service.create_order("unknownuser", [])

        # Assert
        assert result is None
        mock_robot_coordinator.dispatch_pick_task.assert_not_awaited()
        mock_session.rollback.assert_called_once()
        mock_allocator.reserve_robot.assert_not_awaited()

    async def test_create_order_robot_dispatch_fails(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
        mock_robot_coordinator: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Test order creation failure when robot dispatch fails."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        self._setup_mocks(mock_session)

        # Simulate robot dispatch failure
        mock_robot_coordinator.dispatch_pick_task.return_value = MagicMock(success=False, message="Robot busy")
        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=6)

        user_id = "testuser"
        items = [{"product_id": 42, "quantity": 2}]

        # Act
        result = await order_service.create_order(user_id, items)

        # Assert
        assert result is None
        mock_robot_coordinator.dispatch_pick_task.assert_awaited_once()
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_allocator.release_robot.assert_awaited_once()


class TestHandleMovingStatus:
    """Test suite for the OrderService.handle_moving_status method."""

    async def test_emits_work_info_notification(
        self,
        order_service: OrderService,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Robot moving 이벤트 시 관리자 알림이 발행되는지 확인."""
        move_msg = PickeeMoveStatus(robot_id=2, order_id=77, location_id=5)
        order_service._notifier.register_order_user(77, "user77")

        await order_service.handle_moving_status(move_msg)

        payload = find_push_payload(mock_event_bus, "robot_moving_notification")
        assert payload is not None
        assert payload["result"] is True
        assert payload["error_code"] is None
        assert payload["data"]["order_id"] == 77
        assert payload["data"]["robot_id"] == 2
        assert payload["data"]["destination"] == "LOCATION_5"
        assert payload["data"]["location_id"] == 5
        assert payload["target_user_id"] == "user77"

class TestHandleArrivalNotice:
    """Test suite for the OrderService.handle_arrival_notice method."""

    async def test_sends_notification_and_dispatches_detection(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock, mock_event_bus: AsyncMock):
        """Verify that on arrival, a notification is sent and product detection is dispatched."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.all.return_value = [OrderItem(product_id=101)]
        
        arrival_msg = PickeeArrival(robot_id=1, order_id=99, location_id=5, section_id=10)
        order_service._notifier.register_order_user(99, "user99")

        # Act
        await order_service.handle_arrival_notice(arrival_msg)

        # Assert
        payload = find_push_payload(mock_event_bus, "robot_arrived_notification")
        assert payload is not None
        assert payload["data"]["order_id"] == 99
        assert payload["data"]["robot_id"] == 1
        assert payload["data"]["location_id"] == 5
        assert payload["data"]["section_id"] == 10
        assert payload["target_user_id"] == "user99"
        
        # 2. Check that product detection was dispatched
        mock_robot_coordinator.dispatch_product_detect.assert_awaited_once()
        call_args = mock_robot_coordinator.dispatch_product_detect.await_args[0][0]
        assert isinstance(call_args, PickeeProductDetect.Request)
        assert list(call_args.product_ids) == [101]

    async def test_skips_detection_for_non_section_location(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
        mock_robot_coordinator: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """섹션이 아닌 위치에서는 상품 인식을 수행하지 않는다."""
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.all.return_value = [OrderItem(product_id=202)]

        arrival_msg = PickeeArrival(robot_id=3, order_id=55, location_id=900, section_id=-1)
        order_service._notifier.register_order_user(55, "user55")

        await order_service.handle_arrival_notice(arrival_msg)

        payload = find_push_payload(mock_event_bus, "robot_arrived_notification")
        assert payload is not None
        assert payload["data"]["section_id"] == -1
        assert payload["message"] == "목적지에 도착했습니다"
        mock_robot_coordinator.dispatch_product_detect.assert_not_awaited()


class TestHandleCartHandover:
    """Test suite for the OrderService.handle_cart_handover method."""

    async def test_dispatches_packing_when_packee_available(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
        mock_allocator: MagicMock,
        mock_db_manager: MagicMock,
    ):
        """Verify that packing is dispatched if a Packee robot is available."""
        # Arrange
        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=5)
        mock_robot_coordinator.dispatch_pack_task.return_value = MagicMock(success=True, box_id=123)
        mock_robot_coordinator.dispatch_return_to_base.return_value = MagicMock(success=True)
        order_service._assignment_manager.assign_pickee(99, 1)

        # Mock database session
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        with patch.object(order_service._notifier, 'notify_packing_info', new_callable=AsyncMock) as mock_notify_packing, patch('main_service.order_service.settings') as mock_settings:
            mock_settings.PICKEE_HOME_LOCATION_ID = 10  # Set to non-zero value
            await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_robot_coordinator.dispatch_pack_task.assert_awaited_once()
        mock_robot_coordinator.dispatch_return_to_base.assert_awaited_once()
        # notify_packing_info가 호출되었는지 직접 확인
        mock_notify_packing.assert_awaited_once()
        # packee 할당이 해제되었는지 확인 (release_pickee가 호출되었는지 확인)
        # 이 부분은 failure_handler 테스트에서 더 상세히 다루는 것이 좋음
        # 여기서는 간단히 로직의 주요 흐름만 확인

    async def test_does_not_dispatch_when_packee_unavailable(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Verify that packing is not dispatched if no Packee robot is available."""
        # Arrange
        mock_allocator.reserve_robot.return_value = None # 가용한 로봇 없음
        order_service._assignment_manager.assign_pickee(99, 1)
        
        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        # _failure_handler.fail_order를 patch하여 호출 여부 확인
        with patch.object(order_service._failure_handler, 'fail_order', new_callable=AsyncMock) as mock_fail_order:
            await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_robot_coordinator.dispatch_pack_task.assert_not_awaited()
        # fail_order가 올바른 인자와 함께 호출되었는지 확인
        mock_fail_order.assert_awaited_once_with(99, "No available Packee robot.")


class TestHandlePackeeComplete:
    """Test suite for the OrderService.handle_packee_complete method."""

    async def test_releases_packee_on_completion(
        self,
        order_service: OrderService,
        mock_event_bus: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Packee 완료 시 예약 해제가 호출되는지 확인."""
        order_service._assignment_manager.assign_packee(123, 7)  # simulate reserved packee

        msg = PackeePackingComplete(order_id=123, robot_id=7, success=True, message="done")

        await order_service.handle_packee_complete(msg)

        mock_event_bus.publish.assert_awaited()
        mock_allocator.release_robot.assert_any_await(7, 123)
        assert mock_allocator.release_robot.await_count == 1
        assert order_service._assignment_manager.get_packee(123) is None



class TestOrderServiceDashboardHelpers:
    """대시보드용 헬퍼 메서드 테스트 모음."""

    async def test_get_active_orders_snapshot_basic(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
    ):
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value

        order = MagicMock()
        order.order_id = 101
        order.order_status = 2
        order.customer = MagicMock(id="customer001")
        order.start_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        query_mock = MagicMock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.all.return_value = [order]
        mock_session.query.return_value = query_mock

        order_service._assignment_manager.assign_pickee(101, 5)
        order_service._detected_product_bbox[101] = {77: 1}

        original_calc = order_service._calculate_order_summary
        order_service._calculate_order_summary = MagicMock(return_value=(3, 12900))

        snapshot = await order_service.get_active_orders_snapshot()

        order_service._calculate_order_summary = original_calc

        assert snapshot["summary"]["total_active"] == 1
        first = snapshot["orders"][0]
        assert first["order_id"] == 101
        assert first["pickee_robot_id"] == 5

    async def test_get_recent_failed_orders(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
    ):
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value

        failed_order = MagicMock()
        failed_order.order_id = 202
        failed_order.order_status = 9
        failed_order.failure_reason = "robot timeout"
        failed_order.end_time = datetime.now(timezone.utc)

        query_mock = MagicMock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = [failed_order]
        mock_session.query.return_value = query_mock

        original_calc = order_service._calculate_order_summary
        order_service._calculate_order_summary = MagicMock(return_value=(2, 5000))

        payload = await order_service.get_recent_failed_orders(limit=5)

        order_service._calculate_order_summary = original_calc

        assert payload
        assert payload[0]["order_id"] == 202
        assert payload[0]["failure_reason"] == "robot timeout"
