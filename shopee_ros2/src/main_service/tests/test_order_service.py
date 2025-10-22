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
from main_service.config import settings

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
        assert payload["error_code"] == ""  # 성공 시 빈 문자열
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
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.filter.return_value.all.return_value = [(101,)] # Query returns tuples of product_ids
        
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
        mock_robot_coordinator.check_packee_availability.return_value = MagicMock(success=True, message="OK")
        mock_robot_coordinator.dispatch_return_to_base.return_value = MagicMock(success=True)
        order_service._assignment_manager.assign_pickee(99, 1)

        # Mock database session - DB 쿼리 결과 모킹
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_order_item = MagicMock(product_id=1, quantity=2)
        mock_product = MagicMock(
            product_id=1,
            length=10,
            width=20,
            height=30,
            weight=100,
            fragile=True
        )
        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = [
            (mock_order_item, mock_product)
        ]

        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        with patch.object(order_service._notifier, 'notify_packing_info', new_callable=AsyncMock) as mock_notify_packing, patch('main_service.order_service.settings') as mock_settings:
            mock_settings.PICKEE_HOME_LOCATION_ID = 10  # Set to non-zero value
            await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_robot_coordinator.check_packee_availability.assert_awaited_once()
        mock_robot_coordinator.dispatch_return_to_base.assert_awaited_once()
        # notify_packing_info가 호출되었는지 직접 확인
        mock_notify_packing.assert_awaited_once()
        # packee 할당이 되었는지 확인
        assert order_service._assignment_manager.get_packee(99) == 5

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
        # reserve_robot이 3번 재시도되므로 최대 3번 호출됨
        assert mock_allocator.reserve_robot.await_count == 3
        mock_robot_coordinator.check_packee_availability.assert_not_awaited()
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


class TestOrderServicePickingModes:
    """
    새로운 수동/자동 피킹 모드 로직에 대한 테스트 스위트.
    """

    async def test_create_order_prioritizes_manual_sections(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
        mock_robot_coordinator: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """
        create_order가 수동 섹션을 자동 섹션보다 우선하여 큐를 생성하는지 테스트.
        """
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value

        # 1. Mock DB Models & Queries
        mock_customer = Customer(customer_id=1, id="testuser")
        manual_prod = Product(product_id=1, name="Manual Pick", auto_select=False)
        auto_prod = Product(product_id=2, name="Auto Pick", auto_select=True)

        mock_customer_query = MagicMock()
        mock_customer_query.filter_by.return_value.first.return_value = mock_customer

        mock_product_query = MagicMock()
        mock_product_query.filter.return_value.all.return_value = [manual_prod, auto_prod]

        def query_side_effect(model):
            if model == Customer:
                return mock_customer_query
            if model == Product:
                return mock_product_query
            return MagicMock()
        mock_session.query.side_effect = query_side_effect

        # Mock flush/add to get an order_id
        def flush_side_effect():
            if hasattr(mock_session, 'new_order'):
                mock_session.new_order.order_id = 99
        mock_session.flush = flush_side_effect
        def add_side_effect(instance):
            if isinstance(instance, Order):
                mock_session.new_order = instance
        mock_session.add.side_effect = add_side_effect

        # 2. Mock ProductLocationBuilder result
        from shopee_interfaces.msg import ProductLocation
        product_locations = [
            ProductLocation(product_id=2, location_id=20, section_id=200), # Auto - intentionally put first
            ProductLocation(product_id=1, location_id=10, section_id=100), # Manual
        ]
        order_service._product_builder = MagicMock()
        order_service._product_builder.build_product_locations.return_value = product_locations

        # 3. Mock Robot Coordinator and Allocator
        mock_robot_coordinator.dispatch_pick_task.return_value = MagicMock(success=True)
        mock_robot_coordinator.dispatch_move_to_section.return_value = MagicMock(success=True)
        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=5)

        user_id = "testuser"
        items = [{"product_id": 1, "quantity": 1}, {"product_id": 2, "quantity": 1}]

        # Act
        await order_service.create_order(user_id, items)

        # Assert
        queue = order_service._order_section_queue.get(99)
        assert queue is not None, "Section queue was not created"
        assert len(queue) == 2, "Queue should have two sections"

        # Manual section (100) should be first
        assert queue[0]["section_id"] == 100
        assert queue[0]["is_manual"] is True
        
        # Auto section (200) should be second
        assert queue[1]["section_id"] == 200
        assert queue[1]["is_manual"] is False

        # Check that the first move command goes to the manual section
        mock_robot_coordinator.dispatch_move_to_section.assert_awaited_once()


class TestOrderServiceEndShopping:
    """Test suite for OrderService.end_shopping."""

    async def test_end_shopping_resolves_robot_assignment(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
        mock_db_manager: MagicMock,
    ) -> None:
        order_id = 501
        robot_id = 42

        order_service._assignment_manager.assign_pickee(order_id, robot_id)

        mock_robot_coordinator.dispatch_shopping_end.return_value = MagicMock(success=True, message="")
        mock_robot_coordinator.dispatch_move_to_packaging.return_value = MagicMock(success=True)
        order_service._state_manager.set_status_picked_up = MagicMock(return_value=True)

        with patch.object(order_service, "_calculate_order_summary", return_value=(2, 3000)):
            with patch.object(settings, "PICKEE_PACKING_LOCATION_ID", 99):
                success, summary = await order_service.end_shopping(order_id)

        assert success is True
        assert summary == {"total_items": 2, "total_price": 3000}
        mock_robot_coordinator.dispatch_shopping_end.assert_awaited_once()
        sent_request = mock_robot_coordinator.dispatch_shopping_end.await_args[0][0]
        assert sent_request.robot_id == robot_id
        # 포장 위치로 이동 명령이 호출되었는지 확인
        mock_robot_coordinator.dispatch_move_to_packaging.assert_awaited_once()

    async def test_handle_product_detected_triggers_auto_pick(
        self,
        order_service: OrderService,
    ):
        """
        handle_product_detected가 자동 섹션에서 select_product를 호출하는지 테스트.
        """
        # Arrange
        from shopee_interfaces.msg import PickeeProductDetection
        order_id = 101
        robot_id = 5
        
        # Set up the service state to simulate being in an auto-pick section
        order_service._current_section_info[order_id] = {'section_id': 200, 'is_manual': False}
        
        # Mock the select_product method to spy on it
        order_service.select_product = AsyncMock()
        
        # Mock the notifier to ensure it's NOT called
        order_service._notifier.notify_product_selection_start = AsyncMock()

        # Create the incoming message
        detected_product = MagicMock()
        detected_product.product_id = 2
        detected_product.bbox_number = 1
        msg = PickeeProductDetection(order_id=order_id, robot_id=robot_id, products=[detected_product])

        # Act
        await order_service.handle_product_detected(msg)

        # Assert
        order_service.select_product.assert_awaited_once_with(
            order_id=order_id,
            robot_id=robot_id,
            product_id=2,
            bbox_number=1
        )
        order_service._notifier.notify_product_selection_start.assert_not_awaited()

    async def test_handle_product_detected_notifies_user_for_manual_pick(
        self,
        order_service: OrderService,
        mock_db_manager: MagicMock,
    ):
        """
        handle_product_detected가 수동 섹션에서 사용자에게 알림을 보내는지 테스트.
        """
        # Arrange
        from shopee_interfaces.msg import PickeeProductDetection
        order_id = 102
        robot_id = 6

        # Set up the service state to simulate being in a manual-pick section
        order_service._current_section_info[order_id] = {'section_id': 100, 'is_manual': True}
        
        # Mock the select_product method to ensure it's NOT called
        order_service.select_product = AsyncMock()
        
        # Mock the notifier to spy on it
        order_service._notifier.notify_product_selection_start = AsyncMock()

        # Mock DB for loading product name
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.all.return_value = [Product(product_id=1, name="Manual")]

        # Create the incoming message
        detected_product = MagicMock()
        detected_product.product_id = 1
        detected_product.bbox_number = 2
        msg = PickeeProductDetection(order_id=order_id, robot_id=robot_id, products=[detected_product])

        # Act
        await order_service.handle_product_detected(msg)

        # Assert
        order_service._notifier.notify_product_selection_start.assert_awaited_once()
        order_service.select_product.assert_not_awaited()

    async def test_move_to_next_or_end_notifies_on_manual_to_auto_transition(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
    ):
        """
        수동->자동 섹션 전환 시 알림이 전송되는지 테스트.
        """
        # Arrange
        order_id = 103
        robot_id = 7

        # Simulate being in a completed manual section
        order_service._current_section_info[order_id] = {'section_id': 100, 'is_manual': True}
        
        # Simulate the queue having an auto section next
        order_service._order_section_queue[order_id] = [
            {'location_id': 20, 'section_id': 200, 'is_manual': False}
        ]

        # Mock the notifier to spy on the new method
        order_service._notifier.notify_manual_picking_complete = AsyncMock()

        # Act
        await order_service._move_to_next_or_end(order_id, robot_id)

        # Assert
        order_service._notifier.notify_manual_picking_complete.assert_awaited_once_with(order_id)
        # Also assert that the robot is dispatched to the next section
        mock_robot_coordinator.dispatch_move_to_section.assert_awaited_once()
