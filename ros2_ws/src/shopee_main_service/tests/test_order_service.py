"""
Unit tests for the OrderService.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from shopee_interfaces.msg import PickeeArrival, PickeeCartHandover, PickeeMoveStatus, PackeePackingComplete
from shopee_interfaces.srv import PickeeProductDetect, PickeeWorkflowStartTask

from shopee_main_service.database_models import Customer, Order, OrderItem, Product, Section, Shelf, Location
from shopee_main_service.order_service import OrderService

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
        assert order_service._pickee_assignments[99] == 5

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
        order_service._order_user_map[77] = "user77"

        with patch.object(order_service, "_emit_work_info_notification", new=AsyncMock()) as mock_emit:
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
        mock_emit.assert_awaited_once_with(order_id=77, robot_id=2, destination="LOCATION_5")

class TestHandleArrivalNotice:
    """Test suite for the OrderService.handle_arrival_notice method."""

    async def test_sends_notification_and_dispatches_detection(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock, mock_event_bus: AsyncMock):
        """Verify that on arrival, a notification is sent and product detection is dispatched."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.all.return_value = [OrderItem(product_id=101)]
        
        arrival_msg = PickeeArrival(robot_id=1, order_id=99, location_id=5, section_id=10)
        order_service._order_user_map[99] = "user99"

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


class TestHandleCartHandover:
    """Test suite for the OrderService.handle_cart_handover method."""

    async def test_dispatches_packing_when_packee_available(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_allocator: MagicMock,
        mock_db_manager: MagicMock,
    ):
        """Verify that packing is dispatched if a Packee robot is available."""
        # Arrange
        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=5)
        mock_robot_coordinator.dispatch_pack_task.return_value = MagicMock(success=True)
        mock_robot_coordinator.dispatch_return_to_base.return_value = MagicMock(success=True)
        order_service._pickee_assignments[99] = 1

        # Mock database session
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)
        order_service._order_user_map[99] = "user99"

        # Act
        with patch('shopee_main_service.order_service.settings') as mock_settings:
            mock_settings.PICKEE_HOME_LOCATION_ID = 10  # Set to non-zero value
            await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_robot_coordinator.check_packee_availability.assert_not_awaited()
        mock_robot_coordinator.dispatch_pack_task.assert_awaited_once()
        mock_robot_coordinator.dispatch_return_to_base.assert_awaited_once()
        payload = find_push_payload(mock_event_bus, "packing_info_notification")
        assert payload is not None
        assert payload["result"] is True
        assert payload["data"]["order_id"] == 99
        assert payload["message"] == "포장 정보 업데이트"
        assert payload["target_user_id"] == "user99"
        call_args = mock_robot_coordinator.dispatch_pack_task.await_args[0][0]
        assert call_args.robot_id == 5
        assert call_args.order_id == 99
        mock_allocator.release_robot.assert_any_await(1, 99)
        assert mock_allocator.release_robot.await_count == 1

    async def test_does_not_dispatch_when_packee_unavailable(
        self,
        order_service: OrderService,
        mock_robot_coordinator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Verify that packing is not dispatched if no Packee robot is available."""
        # Arrange
        mock_allocator.reserve_robot.return_value = None
        mock_robot_coordinator.check_packee_availability.return_value = MagicMock(available=False, message="All busy")
        order_service._pickee_assignments[99] = 1
        
        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_allocator.reserve_robot.assert_awaited_once()
        mock_robot_coordinator.check_packee_availability.assert_awaited_once()
        mock_robot_coordinator.dispatch_pack_task.assert_not_awaited()
        mock_event_bus.publish.assert_not_called()
        mock_allocator.release_robot.assert_not_awaited()


class TestHandlePackeeComplete:
    """Test suite for the OrderService.handle_packee_complete method."""

    async def test_releases_packee_on_completion(
        self,
        order_service: OrderService,
        mock_event_bus: AsyncMock,
        mock_allocator: MagicMock,
    ):
        """Packee 완료 시 예약 해제가 호출되는지 확인."""
        order_service._packee_assignments[123] = 7  # simulate reserved packee

        msg = PackeePackingComplete(order_id=123, robot_id=7, success=True, message="done")

        await order_service.handle_packee_complete(msg)

        mock_event_bus.publish.assert_awaited()
        mock_allocator.release_robot.assert_any_await(7, 123)
        assert mock_allocator.release_robot.await_count == 1
        assert 123 not in order_service._packee_assignments


class TestHandleRobotFailure:
    """Tests for automatic robot reassignment."""

    async def test_reassign_pickee_success(
        self,
        order_service: OrderService,
        mock_allocator: MagicMock,
        mock_event_bus: AsyncMock,
    ):
        order_service._allocator = mock_allocator
        order_service._pickee_assignments[555] = 4
        monitor_mock = MagicMock()
        monitor_mock.cancel = MagicMock()
        order_service._reservation_monitors[(555, 4)] = monitor_mock

        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=9)

        async def fake_reassign(order_id: int, robot_id: int) -> bool:
            order_service._pickee_assignments[order_id] = robot_id
            return True

        with patch.object(order_service, "_reassign_pickee", side_effect=fake_reassign) as mock_reassign:
            await order_service.handle_robot_failure(
                {
                    "robot_id": 4,
                    "robot_type": "pickee",
                    "status": "ERROR",
                    "active_order_id": 555,
                }
            )

        mock_reassign.assert_awaited_once_with(555, 9)
        monitor_mock.cancel.assert_called_once()
        mock_allocator.release_robot.assert_not_awaited()
        assert order_service._pickee_assignments[555] == 9
        success_calls = [
            args for args in mock_event_bus.publish.await_args_list
            if args.args and args.args[0] == "app_push"
            and isinstance(args.args[1], dict)
            and args.args[1].get("type") == "robot_reassignment_notification"
        ]
        assert success_calls

    async def test_reassign_pickee_failure(
        self,
        order_service: OrderService,
        mock_allocator: MagicMock,
        mock_event_bus: AsyncMock,
    ):
        order_service._allocator = mock_allocator
        order_service._pickee_assignments[555] = 4

        mock_allocator.reserve_robot.return_value = MagicMock(robot_id=9)

        async def fake_reassign(order_id: int, robot_id: int) -> bool:
            return False

        with patch.object(order_service, "_reassign_pickee", side_effect=fake_reassign):
            await order_service.handle_robot_failure(
                {
                    "robot_id": 4,
                    "robot_type": "pickee",
                    "status": "ERROR",
                    "active_order_id": 555,
                }
            )

        mock_allocator.release_robot.assert_any_await(9, 555)
        failure_calls = [
            args for args in mock_event_bus.publish.await_args_list
            if args.args and args.args[0] == "app_push"
            and isinstance(args.args[1], dict)
            and args.args[1].get("type") == "robot_failure_notification"
        ]
        assert failure_calls

    async def test_reassign_without_allocator_sends_failure(
        self,
        order_service: OrderService,
        mock_event_bus: AsyncMock,
    ):
        order_service._allocator = None
        order_service._pickee_assignments[555] = 4

        await order_service.handle_robot_failure(
            {
                "robot_id": 4,
                "robot_type": "pickee",
                "status": "ERROR",
                "active_order_id": 555,
            }
        )

        failure_calls = [
            args for args in mock_event_bus.publish.await_args_list
            if args.args and args.args[0] == "app_push"
            and isinstance(args.args[1], dict)
            and args.args[1].get("type") == "robot_failure_notification"
        ]
        assert failure_calls
