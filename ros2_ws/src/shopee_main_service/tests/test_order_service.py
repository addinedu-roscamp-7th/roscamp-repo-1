"""
Unit tests for the OrderService.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY

from shopee_interfaces.msg import PickeeArrival, PickeeCartHandover
from shopee_interfaces.srv import PickeeProductDetect, PickeeWorkflowStartTask

from shopee_main_service.database_models import Customer, Order, OrderItem, Product, Section, Shelf, Location
from shopee_main_service.order_service import OrderService

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


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
def mock_robot_coordinator() -> AsyncMock:
    """Fixture to create a mock RobotCoordinator."""
    return AsyncMock()


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Fixture to create a mock EventBus."""
    return AsyncMock()


@pytest.fixture
def order_service(
    mock_db_manager: MagicMock, 
    mock_robot_coordinator: AsyncMock, 
    mock_event_bus: AsyncMock
) -> OrderService:
    """Fixture to create an OrderService instance with mock dependencies."""
    return OrderService(
        db=mock_db_manager, 
        robot_coordinator=mock_robot_coordinator, 
        event_bus=mock_event_bus
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


    async def test_create_order_success(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock):
        """Test successful order creation and dispatch."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        self._setup_mocks(mock_session)
        
        mock_robot_coordinator.dispatch_pick_task.return_value = MagicMock(success=True)
        
        user_id = "testuser"
        items = [{"product_id": 42, "quantity": 2}]

        # Act
        result = await order_service.create_order(user_id, items)

        # Assert
        assert result is not None
        order_id, robot_id = result
        assert order_id == 99
        assert robot_id == 1
        mock_robot_coordinator.dispatch_pick_task.assert_awaited_once()
        mock_session.commit.assert_called_once()

    async def test_create_order_user_not_found(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock):
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

    async def test_create_order_robot_dispatch_fails(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock):
        """Test order creation failure when robot dispatch fails."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        self._setup_mocks(mock_session)

        # Simulate robot dispatch failure
        mock_robot_coordinator.dispatch_pick_task.return_value = MagicMock(success=False, message="Robot busy")

        user_id = "testuser"
        items = [{"product_id": 42, "quantity": 2}]

        # Act
        result = await order_service.create_order(user_id, items)

        # Assert
        assert result is None
        mock_robot_coordinator.dispatch_pick_task.assert_awaited_once()
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()


class TestHandleArrivalNotice:
    """Test suite for the OrderService.handle_arrival_notice method."""

    async def test_sends_notification_and_dispatches_detection(self, order_service: OrderService, mock_db_manager: MagicMock, mock_robot_coordinator: AsyncMock, mock_event_bus: AsyncMock):
        """Verify that on arrival, a notification is sent and product detection is dispatched."""
        # Arrange
        mock_session = mock_db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter_by.return_value.all.return_value = [OrderItem(product_id=101)]
        
        arrival_msg = PickeeArrival(robot_id=1, order_id=99, location_id=5, section_id=10)

        # Act
        await order_service.handle_arrival_notice(arrival_msg)

        # Assert
        # 1. Check that app notification was published
        mock_event_bus.publish.assert_awaited_once_with(
            "app_push",
            {
                "type": "robot_arrived_notification",
                "data": {
                    "order_id": 99,
                    "robot_id": 1,
                    "location_id": 5,
                    "section_id": 10,
                },
            },
        )
        
        # 2. Check that product detection was dispatched
        mock_robot_coordinator.dispatch_product_detect.assert_awaited_once()
        call_args = mock_robot_coordinator.dispatch_product_detect.await_args[0][0]
        assert isinstance(call_args, PickeeProductDetect.Request)
        assert list(call_args.product_ids) == [101]


class TestHandleCartHandover:
    """Test suite for the OrderService.handle_cart_handover method."""

    async def test_dispatches_packing_when_packee_available(self, order_service: OrderService, mock_robot_coordinator: AsyncMock):
        """Verify that packing is dispatched if a Packee robot is available."""
        # Arrange
        mock_robot_coordinator.check_packee_availability.return_value = MagicMock(available=True, robot_id=5)
        mock_robot_coordinator.dispatch_pack_task.return_value = MagicMock(success=True)
        
        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_robot_coordinator.check_packee_availability.assert_awaited_once()
        mock_robot_coordinator.dispatch_pack_task.assert_awaited_once()
        call_args = mock_robot_coordinator.dispatch_pack_task.await_args[0][0]
        assert call_args.robot_id == 5
        assert call_args.order_id == 99

    async def test_does_not_dispatch_when_packee_unavailable(self, order_service: OrderService, mock_robot_coordinator: AsyncMock):
        """Verify that packing is not dispatched if no Packee robot is available."""
        # Arrange
        mock_robot_coordinator.check_packee_availability.return_value = MagicMock(available=False, message="All busy")
        
        handover_msg = PickeeCartHandover(order_id=99, robot_id=1)

        # Act
        await order_service.handle_cart_handover(handover_msg)

        # Assert
        mock_robot_coordinator.check_packee_availability.assert_awaited_once()
        mock_robot_coordinator.dispatch_pack_task.assert_not_awaited()
