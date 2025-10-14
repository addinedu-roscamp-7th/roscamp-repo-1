"""
Unit tests for the MainServiceApp API handlers.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from shopee_main_service.main_service_node import MainServiceApp
from shopee_interfaces.srv import PickeeMainVideoStreamStart, PickeeMainVideoStreamStop

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def app() -> MainServiceApp:
    """Fixture to create a MainServiceApp instance with injected mocks."""
    # Create mock objects for all dependencies
    mock_robot = MagicMock()
    mock_robot.dispatch_video_stream_start = AsyncMock()
    mock_robot.dispatch_video_stream_stop = AsyncMock()

    mock_streamer = MagicMock()
    mock_db = MagicMock()
    mock_llm = AsyncMock()

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_inventory = AsyncMock()
    mock_robot_history = AsyncMock()

    # Inject mocks into the constructor
    app_instance = MainServiceApp(
        db=mock_db,
        llm=mock_llm,
        robot=mock_robot,
        event_bus=mock_bus,
        streaming_service=mock_streamer,
        inventory_service=mock_inventory,
        robot_history_service=mock_robot_history,
    )
    app_instance._install_handlers()
    return app_instance


class TestVideoStreamHandlers:
    """Test suite for video stream API handlers."""

    async def test_handle_video_stream_start(self, app: MainServiceApp):
        """Verify that the start handler calls the correct services."""
        # Arrange
        handler = app._handlers["video_stream_start"]
        
        # Configure mock return values
        app._robot.dispatch_video_stream_start.return_value = MagicMock(success=True, message="Stream started")

        request_data = {"robot_id": 1, "user_id": "admin", "user_type": "admin"}
        peer_address = ("192.168.1.100", 12345)
        APP_UDP_PORT = 6000

        # Act
        response = await handler(request_data, peer_address)

        # Assert
        app._streaming_service.start_relay.assert_called_once_with(peer_address[0], APP_UDP_PORT)
        app._robot.dispatch_video_stream_start.assert_awaited_once()
        assert response["result"] is True


class TestInventoryHandlers:
    """Test suite for inventory-related handlers."""

    async def test_inventory_search_success(self, app: MainServiceApp):
        handler = app._handlers["inventory_search"]
        app._inventory_service.search_products.return_value = ([{"product_id": 1}], 1)

        filters = {"name": "사과"}
        response = await handler(filters)

        app._inventory_service.search_products.assert_awaited_once_with(filters)
        assert response["type"] == "inventory_search_response"
        assert response["result"] is True
        assert response["data"]["total_count"] == 1

    async def test_inventory_create_failure(self, app: MainServiceApp):
        handler = app._handlers["inventory_create"]
        app._inventory_service.create_product.side_effect = ValueError("Product already exists.")

        response = await handler({"product_id": 1})

        app._inventory_service.create_product.assert_awaited_once()
        assert response["result"] is False
        assert response["error_code"] == "PROD_003"

    async def test_inventory_update_not_found(self, app: MainServiceApp):
        handler = app._handlers["inventory_update"]
        app._inventory_service.update_product.return_value = False

        response = await handler({"product_id": 999})

        app._inventory_service.update_product.assert_awaited_once()
        assert response["result"] is False
        assert response["error_code"] == "PROD_001"

    async def test_inventory_delete_success(self, app: MainServiceApp):
        handler = app._handlers["inventory_delete"]
        app._inventory_service.delete_product.return_value = True

        response = await handler({"product_id": 10})

        app._inventory_service.delete_product.assert_awaited_once_with(10)
        assert response["result"] is True
        assert response["message"] == "재고 정보를 삭제하였습니다."


class TestRobotHistoryHandlers:
    """Test suite for robot history search handler."""

    async def test_robot_history_search_success(self, app: MainServiceApp):
        handler = app._handlers["robot_history_search"]
        history_payload = [{"robot_history_id": 1}]
        app._robot_history_service.search_histories.return_value = (history_payload, 1)

        response = await handler({"robot_id": 1})

        app._robot_history_service.search_histories.assert_awaited_once_with({"robot_id": 1})
        assert response["result"] is True
        assert response["data"]["total_count"] == 1

    async def test_handle_video_stream_stop(self, app: MainServiceApp):
        """Verify that the stop handler calls the correct services."""
        # Arrange
        handler = app._handlers["video_stream_stop"]
        
        # Configure mock return values
        app._robot.dispatch_video_stream_stop.return_value = MagicMock(success=True, message="Stream stopped")

        request_data = {"robot_id": 1, "user_id": "admin", "user_type": "admin"}

        # Act
        response = await handler(request_data, None)

        # Assert
        app._streaming_service.stop_relay.assert_called_once()
        app._robot.dispatch_video_stream_stop.assert_awaited_once()
        assert response["result"] is True
