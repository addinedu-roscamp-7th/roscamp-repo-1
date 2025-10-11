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
    mock_robot = AsyncMock()
    mock_streamer = AsyncMock()
    mock_db = MagicMock()
    mock_llm = AsyncMock()
    mock_bus = AsyncMock()

    # Inject mocks into the constructor
    app_instance = MainServiceApp(
        db=mock_db,
        llm=mock_llm,
        robot=mock_robot,
        event_bus=mock_bus,
        streaming_service=mock_streamer
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
