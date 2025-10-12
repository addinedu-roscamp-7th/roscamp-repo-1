"""
Tests for APIController push broadcasting.
"""

import json

import pytest

from shopee_main_service.api_controller import APIController
from shopee_main_service.event_bus import EventBus


class FakeWriter:
    """Minimal StreamWriter stub for testing."""

    def __init__(self) -> None:
        self.buffer = []
        self._peer = ("127.0.0.1", 5000)
        self._closing = False

    # StreamWriter interface subset ---------------------------------
    def write(self, data: bytes) -> None:
        self.buffer.append(data)

    async def drain(self) -> None:  # noqa: D401
        return

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True

    async def wait_closed(self) -> None:
        return

    def get_extra_info(self, name: str):
        if name == "peername":
            return self._peer
        return None


@pytest.mark.asyncio
async def test_handle_push_event_broadcasts_to_clients() -> None:
    event_bus = EventBus()
    controller = APIController("0.0.0.0", 0, {}, event_bus)

    writer = FakeWriter()
    controller._register_client(writer, writer.get_extra_info("peername"))

    message = {"type": "robot_moving_notification", "result": True, "data": {"order_id": 1}}
    await controller._handle_push_event(message)

    assert len(writer.buffer) == 1
    payload = writer.buffer[0].decode().rstrip("\n")
    assert json.loads(payload) == message
