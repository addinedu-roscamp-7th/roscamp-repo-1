"""
Tests for APIController push broadcasting.
"""

import json
from typing import Tuple

import pytest

from shopee_main_service.api_controller import APIController
from shopee_main_service.event_bus import EventBus


class FakeWriter:
    """Minimal StreamWriter stub for testing."""

    def __init__(self, peer: Tuple[str, int] = ("127.0.0.1", 5000)) -> None:
        self.buffer = []
        self._peer = peer
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


@pytest.mark.asyncio
async def test_handle_push_event_targets_user() -> None:
    event_bus = EventBus()
    controller = APIController("0.0.0.0", 0, {}, event_bus)

    writer1 = FakeWriter(peer=("127.0.0.1", 5001))
    writer2 = FakeWriter(peer=("127.0.0.1", 5002))
    controller._register_client(writer1, writer1.get_extra_info("peername"))
    controller._register_client(writer2, writer2.get_extra_info("peername"))

    controller.associate_peer_with_user(writer1.get_extra_info("peername"), "user1")
    controller.associate_peer_with_user(writer2.get_extra_info("peername"), "user2")

    message = {
        "type": "robot_moving_notification",
        "result": True,
        "data": {"order_id": 1},
        "target_user_id": "user1",
    }
    await controller._handle_push_event(message)

    assert len(writer1.buffer) == 1
    assert writer2.buffer == []
    payload = json.loads(writer1.buffer[0].decode().rstrip("\n"))
    assert payload == message
