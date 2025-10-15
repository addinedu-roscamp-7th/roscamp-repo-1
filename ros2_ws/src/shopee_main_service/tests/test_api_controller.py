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


class FlakyWriter(FakeWriter):
    """전송 실패를 시뮬레이션하기 위한 StreamWriter."""

    def __init__(self, peer: Tuple[str, int] = ("127.0.0.1", 5003), fail_times: int = 1) -> None:
        super().__init__(peer=peer)
        self._fail_times = fail_times
        self.attempts = 0

    def write(self, data: bytes) -> None:
        self.attempts += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("temporary send failure")
        super().write(data)


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
    parsed = json.loads(payload)
    assert parsed["type"] == message["type"]
    assert parsed["result"] == message["result"]
    assert parsed["data"] == message["data"]
    assert "correlation_id" in parsed
    assert "timestamp" in parsed
    assert "correlation_id" not in message


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
    assert payload["type"] == message["type"]
    assert payload["result"] == message["result"]
    assert payload["data"] == message["data"]
    assert payload["target_user_id"] == message["target_user_id"]
    assert "correlation_id" in payload
    assert "timestamp" in payload


@pytest.mark.asyncio
async def test_handle_push_event_retries_when_send_fails() -> None:
    event_bus = EventBus()
    controller = APIController("0.0.0.0", 0, {}, event_bus)

    flaky_writer = FlakyWriter(fail_times=2)
    controller._register_client(flaky_writer, flaky_writer.get_extra_info("peername"))

    message = {"type": "robot_moving_notification", "result": True, "data": {"order_id": 2}}
    await controller._handle_push_event(message)

    # 3회 시도 끝에 성공해야 하며, 버퍼에는 최종 메시지가 기록된다.
    assert flaky_writer.attempts >= 3
    assert len(flaky_writer.buffer) == 1
    parsed = json.loads(flaky_writer.buffer[0].decode().rstrip("\n"))
    assert parsed["type"] == message["type"]
    assert parsed["result"] == message["result"]
    assert parsed["data"] == message["data"]
    assert "correlation_id" in parsed
    assert "timestamp" in parsed
