"""
RobotHealthMonitor 및 서비스 재시도 헬퍼 테스트.
"""
from __future__ import annotations

import asyncio
import pytest

from main_service.constants import RobotType
from main_service.robot_coordinator import RobotHealthMonitor, _call_service_with_retry


class DummyClient:
    """ROS 서비스 클라이언트를 모사하기 위한 더미 객체."""

    def __init__(self, failures: int = 0, timeout: bool = False) -> None:
        self.failures = failures
        self.timeout = timeout
        self.calls = 0
        self.srv_name = '/dummy'

    def wait_for_service(self, timeout_sec: float) -> bool:
        return True

    def call_async(self, request):
        self.calls += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        if self.timeout:
            return future
        if self.failures > 0:
            self.failures -= 1
            future.set_exception(RuntimeError('temporary failure'))
        else:
            future.set_result({'ok': True, 'request': request})
        return future


@pytest.mark.asyncio
async def test_call_service_with_retry_succeeds_after_failures() -> None:
    from unittest.mock import MagicMock
    client = DummyClient(failures=2)
    response = await _call_service_with_retry(
        MagicMock(), # coordinator
        client,
        request={'value': 1},
        timeout_sec=0.2,
        max_attempts=3,
        base_delay=0.01,
    )
    assert client.calls == 3
    assert response['ok'] is True


@pytest.mark.asyncio
async def test_call_service_with_retry_raises_after_exhausting_attempts() -> None:
    from unittest.mock import MagicMock
    client = DummyClient(failures=3)
    with pytest.raises(RuntimeError):
        await _call_service_with_retry(
            MagicMock(), # coordinator
            client,
            request={},
            timeout_sec=0.1,
            max_attempts=2,
            base_delay=0.01,
        )
    assert client.calls == 2


def test_robot_health_monitor_detects_timeouts() -> None:
    monitor = RobotHealthMonitor(timeout=5.0)
    monitor.record(RobotType.PICKEE, 1, now=0.0)
    # 아직 타임아웃 미도달
    assert monitor.detect_unresponsive(now=4.0) == []
    # 타임아웃 발생
    offline = monitor.detect_unresponsive(now=6.0)
    assert offline == [(RobotType.PICKEE, 1)]
    # 재호출 시 중복 반환되지 않는다.
    assert monitor.detect_unresponsive(now=7.0) == []
    # 다시 상태가 들어오면 복구됨
    monitor.record(RobotType.PICKEE, 1, now=8.0)
    assert monitor.detect_unresponsive(now=14.0) == []
    assert monitor.detect_unresponsive(now=14.1) == [(RobotType.PICKEE, 1)]
