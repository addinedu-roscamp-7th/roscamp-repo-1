"""RedisRobotStateBackend 행동 검증 테스트."""
from __future__ import annotations

import pytest

from shopee_main_service.constants import RobotStatus, RobotType
from shopee_main_service.redis_robot_state_backend import RedisRobotStateBackend
from shopee_main_service.robot_state_backend import RobotState

try:
    import fakeredis.aioredis as fakeredis
except ModuleNotFoundError:  # pragma: no cover - 테스트 환경에서 fakeredis 미설치 시
    fakeredis = None


pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(fakeredis is None, reason="fakeredis가 설치되어 있지 않습니다.")
async def test_redis_backend_upsert_reserve_release() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    backend = RedisRobotStateBackend(url="redis://localhost:6379/0", client=client)

    state = RobotState(
        robot_id=1,
        robot_type=RobotType.PICKEE,
        status=RobotStatus.IDLE.value,
        battery_level=85.0,
    )
    await backend.upsert_state(state)

    stored = await backend.get_state(1)
    assert stored is not None
    assert stored.robot_id == 1
    assert stored.robot_type == RobotType.PICKEE
    assert stored.status == RobotStatus.IDLE.value

    available = await backend.list_available(RobotType.PICKEE)
    assert len(available) == 1
    assert available[0].robot_id == 1

    reserved = await backend.try_reserve(robot_id=1, order_id=42)
    assert reserved is True

    # 두 번째 예약은 실패해야 함
    reserved_again = await backend.try_reserve(robot_id=1, order_id=43)
    assert reserved_again is False

    await backend.release(robot_id=1, order_id=42)
    released_state = await backend.get_state(1)
    assert released_state is not None
    assert released_state.reserved is False
    assert released_state.active_order_id is None

    await backend.close()


@pytest.mark.skipif(fakeredis is None, reason="fakeredis가 설치되어 있지 않습니다.")
async def test_redis_backend_recovers_from_unknown_robot_type() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    backend = RedisRobotStateBackend(url="redis://localhost:6379/0", client=client)

    key = "shopee:robot_state:5"
    await client.hset(
        key,
        mapping={
            "robot_id": "5",
            "robot_type": "STRANGE",
            "status": RobotStatus.IDLE.value,
            "reserved": "0",
            "active_order_id": "",
            "battery_level": "50.0",
            "last_update": "",
            "maintenance_mode": "0",
        },
    )

    state = await backend.get_state(5)
    assert state is not None
    assert state.robot_type == RobotType.PICKEE  # fallback

    await backend.close()
