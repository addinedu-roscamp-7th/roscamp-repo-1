"""
Robot state storage façade.

백엔드 구현을 주입 받아 일관된 API를 제공합니다.
"""
from __future__ import annotations

from typing import List, Optional

from .constants import RobotType
from .robot_state_backend import (
    InMemoryRobotStateBackend,
    RobotState,
    RobotStateBackend,
)

__all__ = ["RobotState", "RobotStateStore"]


class RobotStateStore:
    """
    로봇 상태 캐시 및 예약 관리 스토어.

    실제 저장소 동작은 RobotStateBackend 구현이 담당하며,
    기본값은 인메모리 버전(InMemoryRobotStateBackend)입니다.
    """

    def __init__(self, backend: Optional[RobotStateBackend] = None) -> None:
        self._backend = backend or InMemoryRobotStateBackend()

    @property
    def backend(self) -> RobotStateBackend:
        """현재 사용 중인 백엔드를 반환합니다."""
        return self._backend

    async def upsert_state(self, state: RobotState) -> None:
        await self._backend.upsert_state(state)

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        state = await self._backend.get_state(robot_id)
        return state.clone() if state else None

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        states = await self._backend.list_states(robot_type)
        return [state.clone() for state in states]

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        states = await self._backend.list_available(robot_type)
        return [state.clone() for state in states]

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        return await self._backend.try_reserve(robot_id, order_id)

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        await self._backend.release(robot_id, order_id)

    async def mark_offline(self, robot_id: int) -> None:
        await self._backend.mark_offline(robot_id)

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        return await self._backend.set_maintenance_mode(robot_id, enabled)

    async def close(self) -> None:
        """백엔드 리소스를 정리합니다."""
        await self._backend.close()
