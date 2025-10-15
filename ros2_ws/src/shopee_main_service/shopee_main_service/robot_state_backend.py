"""
Robot state storage backends.

RobotState 데이터 모델과, 다양한 저장소 구현(예: 인메모리)을 위한
공통 인터페이스를 제공합니다.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Protocol

from .constants import RobotStatus, RobotType


@dataclass(slots=True)
class RobotState:
    """단일 로봇의 최신 상태를 표현합니다."""

    robot_id: int
    robot_type: RobotType
    status: str
    reserved: bool = False
    active_order_id: Optional[int] = None
    battery_level: Optional[float] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    maintenance_mode: bool = False  # 유지보수 모드 (True일 때 자동 할당 제외)

    def clone(self) -> "RobotState":
        """현재 상태의 사본을 반환합니다."""
        return RobotState(
            robot_id=self.robot_id,
            robot_type=self.robot_type,
            status=self.status,
            reserved=self.reserved,
            active_order_id=self.active_order_id,
            battery_level=self.battery_level,
            last_update=self.last_update,
            maintenance_mode=self.maintenance_mode,
        )


class RobotStateBackend(Protocol):
    """로봇 상태 저장소 공통 인터페이스."""

    async def upsert_state(self, state: RobotState) -> None:
        """상태를 추가하거나 갱신합니다."""

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        """단일 로봇 상태를 조회합니다."""

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        """모든 로봇 상태를 조회합니다."""

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        """가용 로봇 상태만 조회합니다."""

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        """로봇 예약을 시도합니다."""

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        """예약을 해제합니다."""

    async def mark_offline(self, robot_id: int) -> None:
        """로봇을 오프라인으로 표시합니다."""

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        """유지보수 모드를 설정/해제합니다."""

    async def close(self) -> None:
        """백엔드 리소스를 정리합니다."""


class InMemoryRobotStateBackend(RobotStateBackend):
    """
    인메모리 백엔드.

    기존 RobotStateStore의 동작을 동일하게 보장하기 위해
    asyncio.Lock 기반의 간단한 구현을 사용합니다.
    """

    def __init__(self) -> None:
        self._states: Dict[int, RobotState] = {}
        self._lock = asyncio.Lock()

    async def upsert_state(self, state: RobotState) -> None:
        async with self._lock:
            existing = self._states.get(state.robot_id)
            if existing:
                existing.status = state.status
                existing.battery_level = state.battery_level
                existing.active_order_id = state.active_order_id
                existing.last_update = state.last_update
                # 예약/유지보수 플래그는 외부 흐름에서만 변경합니다.
            else:
                self._states[state.robot_id] = state.clone()

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        async with self._lock:
            state = self._states.get(robot_id)
            return state.clone() if state else None

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        async with self._lock:
            values: Iterable[RobotState] = self._states.values()
            if robot_type is not None:
                values = (s for s in values if s.robot_type == robot_type)
            return [state.clone() for state in values]

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        async with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if state.robot_type == robot_type
                and not state.reserved
                and not state.maintenance_mode
                and state.status == RobotStatus.IDLE.value
            ]
            return [state.clone() for state in candidates]

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        async with self._lock:
            state = self._states.get(robot_id)
            if not state or state.reserved:
                return False
            state.reserved = True
            state.active_order_id = order_id
            return True

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return
            if order_id is not None and state.active_order_id != order_id:
                return
            state.reserved = False
            state.active_order_id = None

    async def mark_offline(self, robot_id: int) -> None:
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return
            state.status = RobotStatus.OFFLINE.value
            state.reserved = False
            state.active_order_id = None
            state.last_update = datetime.now(timezone.utc)

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return False

            state.maintenance_mode = enabled
            state.last_update = datetime.now(timezone.utc)

            if enabled and state.reserved:
                state.reserved = False
                state.active_order_id = None

            return True

    async def close(self) -> None:
        """인메모리 백엔드는 정리할 리소스가 없습니다."""
        return None
