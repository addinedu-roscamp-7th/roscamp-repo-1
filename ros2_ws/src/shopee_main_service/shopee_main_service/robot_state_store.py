"""
Robot state storage primitives.

이 모듈은 로봇 상태를 인메모리로 캐시하고, 예약/해제를 위한
동시성 안전 API를 제공합니다. 향후 Redis 등의 외부 저장소로
교체할 수 있도록 최소한의 추상화를 유지합니다.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

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


class RobotStateStore:
    """
    로봇 상태 캐시 및 예약 관리 스토어.

    - ROS 토픽으로부터 전달되는 상태를 수신하여 저장합니다.
    - OrderService 등에서 가용 로봇을 조회하고 예약/해제할 수 있도록 합니다.
    - 현재는 인메모리 dict 기반으로 구현되어 있으며, 향후 외부 스토리지로 교체할 수 있도록
      메서드 시그니처를 비동기로 정의했습니다.
    """

    def __init__(self) -> None:
        self._states: Dict[int, RobotState] = {}
        self._lock = asyncio.Lock()

    async def upsert_state(self, state: RobotState) -> None:
        """
        로봇 상태를 추가하거나 갱신합니다.

        Args:
            state: 최신 상태 정보
        """
        async with self._lock:
            existing = self._states.get(state.robot_id)
            if existing:
                existing.status = state.status
                existing.battery_level = state.battery_level
                existing.active_order_id = state.active_order_id
                existing.last_update = state.last_update
                # 예약 플래그는 외부 흐름에서만 변경합니다.
            else:
                self._states[state.robot_id] = state

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        """로봇 ID에 해당하는 상태를 반환합니다."""
        async with self._lock:
            state = self._states.get(robot_id)
            return state.clone() if state else None

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        """
        모든 로봇 상태를 조회합니다.

        Args:
            robot_type: 특정 타입(Pickee/Packee)으로 필터링할 경우 지정
        """
        async with self._lock:
            values: Iterable[RobotState] = self._states.values()
            if robot_type is not None:
                values = (s for s in values if s.robot_type == robot_type)
            return [state.clone() for state in values]

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        """
        가용 상태의 로봇 목록을 반환합니다.
        유지보수 모드인 로봇은 제외됩니다.
        """
        async with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if state.robot_type == robot_type
                and not state.reserved
                and not state.maintenance_mode  # 유지보수 모드 제외
                and state.status == RobotStatus.IDLE.value
            ]
            return [state.clone() for state in candidates]

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        """
        로봇을 예약합니다.

        예약에 성공하면 True, 이미 예약 중이거나 상태가 없는 경우 False 를 반환합니다.
        """
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return False
            if state.reserved:
                return False
            state.reserved = True
            state.active_order_id = order_id
            return True

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        """
        예약된 로봇을 해제합니다.

        Args:
            robot_id: 대상 로봇 ID
            order_id: 특정 주문에 할당된 상태인지 검증할 때 사용
        """
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return
            if order_id is not None and state.active_order_id != order_id:
                return
            state.reserved = False
            state.active_order_id = None

    async def mark_offline(self, robot_id: int) -> None:
        """
        로봇을 오프라인 상태로 표시하고 예약을 해제합니다.
        """
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return
            state.status = RobotStatus.OFFLINE.value
            state.reserved = False
            state.active_order_id = None
            state.last_update = datetime.now(timezone.utc)

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        """
        로봇의 유지보수 모드를 설정/해제합니다.

        Args:
            robot_id: 로봇 ID
            enabled: True일 때 유지보수 모드 활성화, False일 때 비활성화

        Returns:
            성공 여부

        Note:
            유지보수 모드 진입 시 예약이 해제됩니다.
        """
        async with self._lock:
            state = self._states.get(robot_id)
            if not state:
                return False

            state.maintenance_mode = enabled
            state.last_update = datetime.now(timezone.utc)

            # 유지보수 모드 진입 시 예약 해제
            if enabled and state.reserved:
                state.reserved = False
                state.active_order_id = None

            return True
