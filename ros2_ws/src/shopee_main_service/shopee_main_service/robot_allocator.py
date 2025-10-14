"""
Robot allocation strategies and orchestrator skeleton.

OrderService 등에서 로봇을 선택/예약하기 위한 진입점을 정의합니다.
현재는 라운드로빈 전략만 기본으로 제공하며, 실제 선택 로직은
향후 요구사항에 맞춰 확장 예정입니다.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

from .constants import RobotType
from .robot_state_store import RobotState, RobotStateStore


@dataclass(slots=True)
class AllocationContext:
    """로봇 할당 시 참고할 추가 정보를 담습니다."""

    order_id: Optional[int] = None
    required_type: RobotType = RobotType.PICKEE
    metadata: Dict[str, object] = field(default_factory=dict)


class AllocationStrategy(Protocol):
    """로봇 선택 전략 프로토콜."""

    async def select(self, context: AllocationContext, candidates: list[RobotState]) -> Optional[RobotState]:
        """후보 목록 중 하나를 선택하거나 None 을 반환합니다."""


class RoundRobinStrategy:
    """
    단순 라운드로빈 전략.

    마지막으로 선택된 로봇 ID를 기억했다가, 다음 호출에서 그 다음 후보를 반환합니다.
    """

    def __init__(self) -> None:
        self._last_robot_id: Optional[int] = None
        self._lock = asyncio.Lock()

    async def select(self, context: AllocationContext, candidates: list[RobotState]) -> Optional[RobotState]:
        if not candidates:
            return None

        ordered = sorted(candidates, key=lambda state: state.robot_id)

        async with self._lock:
            if self._last_robot_id is None:
                chosen = ordered[0]
                self._last_robot_id = chosen.robot_id
                return chosen

            # 현재 마지막 ID보다 큰 후보를 우선 찾고, 없으면 맨 앞을 선택한다.
            for state in ordered:
                if state.robot_id > self._last_robot_id:
                    self._last_robot_id = state.robot_id
                    return state

            chosen = ordered[0]
            self._last_robot_id = chosen.robot_id
            return chosen

        return None


class LeastWorkloadStrategy:
    """
    최소 작업 부하 전략.

    현재 활성화된 작업이 없는 로봇을 우선 선택합니다.
    모든 로봇이 작업 중이면 그 중에서도 ID가 가장 작은 로봇을 선택합니다.
    (향후 작업 이력 DB와 연동하여 실제 작업 횟수/시간 기반으로 확장 가능)
    """

    async def select(self, context: AllocationContext, candidates: list[RobotState]) -> Optional[RobotState]:
        if not candidates:
            return None

        # active_order_id가 없는 로봇 우선
        idle_robots = [c for c in candidates if not c.active_order_id]

        if idle_robots:
            # 유휴 로봇 중 ID가 가장 작은 것 선택
            return min(idle_robots, key=lambda s: s.robot_id)

        # 모두 작업 중이면 ID가 가장 작은 것 선택
        return min(candidates, key=lambda s: s.robot_id)


class BatteryAwareStrategy:
    """
    배터리 고려 전략.

    배터리 임계치 이상인 로봇 중에서 배터리가 가장 높은 로봇을 선택합니다.
    임계치 이상 로봇이 없으면 None을 반환하여 충전을 유도합니다.
    """

    def __init__(self, min_battery_level: float = 20.0) -> None:
        """
        Args:
            min_battery_level: 최소 배터리 임계치 (%)
        """
        self._min_battery_level = min_battery_level

    async def select(self, context: AllocationContext, candidates: list[RobotState]) -> Optional[RobotState]:
        if not candidates:
            return None

        # 배터리 임계치 이상인 로봇만 필터링
        sufficient_battery = [
            c for c in candidates
            if c.battery_level is not None and c.battery_level >= self._min_battery_level
        ]

        if not sufficient_battery:
            # 배터리 충분한 로봇 없음
            return None

        # 배터리가 가장 높은 로봇 선택
        return max(sufficient_battery, key=lambda s: s.battery_level or 0.0)


class RobotAllocator:
    """
    로봇 선택 및 예약을 담당하는 오케스트레이터.

    현재는 가용 로봇 목록을 가져와 전략에 위임한 후, 예약을 시도하는
    최소한의 골격만 제공한다.
    """

    def __init__(
        self,
        store: RobotStateStore,
        strategy: Optional[AllocationStrategy] = None,
    ) -> None:
        self._store = store
        self._strategy = strategy or RoundRobinStrategy()

    async def reserve_robot(self, context: AllocationContext) -> Optional[RobotState]:
        """
        로봇을 선택/예약한 뒤 상태를 반환합니다.

        예약에 실패하면 None을 반환하며, 호출 측에서 재시도나 다른 대응을 결정합니다.
        """
        candidates = await self._store.list_available(context.required_type)
        if not candidates:
            return None

        chosen = await self._strategy.select(context, candidates)
        if not chosen:
            return None

        reserved = await self._store.try_reserve(chosen.robot_id, context.order_id or -1)
        if not reserved:
            return None

        return await self._store.get_state(chosen.robot_id)

    async def release_robot(self, robot_id: int, order_id: Optional[int] = None) -> None:
        """예약된 로봇을 해제합니다."""
        await self._store.release(robot_id, order_id)
