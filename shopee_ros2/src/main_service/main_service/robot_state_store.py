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
        """
        로봇 상태를 생성하거나 업데이트합니다.

        존재하지 않는 로봇이면 새로 생성하고, 이미 존재하면 상태를 업데이트합니다.

        Args:
            state: 저장할 로봇 상태 정보

        사용 예:
            state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status="IDLE")
            await robot_state_store.upsert_state(state)
        """
        await self._backend.upsert_state(state)

    async def get_state(self, robot_id: int) -> Optional[RobotState]:
        """
        특정 로봇의 상태를 조회합니다.

        Args:
            robot_id: 조회할 로봇 ID

        Returns:
            Optional[RobotState]: 로봇 상태 객체. 존재하지 않으면 None

        사용 예:
            state = await robot_state_store.get_state(robot_id=1)
            if state:
                print(f"로봇 상태: {state.status}")
        """
        state = await self._backend.get_state(robot_id)
        return state.clone() if state else None

    async def list_states(self, robot_type: Optional[RobotType] = None) -> List[RobotState]:
        """
        모든 로봇 또는 특정 타입의 로봇 상태 목록을 조회합니다.

        Args:
            robot_type: 필터링할 로봇 타입. None이면 모든 타입 반환

        Returns:
            List[RobotState]: 로봇 상태 목록

        사용 예:
            # 모든 로봇 조회
            all_robots = await robot_state_store.list_states()

            # Pickee 로봇만 조회
            pickee_robots = await robot_state_store.list_states(RobotType.PICKEE)
        """
        states = await self._backend.list_states(robot_type)
        return [state.clone() for state in states]

    async def list_available(self, robot_type: RobotType) -> List[RobotState]:
        """
        작업 할당 가능한 로봇 목록을 조회합니다.

        가용 조건:
        - 예약되지 않음 (reserved=False)
        - 유지보수 모드가 아님 (maintenance_mode=False)
        - 오프라인 상태가 아님 (status != OFFLINE)

        Args:
            robot_type: 조회할 로봇 타입 (PICKEE 또는 PACKEE)

        Returns:
            List[RobotState]: 가용한 로봇 상태 목록

        사용 예:
            available = await robot_state_store.list_available(RobotType.PICKEE)
            if available:
                selected = available[0]
                print(f"가용 로봇: {selected.robot_id}")
        """
        states = await self._backend.list_available(robot_type)
        return [state.clone() for state in states]

    async def try_reserve(self, robot_id: int, order_id: int) -> bool:
        """
        로봇을 특정 주문에 예약 시도합니다.

        이미 예약된 로봇이거나 유지보수 모드인 경우 실패합니다.

        Args:
            robot_id: 예약할 로봇 ID
            order_id: 할당할 주문 ID

        Returns:
            bool: 예약 성공 여부

        사용 예:
            success = await robot_state_store.try_reserve(robot_id=1, order_id=100)
            if success:
                print("로봇 예약 성공")
            else:
                print("로봇이 이미 예약되었거나 사용 불가능합니다")
        """
        return await self._backend.try_reserve(robot_id, order_id)

    async def release(self, robot_id: int, order_id: Optional[int] = None) -> None:
        """
        로봇 예약을 해제합니다.

        Args:
            robot_id: 예약 해제할 로봇 ID
            order_id: 확인용 주문 ID (선택적). 제공 시 해당 주문으로 예약된 경우만 해제

        사용 예:
            # 무조건 해제
            await robot_state_store.release(robot_id=1)

            # 특정 주문으로 예약된 경우만 해제
            await robot_state_store.release(robot_id=1, order_id=100)
        """
        await self._backend.release(robot_id, order_id)

    async def mark_offline(self, robot_id: int) -> None:
        """
        로봇을 오프라인 상태로 표시하고 예약을 해제합니다.

        로봇이 응답하지 않거나 연결이 끊긴 경우 호출됩니다.

        Args:
            robot_id: 오프라인 처리할 로봇 ID

        사용 예:
            await robot_state_store.mark_offline(robot_id=1)
        """
        await self._backend.mark_offline(robot_id)

    async def set_maintenance_mode(self, robot_id: int, enabled: bool) -> bool:
        """
        로봇의 유지보수 모드를 설정/해제합니다.

        유지보수 모드가 활성화된 로봇은 새로운 작업에 할당되지 않습니다.

        Args:
            robot_id: 대상 로봇 ID
            enabled: True면 유지보수 모드 활성화, False면 비활성화

        Returns:
            bool: 성공 여부 (로봇이 존재하지 않으면 False)

        사용 예:
            # 유지보수 모드 활성화
            success = await robot_state_store.set_maintenance_mode(robot_id=1, enabled=True)

            # 유지보수 모드 비활성화
            await robot_state_store.set_maintenance_mode(robot_id=1, enabled=False)
        """
        return await self._backend.set_maintenance_mode(robot_id, enabled)

    async def close(self) -> None:
        """백엔드 리소스를 정리합니다."""
        await self._backend.close()
