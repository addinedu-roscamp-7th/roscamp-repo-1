"""
Unit tests for RobotAllocator.

RobotAllocator의 로봇 선택 전략 및 예약 로직을 검증합니다.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from shopee_main_service.robot_allocator import (
    AllocationContext,
    RobotAllocator,
    RoundRobinStrategy,
)
from shopee_main_service.robot_state_store import RobotState, RobotStateStore
from shopee_main_service.constants import RobotStatus, RobotType

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def store() -> RobotStateStore:
    """RobotStateStore 인스턴스를 생성합니다."""
    return RobotStateStore()


@pytest.fixture
def allocator(store: RobotStateStore) -> RobotAllocator:
    """RobotAllocator 인스턴스를 생성합니다."""
    return RobotAllocator(store)


class TestRoundRobinStrategy:
    """RoundRobinStrategy 테스트"""

    async def test_selects_first_robot_when_empty(self):
        """첫 호출 시 첫 번째 로봇을 선택합니다."""
        strategy = RoundRobinStrategy()
        candidates = [
            RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=2, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=3, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
        ]
        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)

        selected = await strategy.select(context, candidates)

        assert selected is not None
        assert selected.robot_id == 1

    async def test_round_robin_selection(self):
        """라운드로빈 방식으로 순차적으로 선택합니다."""
        strategy = RoundRobinStrategy()
        candidates = [
            RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=2, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=3, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
        ]
        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)

        # 첫 번째: robot_id=1
        selected1 = await strategy.select(context, candidates)
        assert selected1.robot_id == 1

        # 두 번째: robot_id=2
        selected2 = await strategy.select(context, candidates)
        assert selected2.robot_id == 2

        # 세 번째: robot_id=3
        selected3 = await strategy.select(context, candidates)
        assert selected3.robot_id == 3

        # 네 번째: 다시 robot_id=1 (순환)
        selected4 = await strategy.select(context, candidates)
        assert selected4.robot_id == 1

    async def test_wraps_around_correctly(self):
        """마지막 로봇 이후 다시 처음으로 돌아갑니다."""
        strategy = RoundRobinStrategy()
        candidates = [
            RobotState(robot_id=5, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=10, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
        ]
        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)

        # 5 -> 10 -> 5 순서로 선택
        selected1 = await strategy.select(context, candidates)
        assert selected1.robot_id == 5

        selected2 = await strategy.select(context, candidates)
        assert selected2.robot_id == 10

        selected3 = await strategy.select(context, candidates)
        assert selected3.robot_id == 5

    async def test_returns_none_when_no_candidates(self):
        """후보가 없으면 None을 반환합니다."""
        strategy = RoundRobinStrategy()
        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)

        selected = await strategy.select(context, [])

        assert selected is None

    async def test_handles_unordered_robot_ids(self):
        """로봇 ID가 순서대로 정렬되지 않아도 동작합니다."""
        strategy = RoundRobinStrategy()
        candidates = [
            RobotState(robot_id=10, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=3, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
            RobotState(robot_id=7, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value),
        ]
        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)

        # 정렬 후 선택: 3 -> 7 -> 10
        selected1 = await strategy.select(context, candidates)
        assert selected1.robot_id == 3

        selected2 = await strategy.select(context, candidates)
        assert selected2.robot_id == 7

        selected3 = await strategy.select(context, candidates)
        assert selected3.robot_id == 10


class TestRobotAllocatorReservation:
    """RobotAllocator.reserve_robot 테스트"""

    async def test_reserve_robot_success(self, store: RobotStateStore, allocator: RobotAllocator):
        """가용 로봇을 성공적으로 예약할 수 있습니다."""
        # 가용 로봇 추가
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        context = AllocationContext(order_id=123, required_type=RobotType.PICKEE)
        reserved = await allocator.reserve_robot(context)

        assert reserved is not None
        assert reserved.robot_id == 1
        assert reserved.reserved is True
        assert reserved.active_order_id == 123

    async def test_reserve_robot_no_available(self, store: RobotStateStore, allocator: RobotAllocator):
        """가용 로봇이 없으면 None을 반환합니다."""
        # 사용 중인 로봇만 추가
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.WORKING.value)
        await store.upsert_state(state)

        context = AllocationContext(order_id=123, required_type=RobotType.PICKEE)
        reserved = await allocator.reserve_robot(context)

        assert reserved is None

    async def test_reserve_robot_filters_by_type(self, store: RobotStateStore, allocator: RobotAllocator):
        """요청한 타입의 로봇만 선택합니다."""
        # Pickee 로봇
        pickee_state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(pickee_state)

        # Packee 로봇
        packee_state = RobotState(robot_id=2, robot_type=RobotType.PACKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(packee_state)

        # Packee 요청
        context = AllocationContext(order_id=123, required_type=RobotType.PACKEE)
        reserved = await allocator.reserve_robot(context)

        assert reserved is not None
        assert reserved.robot_id == 2
        assert reserved.robot_type == RobotType.PACKEE

    async def test_reserve_robot_uses_strategy(self, store: RobotStateStore):
        """전략을 사용하여 로봇을 선택합니다."""
        # 여러 로봇 추가
        for robot_id in [1, 2, 3]:
            state = RobotState(robot_id=robot_id, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
            await store.upsert_state(state)

        strategy = RoundRobinStrategy()
        allocator = RobotAllocator(store, strategy)

        # 라운드로빈 순서대로 예약
        context1 = AllocationContext(order_id=100, required_type=RobotType.PICKEE)
        reserved1 = await allocator.reserve_robot(context1)
        assert reserved1.robot_id == 1

        # 두 번째 예약 (robot 1은 이미 예약됨)
        context2 = AllocationContext(order_id=101, required_type=RobotType.PICKEE)
        reserved2 = await allocator.reserve_robot(context2)
        assert reserved2.robot_id == 2

    async def test_reserve_robot_excludes_reserved_robots(self, store: RobotStateStore, allocator: RobotAllocator):
        """이미 예약된 로봇은 후보에서 제외합니다."""
        # 두 로봇 추가
        state1 = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        state2 = RobotState(robot_id=2, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state1)
        await store.upsert_state(state2)

        # 첫 번째 로봇 예약
        context1 = AllocationContext(order_id=100, required_type=RobotType.PICKEE)
        reserved1 = await allocator.reserve_robot(context1)
        assert reserved1.robot_id == 1

        # 두 번째 예약은 두 번째 로봇 선택
        context2 = AllocationContext(order_id=101, required_type=RobotType.PICKEE)
        reserved2 = await allocator.reserve_robot(context2)
        assert reserved2.robot_id == 2


class TestRobotAllocatorRelease:
    """RobotAllocator.release_robot 테스트"""

    async def test_release_robot(self, store: RobotStateStore, allocator: RobotAllocator):
        """예약된 로봇을 해제할 수 있습니다."""
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        # 예약
        context = AllocationContext(order_id=123, required_type=RobotType.PICKEE)
        reserved = await allocator.reserve_robot(context)
        assert reserved is not None

        # 해제
        await allocator.release_robot(1, 123)

        # 해제 확인
        released_state = await store.get_state(1)
        assert released_state is not None
        assert released_state.reserved is False
        assert released_state.active_order_id is None

    async def test_release_robot_allows_reuse(self, store: RobotStateStore, allocator: RobotAllocator):
        """해제된 로봇은 다시 예약할 수 있습니다."""
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        # 첫 번째 예약
        context1 = AllocationContext(order_id=100, required_type=RobotType.PICKEE)
        reserved1 = await allocator.reserve_robot(context1)
        assert reserved1 is not None

        # 해제
        await allocator.release_robot(1, 100)

        # 두 번째 예약
        context2 = AllocationContext(order_id=101, required_type=RobotType.PICKEE)
        reserved2 = await allocator.reserve_robot(context2)
        assert reserved2 is not None
        assert reserved2.robot_id == 1
        assert reserved2.active_order_id == 101


class TestRobotAllocatorEdgeCases:
    """RobotAllocator 엣지 케이스 테스트"""

    async def test_reserve_without_order_id(self, store: RobotStateStore, allocator: RobotAllocator):
        """order_id 없이도 예약할 수 있습니다."""
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        context = AllocationContext(order_id=None, required_type=RobotType.PICKEE)
        reserved = await allocator.reserve_robot(context)

        assert reserved is not None
        assert reserved.robot_id == 1
        assert reserved.reserved is True
        # order_id=None이면 -1로 저장됨
        assert reserved.active_order_id == -1

    async def test_concurrent_reservations(self, store: RobotStateStore, allocator: RobotAllocator):
        """동시 예약 시도가 안전하게 처리됩니다."""
        import asyncio

        # 여러 로봇 추가
        for robot_id in range(1, 6):
            state = RobotState(robot_id=robot_id, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
            await store.upsert_state(state)

        # 10개의 동시 예약 시도
        async def reserve_for_order(order_id: int):
            context = AllocationContext(order_id=order_id, required_type=RobotType.PICKEE)
            return await allocator.reserve_robot(context)

        tasks = [reserve_for_order(order_id) for order_id in range(100, 110)]
        results = await asyncio.gather(*tasks)

        # 모든 예약이 성공했는지 확인 (5개 로봇이므로 5개만 성공)
        successful = [r for r in results if r is not None]
        failed = [r for r in results if r is None]

        assert len(successful) == 5
        assert len(failed) == 5

        # 예약된 로봇 ID가 중복되지 않음
        reserved_ids = [r.robot_id for r in successful]
        assert len(reserved_ids) == len(set(reserved_ids))

    async def test_custom_strategy_integration(self, store: RobotStateStore):
        """커스텀 전략을 사용할 수 있습니다."""
        # 커스텀 전략: 항상 가장 높은 ID 선택
        class HighestIdStrategy:
            async def select(self, context, candidates):
                if not candidates:
                    return None
                return max(candidates, key=lambda s: s.robot_id)

        # 로봇 추가
        for robot_id in [3, 1, 5, 2]:
            state = RobotState(robot_id=robot_id, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
            await store.upsert_state(state)

        strategy = HighestIdStrategy()
        allocator = RobotAllocator(store, strategy)

        context = AllocationContext(order_id=100, required_type=RobotType.PICKEE)
        reserved = await allocator.reserve_robot(context)

        assert reserved is not None
        assert reserved.robot_id == 5  # 가장 높은 ID
