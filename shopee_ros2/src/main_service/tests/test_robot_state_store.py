"""
Unit tests for RobotStateStore.

RobotStateStore의 상태 관리, 예약/해제, 동시성 제어를 검증합니다.
"""

import asyncio
import pytest
from datetime import datetime, timezone

from main_service.robot_state_store import RobotState, RobotStateStore
from main_service.constants import RobotStatus, RobotType

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def store() -> RobotStateStore:
    """RobotStateStore 인스턴스를 생성합니다."""
    return RobotStateStore()


@pytest.fixture
def sample_pickee_state() -> RobotState:
    """테스트용 Pickee 로봇 상태를 생성합니다."""
    return RobotState(
        robot_id=1,
        robot_type=RobotType.PICKEE,
        status=RobotStatus.IDLE.value,
        battery_level=85.0,
    )


@pytest.fixture
def sample_packee_state() -> RobotState:
    """테스트용 Packee 로봇 상태를 생성합니다."""
    return RobotState(
        robot_id=2,
        robot_type=RobotType.PACKEE,
        status=RobotStatus.IDLE.value,
        battery_level=90.0,
    )


class TestRobotStateStoreUpsert:
    """RobotStateStore.upsert_state 테스트"""

    async def test_upsert_new_state(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """새로운 로봇 상태를 추가할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)

        retrieved = await store.get_state(sample_pickee_state.robot_id)
        assert retrieved is not None
        assert retrieved.robot_id == sample_pickee_state.robot_id
        assert retrieved.robot_type == sample_pickee_state.robot_type
        assert retrieved.status == sample_pickee_state.status
        assert retrieved.battery_level == sample_pickee_state.battery_level

    async def test_upsert_update_existing_state(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """기존 로봇 상태를 업데이트할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)

        # 상태 변경
        updated_state = RobotState(
            robot_id=1,
            robot_type=RobotType.PICKEE,
            status=RobotStatus.WORKING.value,
            battery_level=75.0,
            active_order_id=123,
        )
        await store.upsert_state(updated_state)

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.status == RobotStatus.WORKING.value
        assert retrieved.battery_level == 75.0
        assert retrieved.active_order_id == 123

    async def test_upsert_preserves_reservation(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """upsert는 예약 플래그를 보존합니다."""
        await store.upsert_state(sample_pickee_state)
        await store.try_reserve(1, 999)

        # 상태 업데이트
        updated_state = RobotState(
            robot_id=1,
            robot_type=RobotType.PICKEE,
            status=RobotStatus.WORKING.value,
            battery_level=70.0,
        )
        await store.upsert_state(updated_state)

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is True  # 예약은 유지됨
        assert retrieved.status == RobotStatus.WORKING.value  # 상태는 업데이트됨


class TestRobotStateStoreReservation:
    """RobotStateStore 예약/해제 테스트"""

    async def test_try_reserve_success(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """가용 로봇을 예약할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)

        success = await store.try_reserve(1, 123)
        assert success is True

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is True
        assert retrieved.active_order_id == 123

    async def test_try_reserve_already_reserved(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """이미 예약된 로봇은 다시 예약할 수 없습니다."""
        await store.upsert_state(sample_pickee_state)

        # 첫 번째 예약 성공
        success1 = await store.try_reserve(1, 123)
        assert success1 is True

        # 두 번째 예약 실패
        success2 = await store.try_reserve(1, 456)
        assert success2 is False

        # 첫 번째 예약이 유지됨
        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.active_order_id == 123

    async def test_try_reserve_nonexistent_robot(self, store: RobotStateStore):
        """존재하지 않는 로봇은 예약할 수 없습니다."""
        success = await store.try_reserve(999, 123)
        assert success is False

    async def test_release(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """예약된 로봇을 해제할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)
        await store.try_reserve(1, 123)

        await store.release(1, 123)

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is False
        assert retrieved.active_order_id is None

    async def test_release_wrong_order_id(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """잘못된 order_id로 해제하면 예약이 유지됩니다."""
        await store.upsert_state(sample_pickee_state)
        await store.try_reserve(1, 123)

        await store.release(1, 456)  # 다른 order_id

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is True  # 예약 유지
        assert retrieved.active_order_id == 123

    async def test_release_without_order_id_validation(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """order_id 검증 없이 해제할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)
        await store.try_reserve(1, 123)

        await store.release(1)  # order_id 없이 호출

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is False
        assert retrieved.active_order_id is None


class TestRobotStateStoreQuery:
    """RobotStateStore 조회 테스트"""

    async def test_list_states_all(self, store: RobotStateStore, sample_pickee_state: RobotState, sample_packee_state: RobotState):
        """모든 로봇 상태를 조회할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)
        await store.upsert_state(sample_packee_state)

        states = await store.list_states()
        assert len(states) == 2

    async def test_list_states_filtered_by_type(self, store: RobotStateStore, sample_pickee_state: RobotState, sample_packee_state: RobotState):
        """타입별로 필터링하여 조회할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)
        await store.upsert_state(sample_packee_state)

        pickee_states = await store.list_states(RobotType.PICKEE)
        assert len(pickee_states) == 1
        assert pickee_states[0].robot_type == RobotType.PICKEE

        packee_states = await store.list_states(RobotType.PACKEE)
        assert len(packee_states) == 1
        assert packee_states[0].robot_type == RobotType.PACKEE

    async def test_list_available(self, store: RobotStateStore):
        """가용 로봇만 조회할 수 있습니다."""
        # IDLE 상태 로봇 (가용)
        state1 = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state1)

        # WORKING 상태 로봇 (불가용)
        state2 = RobotState(robot_id=2, robot_type=RobotType.PICKEE, status=RobotStatus.WORKING.value)
        await store.upsert_state(state2)

        # IDLE이지만 예약됨 (불가용)
        state3 = RobotState(robot_id=3, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state3)
        await store.try_reserve(3, 123)

        available = await store.list_available(RobotType.PICKEE)
        assert len(available) == 1
        assert available[0].robot_id == 1

    async def test_mark_offline(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """로봇을 오프라인 상태로 표시할 수 있습니다."""
        await store.upsert_state(sample_pickee_state)
        await store.try_reserve(1, 123)

        await store.mark_offline(1)

        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.status == RobotStatus.OFFLINE.value
        assert retrieved.reserved is False
        assert retrieved.active_order_id is None


class TestRobotStateStoreConcurrency:
    """RobotStateStore 동시성 테스트"""

    async def test_concurrent_reservations(self, store: RobotStateStore):
        """동시에 여러 예약 시도가 있어도 하나만 성공해야 합니다."""
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        # 10개의 동시 예약 시도
        tasks = [store.try_reserve(1, order_id) for order_id in range(100, 110)]
        results = await asyncio.gather(*tasks)

        # 정확히 하나만 성공해야 함
        success_count = sum(1 for r in results if r)
        assert success_count == 1

        # 예약된 상태 확인
        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.reserved is True
        assert 100 <= retrieved.active_order_id < 110

    async def test_concurrent_state_updates(self, store: RobotStateStore):
        """동시에 여러 상태 업데이트가 있어도 안전해야 합니다."""
        initial_state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(initial_state)

        # 10개의 동시 상태 업데이트
        async def update_state(battery: float):
            state = RobotState(
                robot_id=1,
                robot_type=RobotType.PICKEE,
                status=RobotStatus.WORKING.value,
                battery_level=battery,
            )
            await store.upsert_state(state)

        tasks = [update_state(float(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        # 최종 상태가 일관성 있게 저장되어야 함
        retrieved = await store.get_state(1)
        assert retrieved is not None
        assert retrieved.status == RobotStatus.WORKING.value
        assert retrieved.battery_level is not None

    async def test_reserve_and_release_race_condition(self, store: RobotStateStore):
        """예약과 해제가 동시에 발생해도 안전해야 합니다."""
        state = RobotState(robot_id=1, robot_type=RobotType.PICKEE, status=RobotStatus.IDLE.value)
        await store.upsert_state(state)

        # 예약 시도
        reserve_task = store.try_reserve(1, 123)
        # 거의 동시에 해제 시도
        release_task = store.release(1, 123)

        await asyncio.gather(reserve_task, release_task)

        # 최종 상태 확인 (일관성만 유지하면 됨)
        retrieved = await store.get_state(1)
        assert retrieved is not None
        # 예약되었거나 해제되었거나 둘 중 하나
        if retrieved.reserved:
            assert retrieved.active_order_id == 123
        else:
            assert retrieved.active_order_id is None


class TestRobotStateClone:
    """RobotState.clone 테스트"""

    async def test_clone_creates_independent_copy(self, store: RobotStateStore, sample_pickee_state: RobotState):
        """clone은 독립적인 복사본을 생성합니다."""
        await store.upsert_state(sample_pickee_state)

        original = await store.get_state(1)
        assert original is not None

        cloned = original.clone()

        # 값은 동일
        assert cloned.robot_id == original.robot_id
        assert cloned.status == original.status
        assert cloned.battery_level == original.battery_level

        # 하지만 다른 객체
        assert cloned is not original

        # 한쪽을 수정해도 다른 쪽에 영향 없음
        cloned.status = RobotStatus.ERROR.value
        assert original.status == RobotStatus.IDLE.value
