"""
DashboardController 및 관련 헬퍼에 대한 단위 테스트.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from main_service.dashboard import DashboardBridge, DashboardController, DashboardDataProvider
from main_service.robot_state_backend import RobotState, RobotType


pytestmark = pytest.mark.asyncio


async def test_dashboard_bridge_publish_and_receive():
    loop = asyncio.get_running_loop()
    bridge = DashboardBridge(loop)

    await bridge.publish_async({'type': 'test', 'value': 1})
    await asyncio.sleep(0)  # 큐 이동 대기

    message = bridge.get_for_gui(timeout=0.1)
    assert message == {'type': 'test', 'value': 1}

    await bridge.close()


async def test_dashboard_data_provider_collect_snapshot():
    order_service = AsyncMock()
    order_service.get_active_orders_snapshot.return_value = {'orders': [], 'summary': {}}

    state_store = AsyncMock()
    state_store.list_states.return_value = [
        RobotState(robot_id=1, robot_type=RobotType.PICKEE, status='IDLE'),
    ]

    async def metrics_provider():
        return {'latency': 10}

    provider = DashboardDataProvider(order_service, state_store, metrics_provider)
    snapshot = await provider.collect_snapshot()

    assert snapshot['orders'] == {'orders': [], 'summary': {}}
    assert snapshot['robots'][0]['robot_id'] == 1
    assert snapshot['metrics'] == {'latency': 10}


async def test_dashboard_controller_cycle():
    loop = asyncio.get_running_loop()

    data_provider = MagicMock()
    data_provider.collect_snapshot = AsyncMock(return_value={'orders': {}, 'robots': [], 'metrics': {}})

    event_bus = MagicMock()
    event_bus.register_listener = MagicMock()
    event_bus.unregister_listener = MagicMock()

    controller = DashboardController(
        loop,
        data_provider,
        event_bus,
        interval=0.05,
    )

    await controller.start()
    await asyncio.sleep(0.12)

    message = controller.bridge.get_for_gui(timeout=0.2)
    assert message is not None
    assert message['type'] == 'snapshot'

    await controller.stop()
    event_bus.register_listener.assert_called()
    event_bus.unregister_listener.assert_called()
