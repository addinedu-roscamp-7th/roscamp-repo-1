"""
대시보드 데이터 컨트롤러와 스레드 브릿지 구현

PyQt 기반 GUI가 asyncio/ROS 루프와 데이터를 주고받을 수 있도록
안전한 통신 경로를 제공한다.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..constants import EventTopic
from ..robot_state_store import RobotState, RobotStateStore


class DashboardBridge:
    """
    asyncio 이벤트 루프와 GUI 스레드 간 양방향 통신을 위한 브릿지.

    - 서비스 측에서는 `publish_async`로 데이터를 GUI 큐에 전달한다.
    - GUI 스레드에서는 `get_for_gui`로 데이터를 폴링한다.
    - GUI에서 전달한 명령은 `send_from_gui` → `receive_command` 경로로 이벤트 루프에 전달된다.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._to_gui_async: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._to_gui_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._from_gui_async: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._stop_sentinel: Dict[str, Any] = {'type': '__stop__'}
        self._pump_task: Optional[asyncio.Task] = loop.create_task(self._pump_to_gui())
        self._closed = threading.Event()

    async def _pump_to_gui(self) -> None:
        """
        asyncio 큐에서 GUI 큐로 데이터를 이동한다.
        """
        try:
            while True:
                payload = await self._to_gui_async.get()
                if payload is self._stop_sentinel:
                    break
                self._to_gui_queue.put(payload)
        finally:
            self._closed.set()

    async def publish_async(self, payload: Dict[str, Any]) -> None:
        """
        GUI 스레드로 전달할 데이터를 비동기적으로 등록한다.
        """
        await self._to_gui_async.put(payload)

    def get_for_gui(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        GUI 스레드에서 호출하여 최신 데이터를 폴링한다.

        Args:
            timeout: 대기 시간(초). 0이면 즉시 반환.
        """
        try:
            return self._to_gui_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_from_gui(self, payload: Dict[str, Any]) -> None:
        """
        GUI 스레드에서 발생한 명령을 이벤트 루프로 전달한다.
        """
        asyncio.run_coroutine_threadsafe(self._from_gui_async.put(payload), self._loop)

    async def receive_command(self) -> Dict[str, Any]:
        """
        GUI에서 전달한 명령을 비동기적으로 대기한다.
        """
        return await self._from_gui_async.get()

    async def close(self) -> None:
        """
        브릿지를 종료하고 내부 자원을 정리한다.
        """
        from ..constants import GUI_SHUTDOWN_TIMEOUT

        await self._to_gui_async.put(self._stop_sentinel)
        if self._pump_task:
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
        self._closed.wait(timeout=GUI_SHUTDOWN_TIMEOUT)


class DashboardDataProvider:
    """
    대시보드에 필요한 스냅샷 데이터를 수집하는 헬퍼.
    """

    def __init__(
        self,
        order_service,
        robot_state_store: RobotStateStore,
        metrics_provider: Optional[
            Callable[[Dict[str, Any], List[RobotState]], Awaitable[Dict[str, Any]]]
        ] = None,
        metrics_collector=None,  # v5.1 추가
    ) -> None:
        self._order_service = order_service
        self._robot_state_store = robot_state_store
        self._metrics_provider = metrics_provider
        self._metrics_collector = metrics_collector  # v5.1 추가

    async def collect_snapshot(self) -> Dict[str, Any]:
        """
        진행 중 주문, 로봇 상태, 메트릭을 묶어 반환한다.
        """
        orders_snapshot = await self._order_service.get_active_orders_snapshot()
        robot_states = await self._robot_state_store.list_states()

        # 로봇 상태를 직렬화
        serialized_states = [asdict(state) for state in robot_states]

        # 메트릭 수집
        metrics = {}

        # v5.1: MetricsCollector 사용 (우선순위 높음)
        if self._metrics_collector:
            try:
                metrics = await self._metrics_collector.collect_metrics(serialized_states)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f'MetricsCollector failed: {e}')

        # 레거시 metrics_provider 지원 (하위 호환성)
        elif self._metrics_provider:
            metrics = await self._metrics_provider(
                orders_snapshot=orders_snapshot,
                robot_states=robot_states,
            )

        return {
            'orders': orders_snapshot,
            'robots': serialized_states,
            'metrics': metrics,
        }


class DashboardController:
    """
    대시보드 데이터 수집 루프 및 이벤트 포워더.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        data_provider: DashboardDataProvider,
        event_bus,
        *,
        interval: float = 1.0,
    ) -> None:
        self._loop = loop
        self._data_provider = data_provider
        self._event_bus = event_bus
        self._interval = max(0.1, interval)
        self._bridge = DashboardBridge(loop)
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def bridge(self) -> DashboardBridge:
        """GUI와 통신할 브릿지를 반환한다."""
        return self._bridge

    async def start(self) -> None:
        """
        주기적 데이터 수집과 이벤트 전송을 시작한다.
        """
        if self._running:
            return
        self._running = True
        self._event_bus.subscribe(EventTopic.APP_PUSH, self._on_app_push)
        self._event_bus.subscribe(EventTopic.ROS_TOPIC_RECEIVED, self._on_ros_topic_received)
        self._event_bus.subscribe(EventTopic.ROS_SERVICE_CALLED, self._on_ros_service_event)
        self._event_bus.subscribe(EventTopic.ROS_SERVICE_RESPONDED, self._on_ros_service_event)
        self._event_bus.subscribe(EventTopic.TCP_MESSAGE_RECEIVED, self._on_tcp_received)
        self._event_bus.subscribe(EventTopic.TCP_MESSAGE_SENT, self._on_tcp_sent)
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())

    async def stop(self) -> None:
        """
        데이터 수집과 이벤트 포워딩을 중단한다.
        """
        if not self._running:
            return
        self._running = False
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        await self._bridge.close()

    async def _snapshot_loop(self) -> None:
        """
        지정된 간격으로 스냅샷을 수집하여 GUI로 전달한다.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info('Dashboard snapshot loop started')

        try:
            iteration = 0
            while True:
                try:
                    snapshot = await self._data_provider.collect_snapshot()
                    await self._bridge.publish_async(
                        {
                            'type': 'snapshot',
                            'data': snapshot,
                        }
                    )
                    iteration += 1
                    if iteration % 300 == 1:  # 5분마다만 로그 (300초)
                        logger.info(f'Dashboard snapshot loop running: {iteration} iterations, robots: {len(snapshot.get("robots", []))}')
                except Exception as e:
                    logger.exception(f'Error in snapshot collection: {e}')

                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info('Dashboard snapshot loop cancelled')
            raise

    async def _on_app_push(self, event_data: Dict[str, Any]) -> None:
        """앱 푸시 이벤트를 GUI로 전달한다."""
        await self._bridge.publish_async({'type': 'event', 'data': event_data})

    async def _on_ros_topic_received(self, event_data: Dict[str, Any]) -> None:
        """ROS 토픽 수신 이벤트를 GUI로 전달한다."""
        await self._bridge.publish_async({'type': 'ros_topic', 'data': event_data})

    async def _on_ros_service_event(self, event_data: Dict[str, Any]) -> None:
        """ROS 서비스 호출/응답 이벤트를 GUI로 전달한다."""
        await self._bridge.publish_async({'type': 'ros_service', 'data': event_data})

    async def _on_tcp_received(self, event_data: Dict[str, Any]) -> None:
        """TCP 수신 이벤트를 GUI로 전달한다."""
        await self._bridge.publish_async({'type': 'tcp_event', 'data': {'topic': EventTopic.TCP_MESSAGE_RECEIVED.value, 'data': event_data}})

    async def _on_tcp_sent(self, event_data: Dict[str, Any]) -> None:
        """TCP 송신 이벤트를 GUI로 전달한다."""
        await self._bridge.publish_async({'type': 'tcp_event', 'data': {'topic': EventTopic.TCP_MESSAGE_SENT.value, 'data': event_data}})
