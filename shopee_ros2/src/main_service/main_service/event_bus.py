"""
내부 이벤트 버스

모듈 간 비동기 이벤트 전달을 위한 경량 Pub/Sub 시스템입니다.
- 토픽 기반 구독/발행
- 비동기 처리
- 에러 격리 (한 핸들러 실패해도 다른 핸들러 실행)
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

# 이벤트 핸들러 타입: 페이로드를 받아 비동기 처리
EventHandler = Callable[[Dict[str, object]], Awaitable[None]]


class EventBus:
    """
    경량 Pub/Sub 이벤트 버스
    
    모듈 간 느슨한 결합을 위한 이벤트 기반 통신을 제공합니다.
    
    사용 예:
        # 구독
        event_bus.register_listener("order_created", handle_order)
        
        # 발행
        await event_bus.publish("order_created", {"order_id": 123})
    """

    def __init__(self) -> None:
        # 토픽별 핸들러 리스트
        self._listeners: Dict[str, List[EventHandler]] = defaultdict(list)

    def register_listener(self, topic: str, handler: EventHandler) -> None:
        """
        토픽 구독을 등록합니다.

        동일한 토픽에 여러 핸들러를 등록할 수 있으며,
        이벤트 발행 시 모든 핸들러가 병렬로 실행됩니다.

        Args:
            topic: 구독할 토픽명 (예: "app_push", "order_created")
            handler: 이벤트 처리 함수 (async). Dict[str, object]를 인자로 받아 None 반환

        사용 예:
            async def on_order_created(payload):
                order_id = payload['order_id']
                print(f"주문 생성됨: {order_id}")

            event_bus.register_listener("order_created", on_order_created)
        """
        self._listeners[topic].append(handler)
        logger.debug("Listener registered for %s (%d total)", topic, len(self._listeners[topic]))

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """
        토픽 구독을 등록합니다. (register_listener의 별칭)

        기존 코드 호환성을 위해 제공되는 메서드입니다.
        내부적으로 register_listener를 호출합니다.

        Args:
            topic: 구독할 토픽명
            handler: 이벤트 처리 함수 (async)

        사용 예:
            event_bus.subscribe("robot_status_changed", handle_status_change)
        """
        self.register_listener(topic, handler)

    def unregister_listener(self, topic: str, handler: EventHandler) -> None:
        """
        토픽 구독을 해제합니다.

        등록되지 않은 핸들러를 해제하려고 해도 에러가 발생하지 않습니다.

        Args:
            topic: 구독 해제할 토픽명
            handler: 등록했던 핸들러 함수 (동일한 함수 객체여야 함)

        사용 예:
            # 구독 등록
            event_bus.register_listener("order_created", handle_order)

            # 구독 해제
            event_bus.unregister_listener("order_created", handle_order)
        """
        if topic in self._listeners:
            self._listeners[topic] = [h for h in self._listeners[topic] if h != handler]

    async def publish(self, topic: str, payload: Dict[str, object]) -> None:
        """
        이벤트를 발행합니다.

        등록된 모든 핸들러를 병렬로 실행합니다.
        한 핸들러가 예외를 발생시켜도 다른 핸들러는 계속 실행되며,
        에러는 로그로 기록됩니다.

        Args:
            topic: 발행할 토픽명
            payload: 전달할 데이터 (dict). 핸들러에게 그대로 전달됨

        사용 예:
            # 단순 이벤트 발행
            await event_bus.publish("order_created", {"order_id": 123, "user_id": "user1"})

            # 로봇 상태 변경 이벤트
            await event_bus.publish("robot_status_changed", {
                "robot_id": 1,
                "old_status": "IDLE",
                "new_status": "BUSY"
            })
        """
        handlers = list(self._listeners.get(topic, []))

        # 각 핸들러를 별도 태스크로 실행 (병렬)
        loop = asyncio.get_running_loop()
        for handler in handlers:
            loop.create_task(self._wrap_handler(handler, payload))

    async def _wrap_handler(self, handler: EventHandler, payload: Dict[str, object]) -> None:
        """
        핸들러 실행 래퍼 (에러 격리용 내부 메서드)

        한 핸들러의 예외가 다른 핸들러에 영향을 주지 않도록 격리합니다.
        예외 발생 시 스택 트레이스와 함께 에러를 로그로 기록합니다.

        Args:
            handler: 실행할 이벤트 핸들러
            payload: 핸들러에게 전달할 데이터
        """
        try:
            await handler(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Event handler failed for payload=%s", payload)
