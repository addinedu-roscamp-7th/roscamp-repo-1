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
        토픽 구독 등록
        
        Args:
            topic: 구독할 토픽명 (예: "app_push", "order_created")
            handler: 이벤트 처리 함수 (async)
        """
        self._listeners[topic].append(handler)
        logger.debug("Listener registered for %s (%d total)", topic, len(self._listeners[topic]))

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """
        register_listener의 alias.
        기존 코드 호환을 위해 제공됩니다.
        """
        self.register_listener(topic, handler)

    def unregister_listener(self, topic: str, handler: EventHandler) -> None:
        """
        토픽 구독 해제
        
        Args:
            topic: 구독 해제할 토픽명
            handler: 등록했던 핸들러 함수
        """
        if topic in self._listeners:
            self._listeners[topic] = [h for h in self._listeners[topic] if h != handler]

    async def publish(self, topic: str, payload: Dict[str, object]) -> None:
        """
        이벤트 발행
        
        등록된 모든 핸들러를 비동기로 실행합니다.
        한 핸들러가 실패해도 다른 핸들러는 계속 실행됩니다.
        
        Args:
            topic: 발행할 토픽명
            payload: 전달할 데이터 (dict)
        """
        handlers = list(self._listeners.get(topic, []))
        
        # 각 핸들러를 별도 태스크로 실행 (병렬)
        loop = asyncio.get_running_loop()
        for handler in handlers:
            loop.create_task(self._wrap_handler(handler, payload))

    async def _wrap_handler(self, handler: EventHandler, payload: Dict[str, object]) -> None:
        """
        핸들러 실행 래퍼 (에러 격리)
        
        한 핸들러의 예외가 다른 핸들러에 영향을 주지 않도록 합니다.
        """
        try:
            await handler(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Event handler failed for payload=%s", payload)
