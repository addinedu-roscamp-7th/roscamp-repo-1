from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

EventHandler = Callable[[Dict[str, object]], Awaitable[None]]


class EventBus:
    """Lightweight pub/sub bus for cross-module async events."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventHandler]] = defaultdict(list)
        self._loop = asyncio.get_event_loop()

    def register_listener(self, topic: str, handler: EventHandler) -> None:
        self._listeners[topic].append(handler)
        logger.debug("Listener registered for %s (%d total)", topic, len(self._listeners[topic]))

    def unregister_listener(self, topic: str, handler: EventHandler) -> None:
        if topic in self._listeners:
            self._listeners[topic] = [h for h in self._listeners[topic] if h != handler]

    async def publish(self, topic: str, payload: Dict[str, object]) -> None:
        handlers = list(self._listeners.get(topic, []))
        for handler in handlers:
            self._loop.create_task(self._wrap_handler(handler, payload))

    async def _wrap_handler(self, handler: EventHandler, payload: Dict[str, object]) -> None:
        try:
            await handler(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Event handler failed for payload=%s", payload)
