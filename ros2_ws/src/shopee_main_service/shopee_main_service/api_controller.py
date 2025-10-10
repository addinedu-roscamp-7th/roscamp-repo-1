from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class APIController:
    """Async TCP entry point that routes App traffic to domain services."""

    def __init__(
        self,
        host: str,
        port: int,
        handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
        event_bus: "EventBus",
    ) -> None:
        self._host = host
        self._port = port
        self._handlers = handlers
        self._event_bus = event_bus
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        """Start TCP server and subscribe to outbound events."""
        self._event_bus.register_listener("app_push", self._handle_push_event)
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        addr = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        logger.info("APIController listening on %s", addr)

    async def stop(self) -> None:
        """Shutdown server and drop event subscriptions."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._event_bus.unregister_listener("app_push", self._handle_push_event)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("Accepted connection from %s", peer)
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                response = await self._dispatch(line.decode())
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Client error: {exc}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Closed connection from %s", peer)

    async def _dispatch(self, raw_payload: str) -> Dict[str, Any]:
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return {"type": "error", "result": False, "error_code": "SYS_001", "message": "invalid_json"}

        msg_type = payload.get("type")
        handler = self._handlers.get(msg_type)
        if not handler:
            return {"type": msg_type or "unknown", "result": False, "error_code": "SYS_002", "message": "unsupported"}

        try:
            return await handler(payload.get("data") or {})
        except Exception:  # noqa: BLE001
            logger.exception("Handler failure for %s", msg_type)
            return {"type": msg_type, "result": False, "error_code": "SYS_500", "message": "internal_error"}

    async def _handle_push_event(self, message: Dict[str, Any]) -> None:
        """Placeholder: attach to client sessions and push notifications."""
        # TODO: maintain client registry and broadcast events.
        logger.debug("Push event queued: %s", message)
