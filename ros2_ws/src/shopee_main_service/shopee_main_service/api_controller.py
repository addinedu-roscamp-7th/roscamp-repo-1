"""
TCP API 컨트롤러

Shopee App과의 TCP 통신을 담당합니다.
- JSON 형식의 요청/응답 처리
- 메시지 타입별 핸들러 라우팅
- 이벤트 푸시 (알림)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from .event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class _ClientSession:
    """TCP 클라이언트 연결 정보를 보관"""
    writer: asyncio.StreamWriter
    peer: Tuple[str, int]
    lock: asyncio.Lock
    user_id: Optional[str] = None


class APIController:
    """
    비동기 TCP API 서버
    
    App에서 오는 TCP 연결을 받아 JSON 메시지를 처리합니다.
    - 포트 5000에서 대기
    - 각 메시지 타입을 적절한 핸들러로 라우팅
    - EventBus에서 오는 알림을 클라이언트로 푸시
    """

    def __init__(
        self,
        host: str,
        port: int,
        handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
        event_bus: "EventBus",
    ) -> None:
        """
        Args:
            host: 바인딩할 IP (보통 "0.0.0.0")
            port: 포트 번호 (5000)
            handlers: 메시지 타입 → 핸들러 함수 매핑
            event_bus: 내부 이벤트 버스 (푸시 알림용)
        """
        self._host = host
        self._port = port
        self._handlers = handlers  # 외부에서 주입받은 핸들러 딕셔너리
        self._event_bus = event_bus
        self._server: Optional[asyncio.AbstractServer] = None
        self._clients: Dict[asyncio.StreamWriter, _ClientSession] = {}

    async def start(self) -> None:
        """
        TCP 서버 시작
        
        1. EventBus의 "app_push" 토픽 구독
        2. TCP 서버 소켓 오픈
        """
        self._event_bus.register_listener("app_push", self._handle_push_event)
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        addr = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        logger.info("APIController listening on %s", addr)

    async def stop(self) -> None:
        """
        서버 종료 및 정리
        
        1. TCP 서버 소켓 닫기
        2. EventBus 구독 해제
        """
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._event_bus.unregister_listener("app_push", self._handle_push_event)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        클라이언트 연결 처리
        
        한 클라이언트와의 세션을 관리합니다.
        - 연결 유지하며 메시지 송수신
        - 라인 단위로 JSON 메시지 읽기/쓰기
        """
        peer = writer.get_extra_info("peername")
        logger.info("Accepted connection from %s", peer)
        self._register_client(writer, peer)
        
        try:
            # 연결이 유지되는 동안 메시지 처리
            while not reader.at_eof():
                # 한 줄 읽기 (JSON + \n)
                line = await reader.readline()
                if not line:
                    break
                
                # 요청 처리 및 응답 생성
                response = await self._dispatch(line.decode(), peer)
                try:
                    await self._send_to_client(writer, response)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to send response to %s: %s", peer, exc)
                    self._unregister_client(writer)
                    break
                
        except asyncio.CancelledError:
            raise  # 정상 종료
        except Exception as exc:  # noqa: BLE001
            logger.exception("Client error: %s", exc)
        finally:
            # 연결 종료
            writer.close()
            await writer.wait_closed()
            self._unregister_client(writer)
            logger.info("Closed connection from %s", peer)

    async def _dispatch(self, raw_payload: str, peer: Tuple[str, int]) -> Dict[str, Any]:
        """
        메시지 디스패칭
        
        1. JSON 파싱
        2. 메시지 타입 확인
        3. 적절한 핸들러 호출 (클라이언트 정보 전달)
        4. 에러 핸들링
        
        Args:
            raw_payload: JSON 문자열
            peer: 요청 클라이언트의 (ip, port) 튜플
            
        Returns:
            dict: 응답 메시지 (항상 type, result, message 포함)
        """
        # JSON 파싱
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from %s: %s", peer, raw_payload[:100])
            return {
                "type": "error",
                "result": False,
                "error_code": "SYS_001",  # 시스템 오류
                "message": "invalid_json"
            }

        # 메시지 타입 확인
        msg_type = payload.get("type")
        data = payload.get("data") or {}
        
        # 요청 로그
        logger.info("→ Received [%s] from %s: %s", msg_type, peer, json.dumps(data, ensure_ascii=False)[:200])
        
        handler = self._handlers.get(msg_type)
        
        if not handler:
            logger.warning("Unsupported message type: %s", msg_type)
            return {
                "type": msg_type or "unknown",
                "result": False,
                "error_code": "SYS_001",  # 미지원 메시지
                "message": "unsupported_message_type"
            }

        # 핸들러 실행
        try:
            start_time = time.perf_counter()
            response = await handler(data, peer)
            elapsed = (time.perf_counter() - start_time) * 1000  # ms

            enriched_response = self._with_metadata(response)

            # 응답 로그
            logger.info(
                "← Sending [%s] result=%s (%.1fms, id=%s): %s",
                enriched_response.get("type", "unknown"),
                enriched_response.get("result", False),
                elapsed,
                enriched_response.get("correlation_id"),
                enriched_response.get("message", ""),
            )

            return enriched_response
        except Exception:  # noqa: BLE001
            logger.exception("Handler failure for %s", msg_type)
            return {
                "type": msg_type,
                "result": False,
                "error_code": "SYS_001",  # 내부 오류
                "message": "internal_error"
            }

    def _register_client(self, writer: asyncio.StreamWriter, peer: Tuple[str, int]) -> None:
        """새 TCP 클라이언트를 레지스트리에 등록"""
        if writer in self._clients:
            return
        self._clients[writer] = _ClientSession(writer=writer, peer=peer, lock=asyncio.Lock())
        logger.debug("Client registered: %s (total=%d)", peer, len(self._clients))

    def _unregister_client(self, writer: asyncio.StreamWriter) -> None:
        """TCP 클라이언트를 레지스트리에서 제거"""
        session = self._clients.pop(writer, None)
        if session:
            logger.debug("Client unregistered: %s (total=%d)", session.peer, len(self._clients))

    def associate_peer_with_user(self, peer: Optional[Tuple[str, int]], user_id: str) -> None:
        """
        특정 peer(연결)와 사용자 ID를 매핑합니다.

        로그인 성공 시 호출되어, 이후 push 이벤트를 사용자별로 필터링할 수 있습니다.
        """
        if peer is None:
            return
        for session in self._clients.values():
            if session.peer == peer:
                session.user_id = user_id
                logger.debug("Associated peer %s with user %s", peer, user_id)
                break

    async def _send_to_client(self, writer: asyncio.StreamWriter, payload: Any) -> None:
        """
        클라이언트에게 메시지를 전송 (응답/푸시 공용)
        - JSON 직렬화 및 개행 처리
        - 동시 전송 시 Lock으로 순서 보장
        """
        session = self._clients.get(writer)
        if not session:
            raise ConnectionError("Client session not found.")

        if session.writer.is_closing():
            raise ConnectionError("Client connection already closing.")

        if isinstance(payload, (bytes, bytearray)):
            serialized = bytes(payload)
        elif isinstance(payload, str):
            serialized = (payload if payload.endswith("\n") else payload + "\n").encode()
        else:
            serialized = (json.dumps(payload) + "\n").encode()

        try:
            async with session.lock:
                session.writer.write(serialized)
                await session.writer.drain()
        except Exception:
            raise

    async def _handle_push_event(self, message: Dict[str, Any]) -> None:
        """
        EventBus에서 발행한 "app_push" 이벤트를 받아 연결된 클라이언트에 전송합니다.

        payload에 `target_user_id` 또는 `target_user_ids`가 지정되면 해당 사용자에게만
        푸시가 전달되고, 지정되지 않은 경우 모든 세션으로 브로드캐스트합니다.
        """
        if not self._clients:
            logger.debug("No clients to push message: %s", message)
            return

        target_ids = set()
        if isinstance(message.get("target_user_ids"), list):
            target_ids.update(str(uid) for uid in message["target_user_ids"])
        if message.get("target_user_id") is not None:
            target_ids.add(str(message["target_user_id"]))

        recipients = []
        for writer, session in list(self._clients.items()):
            if target_ids:
                if session.user_id and str(session.user_id) in target_ids:
                    recipients.append(writer)
            else:
                recipients.append(writer)

        if not recipients:
            logger.debug("No matching recipients for push event: %s", message)
            return

        logger.debug("Broadcasting push event to %d clients: %s", len(recipients), message)
        base_correlation = message.get("correlation_id") or uuid.uuid4().hex
        base_timestamp = message.get("timestamp") or int(time.time() * 1000)

        for writer in recipients:
            enriched_message = self._with_metadata(message, base_correlation, base_timestamp)
            await self._send_with_retry(writer, enriched_message)

    async def _send_with_retry(
        self,
        writer: asyncio.StreamWriter,
        payload: Dict[str, Any],
        *,
        max_attempts: int = 3,
        base_delay: float = 0.1,
    ) -> bool:
        """지수 백오프를 적용한 전송 재시도 로직."""
        attempt = 1
        delay = base_delay
        while attempt <= max_attempts:
            try:
                await self._send_to_client(writer, payload)
                return True
            except Exception as exc:  # noqa: BLE001
                if attempt == max_attempts:
                    logger.warning(
                        "Failed to deliver push after %d attempts: %s",
                        attempt,
                        exc,
                    )
                    self._unregister_client(writer)
                    return False
                await asyncio.sleep(delay)
                delay *= 2
                attempt += 1
        return False

    def _with_metadata(
        self,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        응답/이벤트 페이로드에 상관관계 ID와 타임스탬프를 부여합니다.
        기존 값을 덮어쓰지 않으며 새 딕셔너리를 반환합니다.
        """
        enriched = dict(payload)
        if "correlation_id" not in enriched or correlation_id is not None:
            enriched["correlation_id"] = correlation_id or enriched.get("correlation_id") or uuid.uuid4().hex
        if "timestamp" not in enriched or timestamp_ms is not None:
            enriched["timestamp"] = timestamp_ms or enriched.get("timestamp") or int(time.time() * 1000)
        return enriched
