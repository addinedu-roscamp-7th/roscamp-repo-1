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
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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
        
        try:
            # 연결이 유지되는 동안 메시지 처리
            while not reader.at_eof():
                # 한 줄 읽기 (JSON + \n)
                line = await reader.readline()
                if not line:
                    break
                
                # 요청 처리 및 응답 생성
                response = await self._dispatch(line.decode())
                
                # 응답 전송 (JSON + \n)
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
                
        except asyncio.CancelledError:
            raise  # 정상 종료
        except Exception as exc:  # noqa: BLE001
            logger.exception("Client error: %s", exc)
        finally:
            # 연결 종료
            writer.close()
            await writer.wait_closed()
            logger.info("Closed connection from %s", peer)

    async def _dispatch(self, raw_payload: str) -> Dict[str, Any]:
        """
        메시지 디스패칭
        
        1. JSON 파싱
        2. 메시지 타입 확인
        3. 적절한 핸들러 호출
        4. 에러 핸들링
        
        Args:
            raw_payload: JSON 문자열
            
        Returns:
            dict: 응답 메시지 (항상 type, result, message 포함)
        """
        # JSON 파싱
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return {
                "type": "error",
                "result": False,
                "error_code": "SYS_001",  # 시스템 오류
                "message": "invalid_json"
            }

        # 메시지 타입 확인
        msg_type = payload.get("type")
        handler = self._handlers.get(msg_type)
        
        if not handler:
            return {
                "type": msg_type or "unknown",
                "result": False,
                "error_code": "SYS_001",  # 미지원 메시지
                "message": "unsupported_message_type"
            }

        # 핸들러 실행
        try:
            return await handler(payload.get("data") or {})
        except Exception:  # noqa: BLE001
            logger.exception("Handler failure for %s", msg_type)
            return {
                "type": msg_type,
                "result": False,
                "error_code": "SYS_001",  # 내부 오류
                "message": "internal_error"
            }

    async def _handle_push_event(self, message: Dict[str, Any]) -> None:
        """
        푸시 알림 처리 (Placeholder)
        
        EventBus에서 발행한 "app_push" 이벤트를 받아
        연결된 클라이언트들에게 전송합니다.
        
        TODO: 클라이언트 레지스트리 구현
            - user_id별 연결 관리
            - 특정 사용자에게만 알림 전송
            - 브로드캐스트 지원
        """
        # TODO: maintain client registry and broadcast events.
        logger.debug("Push event queued: %s", message)
