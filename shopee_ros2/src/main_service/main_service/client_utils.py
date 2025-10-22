"""
Main Service TCP 클라이언트 유틸리티

시나리오 스크립트와 테스트 클라이언트에서 재사용할 수 있는
비동기 TCP 클라이언트 구현을 제공합니다.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional


class MainServiceClient:
    """Shopee Main Service와 TCP로 통신하기 위한 비동기 클라이언트."""

    def __init__(self, host: str = 'localhost', port: int = 5000) -> None:
        self._host = host
        self._port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._async_notifications: list[Dict[str, Any]] = []
        self._pending_messages: list[Dict[str, Any]] = []
        self._last_notification_signature: Optional[str] = None
        self._last_notification_type: Optional[str] = None
        self._repeat_count: int = 0
        self._silenced_types = {
            'robot_moving_notification',
            'work_info_notification',
        }
        self._silenced_counts: Dict[str, int] = {}

    async def connect(self) -> None:
        """Main Service에 TCP 연결을 생성합니다."""
        self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        print(f'✓ Connected to {self._host}:{self._port}')

    def _is_async_notification(self, msg_type: str) -> bool:
        """비동기 알림 메시지 타입 여부를 판별합니다."""
        async_types = {
            'robot_moving_notification',
            'robot_arrived_notification',
            'product_selection_start',
            'cart_update_notification',
            'work_info_notification',
            'packing_info_notification',
            'picking_complete_notification',
            'shopping_summary_notification',
            'robot_reassignment_notification',
            'robot_failure_notification',
            'order_failed_notification',
        }
        return msg_type in async_types

    def _notification_signature(self, response: Dict[str, Any]) -> str:
        """알림 고유 서명을 생성한다."""
        data = response.get('data')
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
        return f"{response.get('type','')}|{serialized}"

    def _flush_notification_summary(self) -> None:
        """중복된 알림이 있었다면 요약 메시지를 출력하고 상태를 초기화한다."""
        if self._repeat_count > 1 and self._last_notification_type:
            print(f"    ↳ (same '{self._last_notification_type}' repeated {self._repeat_count - 1} more times)")
        self._repeat_count = 0
        self._last_notification_signature = None
        self._last_notification_type = None
        self._flush_silenced_summary()

    def _flush_silenced_summary(self) -> None:
        """누적된 묵음 알림 요약을 출력한다."""
        if not self._silenced_counts:
            return
        summary = ', '.join(f"{msg_type}: {count}" for msg_type, count in self._silenced_counts.items())
        print(f"    ↳ suppressed notifications -> {summary}")
        self._silenced_counts.clear()

    def _print_notification(self, response: Dict[str, Any]) -> None:
        """비동기 알림 메시지를 콘솔에 출력합니다."""
        msg_type = response.get('type')
        if msg_type in self._silenced_types:
            self._silenced_counts[msg_type] = self._silenced_counts.get(msg_type, 0) + 1
            return

        signature = self._notification_signature(response)
        if signature == self._last_notification_signature:
            self._repeat_count += 1
            return

        if self._last_notification_signature is not None:
            self._flush_notification_summary()

        print(f'\n  📢 [Async Notification] {response["type"]}')
        if response.get('message'):
            print(f'     Message: {response["message"]}')
        if response.get('data'):
            formatted = json.dumps(response['data'], ensure_ascii=False, indent=2)
            print(f'     Data: {formatted}')
        self._last_notification_signature = signature
        self._last_notification_type = response.get('type')
        self._repeat_count = 1

    async def send_request(self, msg_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        지정한 메시지 타입으로 요청을 전송하고 응답을 반환합니다.

        Args:
            msg_type: 전송할 메시지 타입
            data: 메시지 페이로드

        Returns:
            Main Service가 반환한 응답 메시지
        """
        if not self._reader or not self._writer:
            raise RuntimeError('Client is not connected. Call connect() first.')

        request = {'type': msg_type, 'data': data}
        request_json = json.dumps(request) + '\n'
        self._writer.write(request_json.encode())
        await self._writer.drain()

        print(f'\n→ Sent: {msg_type}')
        print(f'  Data: {json.dumps(data, ensure_ascii=False)}')

        expected_response = f'{msg_type}_response'
        max_attempts = 20

        for attempt in range(max_attempts):
            response: Optional[Dict[str, Any]] = None
            if self._pending_messages:
                response = self._pending_messages.pop(0)
            else:
                try:
                    from .constants import TCP_READ_TIMEOUT

                    response_line = await asyncio.wait_for(self._reader.readline(), timeout=TCP_READ_TIMEOUT)
                    if not response_line:
                        print('  ⚠️ Connection closed by server')
                        break
                    response = json.loads(response_line.decode())
                except asyncio.TimeoutError:
                    print(f'  ⚠️ Timeout waiting for {expected_response}')
                    break
                except json.JSONDecodeError:
                    print('  ⚠️ Invalid JSON received from server')
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f'  ⚠️ Error reading response: {exc}')
                    break

            if response is None:
                continue

            if self._is_async_notification(response['type']):
                self._print_notification(response)
                self._async_notifications.append(response)
                continue

            if response['type'] == expected_response:
                print(f'← Received: {response["type"]}')
                print(f'  Result: {response["result"]}')
                if response.get('message'):
                    print(f'  Message: {response["message"]}')
                if response.get('data'):
                    formatted = json.dumps(response['data'], ensure_ascii=False, indent=2)
                    print(f'  Data: {formatted}')
                return response

            print(f'  ⚠️ Unexpected response: {response["type"]} (expected: {expected_response})')
            return response

        print(f'  ❌ Failed to receive {expected_response}')
        return {'type': 'error', 'result': False, 'message': 'No response received'}

    async def close(self) -> None:
        """TCP 연결을 종료합니다."""
        self._flush_notification_summary()
        self._flush_notification_summary()
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        print('\n✓ Connection closed')

    @property
    def async_notifications(self) -> list[Dict[str, Any]]:
        """수신된 비동기 알림 목록을 반환합니다."""
        return list(self._async_notifications)

    async def drain_notifications(self, timeout: float = 1.0, expected_types: Optional[set[str]] = None) -> list[Dict[str, Any]]:
        """
        지정된 시간 동안 비동기 알림을 수집합니다.

        Args:
            timeout: 최대 대기 시간(초)
            expected_types: 기다리는 알림 타입 집합

        Returns:
            수집된 알림 목록
        """
        if not self._reader:
            raise RuntimeError('Client is not connected. Call connect() first.')

        collected: list[Dict[str, Any]] = []
        remaining_types = set(expected_types) if expected_types else None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # 비동기 알림이 이미 수신되어 있는 경우 먼저 확인
        if remaining_types:
            already_collected = [note for note in self._async_notifications if note['type'] in remaining_types]
            for note in already_collected:
                collected.append(note)
                remaining_types.discard(note['type'])
            if not remaining_types:
                return collected

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break

            try:
                response_line = await asyncio.wait_for(self._reader.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            if not response_line:
                break

            try:
                response = json.loads(response_line.decode())
            except json.JSONDecodeError:
                print('  ⚠️ Invalid JSON in notification stream')
                continue

            if self._is_async_notification(response['type']):
                self._print_notification(response)
                self._async_notifications.append(response)
                collected.append(response)
                if remaining_types:
                    remaining_types.discard(response['type'])
                    if not remaining_types:
                        break
                continue

            # 요청 응답이 들어온 경우 다음 send_request 호출에서 처리
            self._pending_messages.append(response)

        return collected
