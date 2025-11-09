from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

from PyQt6 import QtCore

from shopee_app.services.main_service_client import MainServiceConfig


class AppNotificationClient(QtCore.QThread):
    """Main Service로부터 비동기 알림을 수신하는 전용 스레드."""

    notification_received = QtCore.pyqtSignal(dict)
    connection_error = QtCore.pyqtSignal(str)
    reconnected = QtCore.pyqtSignal()

    def __init__(
        self,
        config: Optional[MainServiceConfig] = None,
        *,
        user_id: str = "",
        password: str | None = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._config = config if config is not None else MainServiceConfig()
        # 알림 전용 소켓도 사용자/비밀번호를 기억해 재인증한다.
        self._user_id = user_id
        self._password = password or ""
        self._stop_event = threading.Event()
        self._socket_lock = threading.Lock()
        self._current_socket: Optional[socket.socket] = None

    def stop(self) -> None:
        """수신 스레드를 종료한다."""
        self._stop_event.set()
        with self._socket_lock:
            current_socket = self._current_socket
            self._current_socket = None
        if current_socket is not None:
            try:
                current_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                current_socket.close()
            except OSError:
                pass
        self.wait()

    def run(self) -> None:
        """TCP 연결을 유지하며 알림을 수신한다."""
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self._config.host, self._config.port),
                    timeout=self._config.timeout,
                ) as conn:
                    with self._socket_lock:
                        self._current_socket = conn
                    conn.settimeout(self._config.timeout)
                    # 소켓을 연 직후 로그인 메시지를 전송해 사용자와 연결을 매핑한다.
                    pending_buffer = b""
                    success, pending_buffer = self._authenticate(conn)
                    if not success:
                        raise ConnectionError('알림 인증에 실패했습니다.')
                    conn.settimeout(max(1.0, self._config.timeout))
                    self.reconnected.emit()
                    backoff_seconds = 1.0
                    self._listen_loop(conn, pending_buffer)
            except OSError as exc:
                if self._stop_event.is_set():
                    break
                self.connection_error.emit(str(exc))
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 10.0)
            finally:
                with self._socket_lock:
                    self._current_socket = None

    def _listen_loop(self, conn: socket.socket, initial_buffer: bytes = b"") -> None:
        """연결된 소켓에서 JSON 라인 단위로 알림을 읽는다."""
        buffer = initial_buffer
        while not self._stop_event.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                self.connection_error.emit(str(exc))
                break

            if not chunk:
                self.connection_error.emit('서버와의 연결이 종료되었습니다.')
                break

            buffer += chunk
            while b'\n' in buffer:
                raw_line, buffer = buffer.split(b'\n', 1)
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    message = json.loads(raw_line.decode('utf-8'))
                except json.JSONDecodeError as exc:
                    self.connection_error.emit(f'잘못된 JSON 수신: {exc}')
                    continue
                self.notification_received.emit(message)

    def update_credentials(self, user_id: str, password: str | None) -> None:
        """인증에 사용할 사용자 정보를 갱신한다."""
        self._user_id = user_id
        self._password = password or ""

    def _authenticate(self, conn: socket.socket) -> tuple[bool, bytes]:
        """알림 전용 연결에 사용자 인증을 수행한다."""
        if not self._user_id or not self._password:
            # 자격 증명이 없으면 기존 동작과 동일하게 즉시 수신으로 넘어간다.
            return True, b''

        request = {
            'type': 'user_login',
            'data': {
                'user_id': self._user_id,
                'password': self._password,
            },
        }
        serialized = json.dumps(request, ensure_ascii=False) + '\n'
        try:
            conn.sendall(serialized.encode('utf-8'))
        except OSError:
            return False, b''

        buffer = b''
        deadline = time.monotonic() + self._config.timeout
        while time.monotonic() < deadline and not self._stop_event.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return False, b''

            if not chunk:
                return False, b''

            buffer += chunk
            while b'\n' in buffer:
                raw_line, remainder = buffer.split(b'\n', 1)
                line = raw_line.strip()
                if not line:
                    buffer = remainder
                    continue
                try:
                    response = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    buffer = remainder
                    continue
                if response.get('type') == 'user_login_response':
                    if not response.get('result'):
                        return False, remainder
                    return True, remainder
                # 예상하지 않은 메시지는 버퍼에 되돌려 _listen_loop가 처리하도록 한다.
                buffer = raw_line + b'\n' + remainder
                return True, buffer
        return False, buffer
