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
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._config = config if config is not None else MainServiceConfig()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """수신 스레드를 종료한다."""
        self._stop_event.set()
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
                    conn.settimeout(self._config.timeout)
                    self.reconnected.emit()
                    backoff_seconds = 1.0
                    self._listen_loop(conn)
            except OSError as exc:
                if self._stop_event.is_set():
                    break
                self.connection_error.emit(str(exc))
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 10.0)

    def _listen_loop(self, conn: socket.socket) -> None:
        """연결된 소켓에서 JSON 라인 단위로 알림을 읽는다."""
        buffer = b''
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
