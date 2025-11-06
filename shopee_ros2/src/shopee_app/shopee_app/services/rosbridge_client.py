import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PyQt6 import QtCore

import websocket


@dataclass
class RosbridgeConfig:
    '''rosbridge 서버 접속 정보와 구독 토픽 설정.'''

    host: str
    port: int
    topic: str
    queue_length: int = 1
    retry_interval: float = 3.0

    @property
    def url(self) -> str:
        '''WebSocket 접속 URL을 구성한다.'''
        return f"ws://{self.host}:{self.port}"


class RosbridgePoseSubscriber(QtCore.QObject):
    """rosbridge WebSocket 구독자로부터 Pose 메시지를 수신한다."""

    pose_received = QtCore.pyqtSignal(dict)
    connection_error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        config: RosbridgeConfig,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._thread: Optional[threading.Thread] = None
        self._ws_app: Optional[websocket.WebSocketApp] = None
        self._should_run = False
        self._lock = threading.Lock()

    def start(self) -> None:
        '''백그라운드 스레드를 시작해 rosbridge와 연결한다.'''
        with self._lock:
            if self._thread is not None:
                return
            self._should_run = True
            self._thread = threading.Thread(
                target=self._run_loop,
                name="RosbridgePoseSubscriber",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        '''연결을 종료하고 백그라운드 스레드를 정리한다.'''
        with self._lock:
            self._should_run = False
            if self._ws_app is not None:
                try:
                    self._ws_app.close()
                except Exception:
                    pass
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

    # 내부 구현 ---------------------------------------------------------------

    def _run_loop(self) -> None:
        '''연결 유지와 재시도를 관리하는 루프를 실행한다.'''
        while True:
            with self._lock:
                if not self._should_run:
                    break

            try:
                self._ws_app = websocket.WebSocketApp(
                    self._config.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws_app.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    reconnect=0,
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.connection_error.emit(str(exc))

            with self._lock:
                if not self._should_run:
                    break
            time.sleep(self._config.retry_interval)

    def _on_open(self, ws_app: websocket.WebSocketApp) -> None:  # pragma: no cover
        '''연결이 열리면 즉시 구독 요청을 전송한다.'''
        subscribe_payload = {
            "op": "subscribe",
            "topic": self._config.topic,
            "queue_length": self._config.queue_length,
        }
        ws_app.send(json.dumps(subscribe_payload))

    def _on_message(self, _ws_app: websocket.WebSocketApp, message: str) -> None:
        '''pose 메시지를 파싱해 필요한 경우 시그널을 발생시킨다.'''
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if payload.get("op") != "publish":
            return
        if payload.get("topic") != self._config.topic:
            return
        msg = payload.get("msg")
        if isinstance(msg, dict):
            print(f'[Map] rosbridge 메시지 수신: {msg}')
            self.pose_received.emit(msg)

    def _on_error(self, _ws_app: websocket.WebSocketApp, error: Exception) -> None:
        '''웹소켓 오류를 문자열로 변환해 UI에 전달한다.'''
        self.connection_error.emit(str(error))

    def _on_close(
        self,
        _ws_app: websocket.WebSocketApp,
        _close_status_code,
        _close_msg,
    ) -> None:  # pragma: no cover
        '''rosbridge에서 연결을 끊을 때 호출된다.'''
        pass
