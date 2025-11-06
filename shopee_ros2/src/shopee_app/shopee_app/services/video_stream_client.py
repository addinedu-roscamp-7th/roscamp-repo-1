import json
import socket
import time
from dataclasses import dataclass

from PyQt6 import QtCore
from PyQt6 import QtGui


# UDP 헤더와 청크 구조는 Main Service 명세를 따른다.
HEADER_SIZE = 200
CHUNK_DATA_SIZE = 1400
DEFAULT_UDP_PORT = 6000
FRAME_TIMEOUT_SEC = 2.0


@dataclass
class _FrameBuffer:
    '''한 프레임을 이루는 여러 청크를 조립하기 위한 임시 버퍼.'''

    total_chunks: int
    chunks: list[bytes | None]
    received: int
    last_updated: float


class VideoStreamReceiver(QtCore.QThread):
    """Main Service로부터 UDP 영상 스트림을 수신해 QPixmap으로 변환한다."""

    frame_received = QtCore.pyqtSignal(str, QtGui.QImage)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        robot_id: int,
        camera_type: str,
        port: int = DEFAULT_UDP_PORT,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._robot_id = int(robot_id)
        self._camera_type = camera_type
        self._port = int(port)
        self._running = False
        self._sock: socket.socket | None = None

    def run(self) -> None:
        '''UDP 소켓을 열고 청크를 수신해 한 프레임으로 합친다.'''
        self._running = True
        buffers: dict[int, _FrameBuffer] = {}
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self._port))
            sock.settimeout(1.0)
            self._sock = sock
        except OSError as exc:
            self.error_occurred.emit(f"UDP 소켓을 열 수 없습니다: {exc}")
            self._running = False
            return

        try:
            last_cleanup = time.monotonic()
            while self._running:
                try:
                    packet, _ = self._sock.recvfrom(HEADER_SIZE + CHUNK_DATA_SIZE)
                except socket.timeout:
                    now = time.monotonic()
                    if now - last_cleanup > 1.0:
                        self._cleanup_buffers(buffers, now)
                        last_cleanup = now
                    continue
                except OSError as exc:
                    if self._running:
                        self.error_occurred.emit(f"UDP 수신 오류가 발생했습니다: {exc}")
                    break

                header_raw = packet[:HEADER_SIZE]
                try:
                    header_text = header_raw.decode("utf-8", errors="ignore").rstrip("\x00 ")
                except UnicodeDecodeError:
                    continue
                if not header_text:
                    continue
                try:
                    header = json.loads(header_text)
                except json.JSONDecodeError:
                    continue

                if header.get("type") != "video_frame":
                    continue
                if int(header.get("robot_id", -1)) != self._robot_id:
                    continue

                frame_id = int(header.get("frame_id", -1))
                total_chunks = int(header.get("total_chunks", 0))
                chunk_idx = int(header.get("chunk_idx", -1))
                data_size = int(header.get("data_size", 0))
                if frame_id < 0 or total_chunks <= 0 or chunk_idx < 0:
                    continue
                if data_size <= 0:
                    continue

                chunk_payload = packet[HEADER_SIZE : HEADER_SIZE + data_size]
                if len(chunk_payload) != data_size:
                    continue

                buffer = buffers.get(frame_id)
                if buffer is None or buffer.total_chunks != total_chunks:
                    buffer = _FrameBuffer(
                        total_chunks=total_chunks,
                        chunks=[None] * total_chunks,
                        received=0,
                        last_updated=time.monotonic(),
                    )
                    buffers[frame_id] = buffer

                if chunk_idx >= buffer.total_chunks:
                    continue
                if buffer.chunks[chunk_idx] is None:
                    buffer.chunks[chunk_idx] = chunk_payload
                    buffer.received += 1
                buffer.last_updated = time.monotonic()

                if buffer.received == buffer.total_chunks:
                    image_bytes = b"".join(chunk for chunk in buffer.chunks if chunk is not None)
                    self._emit_image(image_bytes)
                    buffers.pop(frame_id, None)
        finally:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
            self._sock = None
            self._running = False

    def stop(self) -> None:
        '''외부 호출로 수신 루프를 종료하고 소켓을 닫는다.'''
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self.wait()

    def _cleanup_buffers(self, buffers: dict[int, _FrameBuffer], now: float) -> None:
        '''지연된 프레임 버퍼를 제거해 메모리 누수를 방지한다.'''
        stale_ids = [
            frame_id
            for frame_id, buffer in buffers.items()
            if now - buffer.last_updated > FRAME_TIMEOUT_SEC
        ]
        for frame_id in stale_ids:
            buffers.pop(frame_id, None)

    def _emit_image(self, data: bytes) -> None:
        '''완성된 프레임 데이터를 QImage로 변환해 신호로 알린다.'''
        if not data:
            return
        image = QtGui.QImage.fromData(data)
        if image.isNull():
            return
        self.frame_received.emit(self._camera_type, image)
