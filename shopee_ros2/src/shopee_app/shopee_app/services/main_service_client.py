from __future__ import annotations

import json
import socket
from dataclasses import dataclass


class MainServiceClientError(Exception):
    """Main Service 통신 오류."""


@dataclass
class MainServiceConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    timeout: float = 3.0


class MainServiceClient:
    def __init__(self, config: MainServiceConfig | None = None):
        self.config = config if config is not None else MainServiceConfig()

    def request_product_list(self, user_id: str) -> dict:
        payload = {
            "type": "product_list",
            "data": {
                "user_id": user_id,
            },
        }
        return self._send(payload)

    def send(self, payload: dict) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"

        try:
            with socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.timeout,
            ) as conn:
                conn.settimeout(self.config.timeout)
                conn.sendall(encoded)
                conn.shutdown(socket.SHUT_WR)
                response = self._recv_all(conn)
        except OSError as exc:
            raise MainServiceClientError(str(exc)) from exc

        if not response:
            return {}

        try:
            return json.loads(response.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MainServiceClientError("JSON 응답을 해석할 수 없습니다.") from exc

    def recv_all(self, conn: socket.socket) -> bytes:
        chunks: list[bytes] = []
        while True:
            try:
                data = conn.recv(4096)
            except socket.timeout as exc:
                raise MainServiceClientError("서버 응답이 없습니다.") from exc

            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
        return b"".join(chunks).strip()
