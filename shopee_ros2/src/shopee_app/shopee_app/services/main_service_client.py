from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from dataclasses import field


class MainServiceClientError(Exception):
    """Main Service 통신 오류."""


def _int_from_env(key: str, default: int) -> int:
    """환경 변수에서 정수를 읽고 실패 시 기본값을 반환한다."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_from_env(key: str, default: float) -> float:
    """환경 변수에서 실수를 읽고 실패 시 기본값을 반환한다."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class MainServiceConfig:
    host: str = field(default_factory=lambda: os.getenv("SHOPEE_MAIN_HOST", "192.168.0.25"))
    port: int = field(default_factory=lambda: _int_from_env("SHOPEE_MAIN_PORT", 5000))
    timeout: float = field(default_factory=lambda: _float_from_env("SHOPEE_MAIN_TIMEOUT", 3.0))


class MainServiceClient:
    def __init__(self, config: MainServiceConfig | None = None):
        self.config = config if config is not None else MainServiceConfig()

    def login(self, user_id: str, password: str) -> dict:
        payload = {
            "type": "user_login",
            "data": {
                "user_id": user_id,
                "password": password,
            },
        }
        return self.send(payload)

    def create_order(
        self,
        user_id: str,
        cart_items: list,
        payment_method: str,
        total_amount: int,
    ) -> dict:
        order_items: list[dict[str, int]] = []
        for item in cart_items:
            product_id = None
            quantity = None
            if isinstance(item, dict):
                product_id = item.get("product_id")
                quantity = item.get("quantity")
            else:
                product_id = getattr(item, "product_id", None)
                quantity = getattr(item, "quantity", None)
            if product_id is None or quantity is None:
                raise MainServiceClientError("장바구니 항목 정보가 누락되었습니다.")
            order_items.append(
                {
                    "product_id": int(product_id),
                    "quantity": int(quantity),
                }
            )

        payload = {
            "type": "order_create",
            "data": {
                "user_id": user_id,
                "cart_items": order_items,
                "payment_method": payment_method,
                "total_amount": int(total_amount),
            },
        }
        return self.send(payload)

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
                response = self.recv_all(conn)
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
