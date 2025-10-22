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

    def search_products(
        self,
        user_id: str,
        query: str,
        *,
        allergy_filter: dict[str, bool] | None = None,
        is_vegan: bool | None = None,
    ) -> dict:
        # 검색 필터를 누적할 딕셔너리가 없으면 조건별 데이터를 모을 수 없어 서버에 온전한 요청을 보낼 수 없다.
        filter_payload: dict[str, object] = {}
        # 알레르기 필터가 제공되었는지 확인하지 않으면 기본값과 사용자 입력을 구분할 수 없어 의미 없는 필터를 보낼 수 있다.
        if allergy_filter is not None:
            # 알레르기 정보를 저장할 새로운 딕셔너리를 만들지 않으면 원본 입력을 그대로 수정하게 되어 호출자 쪽 데이터가 변형될 수 있다.
            normalized_allergy: dict[str, bool] = {}
            # 개별 항목을 순회하지 않으면 값이 불리언으로 변환되지 않아 서버가 예상한 형식을 받지 못한다.
            for key, value in allergy_filter.items():
                # 각 항목을 불리언으로 변환하지 않으면 문자열이나 정수로 전달될 수 있어 명세와 달라진다.
                normalized_allergy[key] = bool(value)
            # 정규화된 알레르기 정보를 필터에 포함하지 않으면 서버가 사용자의 알레르기 제한을 고려하지 못한다.
            filter_payload['allergy_info'] = normalized_allergy
        # 비건 여부가 명시되었는지 확인하지 않으면 서버에 불필요한 필드가 전달되어 기본 동작을 방해할 수 있다.
        if is_vegan is not None:
            # 비건 여부를 불리언으로 강제하지 않으면 True/False 외의 값이 전달되어 검증 오류가 발생할 수 있다.
            filter_payload['is_vegan'] = bool(is_vegan)
        # 알레르기 정보 키가 없으면 서버는 구조를 파싱할 때 KeyError를 일으킬 수 있다.
        if 'allergy_info' not in filter_payload:
            # 빈 알레르기 딕셔너리를 넣어 두지 않으면 필드가 아예 사라져 명세와 달라진다.
            filter_payload['allergy_info'] = {}
        # 비건 여부가 빠지면 서버가 기본값을 추측해야 하므로 명시적으로 False를 채워 넣는다.
        if 'is_vegan' not in filter_payload:
            # False를 기본값으로 전달하지 않으면 비건 필터가 적용되지 않아 예상과 다른 상품이 포함될 수 있다.
            filter_payload['is_vegan'] = False
        # 검색 요청 페이로드를 구성하지 않으면 서버가 어떤 작업을 수행해야 하는지 전혀 알 수 없다.
        payload = {
            # type 필드를 누락하면 서버가 메시지 유형을 판별할 수 없어 요청을 거부한다.
            'type': 'product_search',
            # data 필드를 비워두면 사용자와 검색 조건이 전달되지 않아 의미 있는 검색이 불가능하다.
            'data': {
                # 사용자 ID를 포함하지 않으면 서버가 개인화된 필터를 적용하거나 권한을 확인할 수 없다.
                'user_id': user_id,
                # 검색어를 전달하지 않으면 서버는 무엇을 찾아야 할지 몰라 전체 목록만 반환하거나 오류를 낼 수 있다.
                'query': query,
                # 필터 객체가 없으면 명세에 정의된 구조를 따르지 않아 서버가 요청을 거절할 수 있다.
                'filter': filter_payload,
            },
        }
        # 소켓 전송을 수행하지 않으면 페이로드가 서버로 전달되지 않아 검색 결과를 받을 수 없다.
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
