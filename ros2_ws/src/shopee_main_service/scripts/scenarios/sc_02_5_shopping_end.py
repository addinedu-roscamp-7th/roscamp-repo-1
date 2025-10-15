#!/usr/bin/env python3
"""
SC_02_5 - 쇼핑 종료 시나리오 실행 스크립트
"""
import argparse
import asyncio
import json

from shopee_main_service.scenario_suite import run_sc_02_5_shopping_end


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_02_5 쇼핑 종료 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument('--user-id', default='admin', help='로그인 사용자 ID (기본값: admin)')
    parser.add_argument('--password', default='admin123', help='로그인 비밀번호 (기본값: admin123)')
    parser.add_argument(
        '--cart-items',
        default='[{"product_id": 1, "quantity": 2}, {"product_id": 2, "quantity": 1}]',
        help='장바구니 항목 JSON 문자열',
    )
    return parser.parse_args()


def load_cart_items(raw: str) -> list[dict[str, int]]:
    """JSON 문자열을 장바구니 목록으로 변환합니다."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    raise ValueError('cart-items 옵션은 JSON 배열 형식이어야 합니다.')


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    cart_items = load_cart_items(args.cart_items)
    asyncio.run(
        run_sc_02_5_shopping_end(
            host=args.host,
            port=args.port,
            user_id=args.user_id,
            password=args.password,
            cart_items=cart_items,
        )
    )


if __name__ == '__main__':
    main()

