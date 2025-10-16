#!/usr/bin/env python3
"""
SC_05_2_2 - 재고 수정 시나리오 실행 스크립트
"""
import argparse
import asyncio

from shopee_main_service.scenario_suite import run_sc_05_2_2_inventory_update


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_05_2_2 재고 수정 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument('--product-id', type=int, help='수정 대상 상품 ID (없으면 첫 상품 자동 선택)')
    parser.add_argument('--quantity-delta', type=int, default=1, help='증감할 수량 (기본값: 1)')
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    asyncio.run(
        run_sc_05_2_2_inventory_update(
            host=args.host,
            port=args.port,
            product_id=args.product_id,
            quantity_delta=args.quantity_delta,
        )
    )


if __name__ == '__main__':
    main()

