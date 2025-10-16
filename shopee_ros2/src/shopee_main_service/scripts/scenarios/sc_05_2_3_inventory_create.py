#!/usr/bin/env python3
"""
SC_05_2_3 - 재고 추가 시나리오 실행 스크립트
"""
import argparse
import asyncio

from shopee_main_service.scenario_suite import run_sc_05_2_3_inventory_create


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_05_2_3 재고 추가 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='시나리오 종료 후 임시 상품을 삭제하지 않습니다.',
    )
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    asyncio.run(
        run_sc_05_2_3_inventory_create(
            host=args.host,
            port=args.port,
            cleanup=not args.no_cleanup,
        )
    )


if __name__ == '__main__':
    main()

