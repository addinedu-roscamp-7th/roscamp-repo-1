#!/usr/bin/env python3
"""
SC_05_2_4 - 재고 삭제 시나리오 실행 스크립트
"""
import argparse
import asyncio

from shopee_main_service.scenario_suite import run_sc_05_2_4_inventory_delete


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_05_2_4 재고 삭제 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    asyncio.run(
        run_sc_05_2_4_inventory_delete(
            host=args.host,
            port=args.port,
        )
    )


if __name__ == '__main__':
    main()

