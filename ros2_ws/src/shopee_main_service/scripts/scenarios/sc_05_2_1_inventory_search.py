#!/usr/bin/env python3
"""
SC_05_2_1 - 재고 조회 시나리오 실행 스크립트
"""
import argparse
import asyncio
import json

from shopee_main_service.scenario_suite import run_sc_05_2_1_inventory_search


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_05_2_1 재고 조회 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument('--filters', default='{"name": "사과"}', help='검색 필터 JSON 문자열')
    return parser.parse_args()


def load_filters(raw: str) -> dict[str, object]:
    """JSON 문자열을 검색 필터로 변환합니다."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise ValueError('filters 옵션은 JSON 객체 형식이어야 합니다.')


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    filters = load_filters(args.filters)
    asyncio.run(
        run_sc_05_2_1_inventory_search(
            host=args.host,
            port=args.port,
            filters=filters,
        )
    )


if __name__ == '__main__':
    main()

