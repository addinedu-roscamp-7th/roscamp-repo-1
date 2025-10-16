#!/usr/bin/env python3
"""
SC_01_2 - 상품 검색 시나리오 실행 스크립트
"""
import argparse
import asyncio

from shopee_main_service.scenario_suite import run_sc_01_2_product_search


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_01_2 상품 검색 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument('--query', default='비건 사과', help='검색어 (기본값: 비건 사과)')
    parser.add_argument('--user-id', default='admin', help='로그인 사용자 ID (기본값: admin)')
    parser.add_argument('--password', help='로그인 비밀번호 (없으면 로그인 생략)')
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    asyncio.run(
        run_sc_01_2_product_search(
            host=args.host,
            port=args.port,
            query=args.query,
            user_id=args.user_id,
            password=args.password,
        )
    )


if __name__ == '__main__':
    main()
