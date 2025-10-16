#!/usr/bin/env python3
"""
SC_01_1 - 사용자 로그인 시나리오 실행 스크립트
"""
import argparse
import asyncio

from shopee_main_service.scenario_suite import run_sc_01_1_login


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='SC_01_1 로그인 시나리오 실행기')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트 (기본값: 5000)')
    parser.add_argument('--user-id', default='admin', help='로그인 사용자 ID (기본값: admin)')
    parser.add_argument('--password', default='admin123', help='로그인 비밀번호 (기본값: admin123)')
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    asyncio.run(
        run_sc_01_1_login(
            host=args.host,
            port=args.port,
            user_id=args.user_id,
            password=args.password,
        )
    )


if __name__ == '__main__':
    main()
