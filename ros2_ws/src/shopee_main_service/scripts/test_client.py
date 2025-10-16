#!/usr/bin/env python3
"""
Main Service 통합 테스트 클라이언트

Mock 환경에서 로그인 → 상품 검색 → 주문 생성 → 상품 선택 → 쇼핑 종료까지
전체 흐름을 검증하기 위한 CLI 도구입니다.
"""
import argparse
import asyncio
from typing import Optional

from shopee_main_service.client_utils import MainServiceClient


DEFAULT_SPEECH_SELECTION = '1번 상품 담아줘'


async def run_full_workflow(host: str, port: int, interactive: bool, speech_selection: Optional[str]) -> None:
    """전체 워크플로우를 순차적으로 실행합니다."""
    client = MainServiceClient(host=host, port=port)
    await client.connect()

    try:
        print('\n' + '=' * 60)
        print('Starting Full Workflow Test')
        if interactive:
            print('(Interactive Mode - Press Enter to proceed each step)')
        print('=' * 60)

        async def wait_step(step_no: int, description: str) -> None:
            """단계별 진행을 안내하고 필요 시 사용자 입력을 대기합니다."""
            title = f'\n[{step_no}] {description}...'
            if interactive:
                input(f'{title}\n→ Press Enter to continue...')
            else:
                print(title)

        await wait_step(1, 'Testing Login')
        login_response = await client.send_request(
            'user_login',
            {'user_id': 'admin', 'password': 'admin123'},
        )
        if not login_response.get('result'):
            print('✗ Login failed, stopping test')
            return
        await asyncio.sleep(0.5)

        await wait_step(2, 'Testing Product Search')
        await client.send_request('product_search', {'query': '비건 사과'})
        await asyncio.sleep(0.5)

        await wait_step(3, 'Testing Order Creation')
        order_response = await client.send_request(
            'order_create',
            {
                'user_id': 'admin',
                'cart_items': [
                    {'product_id': 1, 'quantity': 2},
                    {'product_id': 2, 'quantity': 1},
                ],
            },
        )
        if not order_response.get('result'):
            print('✗ Order creation failed, stopping test')
            return

        order_id = order_response['data']['order_id']
        robot_id = order_response['data']['robot_id']
        await asyncio.sleep(1.0)

        await wait_step(4, 'Testing Video Stream Start')
        await client.send_request(
            'video_stream_start',
            {'robot_id': robot_id, 'user_id': 'admin', 'user_type': 'customer'},
        )
        await asyncio.sleep(0.5)

        await wait_step(5, 'Testing Product Selection')
        await asyncio.sleep(1.0)
        if speech_selection:
            await client.send_request(
                'product_selection_by_text',
                {
                    'order_id': order_id,
                    'robot_id': robot_id,
                    'speech': speech_selection,
                },
            )
        else:
            await client.send_request(
                'product_selection',
                {
                    'order_id': order_id,
                    'robot_id': robot_id,
                    'bbox_number': 1,
                    'product_id': 1,
                },
            )
        await asyncio.sleep(1.0)

        await wait_step(6, 'Testing Shopping End')
        await client.send_request(
            'shopping_end',
            {'user_id': 'admin', 'order_id': order_id, 'robot_id': robot_id},
        )
        await asyncio.sleep(1.0)

        await wait_step(7, 'Testing Video Stream Stop')
        await client.send_request(
            'video_stream_stop',
            {'robot_id': robot_id, 'user_id': 'admin', 'user_type': 'customer'},
        )
        print('\n✓ Full workflow completed')
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Shopee Main Service 테스트 클라이언트')
    parser.add_argument('--host', default='localhost', help='TCP 서버 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='TCP 서버 포트 (기본값: 5000)')
    parser.add_argument('--interactive', action='store_true', help='단계별 사용자 입력 대기 모드')
    parser.add_argument(
        '--speech-selection',
        default=DEFAULT_SPEECH_SELECTION,
        help='텍스트 기반 상품 선택 시 사용할 음성 문장 (예: "사과 가져다줘")',
    )
    parser.add_argument(
        '--no-speech-selection',
        action='store_true',
        help='텍스트 기반 상품 선택 단계를 비활성화하고 기존 bbox 기반 플로우만 수행',
    )
    return parser.parse_args()


def main() -> None:
    """엔트리포인트."""
    args = parse_args()
    speech_value = None if args.no_speech_selection else args.speech_selection
    asyncio.run(run_full_workflow(args.host, args.port, args.interactive, speech_value))


if __name__ == '__main__':
    main()
