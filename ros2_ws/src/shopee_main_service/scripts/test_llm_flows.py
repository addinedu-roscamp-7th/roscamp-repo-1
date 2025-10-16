#!/usr/bin/env python3
"""
LLM 기반 음성 시나리오 테스트 스크립트

텍스트 음성 명령을 통해 Main Service가 LLM 연동 경로를 수행하는지 확인하고,
직접 LLM 엔드포인트도 간단히 점검한다.
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from shopee_main_service.client_utils import MainServiceClient
from shopee_main_service.config import settings
from shopee_main_service.llm_client import LLMClient


async def _run_text_selection_flow(host: str, port: int, speech: str) -> None:
    """product_selection_by_text 메시지를 통해 LLM bbox 추출 경로를 검증한다."""
    client = MainServiceClient(host=host, port=port)
    await client.connect()

    try:
        login_response = await client.send_request(
            'user_login',
            {'user_id': 'admin', 'password': 'admin123'},
        )
        if not login_response.get('result'):
            print('로그인 실패로 테스트를 종료합니다.')
            return

        await asyncio.sleep(0.5)
        await client.send_request('product_search', {'query': '비건 사과'})

        order_response = await client.send_request(
            'order_create',
            {
                'user_id': 'admin',
                'cart_items': [
                    {'product_id': 1, 'quantity': 1},
                    {'product_id': 2, 'quantity': 1},
                ],
            },
        )
        if not order_response.get('result'):
            print('주문 생성 실패로 테스트를 종료합니다.')
            return

        order_id = int(order_response['data']['order_id'])
        robot_id = int(order_response['data']['robot_id'])

        await asyncio.sleep(2.0)

        response = await client.send_request(
            'product_selection_by_text',
            {
                'order_id': order_id,
                'robot_id': robot_id,
                'speech': speech,
            },
        )
        if response.get('result'):
            data = response.get('data') or {}
            resolved_product = data.get('product_id')
            resolved_bbox = data.get('bbox')
            print(
                f'텍스트 기반 담기 성공: product_id={resolved_product}, '
                f'bbox={resolved_bbox}'
            )
        else:
            print('텍스트 기반 담기 실패:', response.get('message'))

        await asyncio.sleep(1.0)
        await client.send_request(
            'shopping_end',
            {'user_id': 'admin', 'order_id': order_id, 'robot_id': robot_id},
        )
    finally:
        await client.close()


async def _run_direct_llm_checks(query_text: str, speech: str, base_url: Optional[str], timeout: Optional[float]) -> None:
    """LLMClient를 이용해 REST 엔드포인트를 직접 호출한다."""
    llm_client = LLMClient(base_url or settings.LLM_BASE_URL, timeout or settings.LLM_TIMEOUT)

    sql_query: Optional[str] = await llm_client.generate_search_query(query_text)
    if sql_query:
        print(f'LLM 검색 쿼리: {sql_query}')
    else:
        print('LLM 검색 쿼리 요청 실패')

    bbox_number: Optional[int] = await llm_client.extract_bbox_number(speech)
    if bbox_number is not None:
        print(f'LLM bbox 추출 결과: {bbox_number}')
    else:
        print('LLM bbox 추출 실패')

    intent_data = await llm_client.detect_intent(speech)
    if intent_data:
        print(f'LLM 의도 분석 결과: {intent_data}')
    else:
        print('LLM 의도 분석 실패')


async def _async_main(args: argparse.Namespace) -> None:
    """비동기 엔트리포인트."""
    if not args.skip_direct:
        await _run_direct_llm_checks(args.query_text, args.speech, args.llm_base_url, args.llm_timeout)
        print('\n---\n')
    await _run_text_selection_flow(args.host, args.port, args.speech)


def _parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description='LLM 연동 시나리오 테스트')
    parser.add_argument('--host', default='localhost', help='Main Service TCP 호스트')
    parser.add_argument('--port', type=int, default=5000, help='Main Service TCP 포트')
    parser.add_argument(
        '--speech',
        default='1번 상품 담아줘',
        help='텍스트 담기 테스트에 사용할 음성 문장',
    )
    parser.add_argument(
        '--query-text',
        default='비건 사과',
        help='LLM 검색 쿼리 생성 테스트에 사용할 문장',
    )
    parser.add_argument(
        '--skip-direct',
        action='store_true',
        help='LLM REST 직접 호출 단계를 건너뜀',
    )
    parser.add_argument(
        '--llm-base-url',
        default=None,
        help='직접 호출에 사용할 LLM Base URL (기본값: 설정값 사용)',
    )
    parser.add_argument(
        '--llm-timeout',
        type=float,
        default=None,
        help='LLM 요청 타임아웃(초)',
    )
    return parser.parse_args()


def main() -> None:
    """동기 엔트리포인트."""
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == '__main__':
    main()
