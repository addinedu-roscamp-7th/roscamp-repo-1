#!/usr/bin/env python3
'''
Shopee App 인터랙티브 Mock 클라이언트

Main Service와 TCP로 통신하며 시퀀스 다이어그램의 주요 요청을
수동으로 전송할 수 있는 CLI 도구입니다.
'''

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .client_utils import MainServiceClient


DEFAULT_CART = [
    {'product_id': 1, 'quantity': 2},
    {'product_id': 2, 'quantity': 1},
]


@dataclass
class AppState:
    '''세션 상태를 보관한다.'''

    user_id: str = 'admin'
    password: str = 'admin123'
    robot_id: Optional[int] = None
    order_id: Optional[int] = None
    last_notifications: list[Dict[str, Any]] = field(default_factory=list)


def prompt_text(prompt: str, default: Optional[str] = None) -> str:
    '''문자열 입력을 처리한다.'''
    suffix = f'[{default}] ' if default is not None else ''
    value = input(f'{prompt}{suffix}').strip()
    return value if value else (default or '')


def prompt_int(prompt: str, default: Optional[int] = None) -> int:
    '''정수 입력을 처리한다.'''
    while True:
        suffix = f'[{default}] ' if default is not None else ''
        raw = input(f'{prompt}{suffix}').strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print('⚠ 올바른 정수를 입력하세요.')


async def send_login(client: MainServiceClient, state: AppState) -> None:
    '''로그인 요청을 전송한다.'''
    user_id = prompt_text('사용자 ID 입력', state.user_id)
    password = prompt_text('비밀번호 입력', state.password)
    response = await client.send_request('user_login', {'user_id': user_id, 'password': password})
    if response.get('result'):
        state.user_id = user_id
        state.password = password


async def send_product_search(client: MainServiceClient, state: AppState) -> None:
    '''상품 검색 요청을 전송한다.'''
    query = prompt_text('검색어 입력', '비건 사과')
    payload = {
        'user_id': state.user_id,
        'query': query,
    }
    await client.send_request('product_search', payload)


def parse_cart_items(raw: str) -> list[dict[str, Any]]:
    '''장바구니 JSON 문자열을 파싱한다.'''
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    print('⚠ JSON 배열 형식이 아닙니다. 기본 장바구니를 사용합니다.')
    return DEFAULT_CART


async def send_order_create(client: MainServiceClient, state: AppState) -> None:
    '''주문 생성 요청을 전송한다.'''
    raw_cart = prompt_text('장바구니 JSON 입력', json.dumps(DEFAULT_CART, ensure_ascii=False))
    cart_items = parse_cart_items(raw_cart)
    payload = {
        'user_id': state.user_id,
        'cart_items': cart_items,
    }
    response = await client.send_request('order_create', payload)
    data = response.get('data') or {}
    if response.get('result'):
        state.order_id = data.get('order_id')
        state.robot_id = data.get('robot_id')


async def send_video_stream(client: MainServiceClient, state: AppState, start: bool) -> None:
    '''영상 스트림 제어 메시지를 전송한다.'''
    if state.robot_id is None:
        state.robot_id = prompt_int('로봇 ID 입력', 1)
    robot_id = prompt_int('로봇 ID 입력', state.robot_id)
    state.robot_id = robot_id
    payload = {
        'robot_id': robot_id,
        'user_id': state.user_id,
        'user_type': 'customer',
    }
    msg_type = 'video_stream_start' if start else 'video_stream_stop'
    await client.send_request(msg_type, payload)


async def send_product_selection(client: MainServiceClient, state: AppState, by_text: bool) -> None:
    '''상품 선택 요청을 전송한다.'''
    if state.order_id is None:
        state.order_id = prompt_int('주문 ID 입력', 1)
    if state.robot_id is None:
        state.robot_id = prompt_int('로봇 ID 입력', 1)
    order_id = prompt_int('주문 ID 입력', state.order_id)
    robot_id = prompt_int('로봇 ID 입력', state.robot_id)
    state.order_id = order_id
    state.robot_id = robot_id

    if by_text:
        speech = prompt_text('음성 명령 텍스트 입력', '1번 상품 담아줘')
        payload = {
            'order_id': order_id,
            'robot_id': robot_id,
            'speech': speech,
        }
        await client.send_request('product_selection_by_text', payload)
    else:
        product_id = prompt_int('상품 ID 입력', 1)
        bbox_number = prompt_int('bbox 번호 입력', 1)
        payload = {
            'order_id': order_id,
            'robot_id': robot_id,
            'bbox_number': bbox_number,
            'product_id': product_id,
        }
        await client.send_request('product_selection', payload)


async def send_shopping_end(client: MainServiceClient, state: AppState) -> None:
    '''쇼핑 종료 요청을 전송한다.'''
    if state.order_id is None:
        state.order_id = prompt_int('주문 ID 입력', 1)
    if state.robot_id is None:
        state.robot_id = prompt_int('로봇 ID 입력', 1)
    order_id = prompt_int('주문 ID 입력', state.order_id)
    robot_id = prompt_int('로봇 ID 입력', state.robot_id)
    state.order_id = order_id
    state.robot_id = robot_id
    payload = {
        'user_id': state.user_id,
        'order_id': order_id,
        'robot_id': robot_id,
    }
    await client.send_request('shopping_end', payload)


async def drain_notifications(client: MainServiceClient, state: AppState) -> None:
    '''비동기 알림을 조회한다.'''
    raw = prompt_text('기다릴 알림 타입 (콤마 구분, 비우면 전체)', '')
    expected = {item.strip() for item in raw.split(',') if item.strip()} or None
    timeout = float(prompt_text('대기 시간(초)', '2.0'))
    notes = await client.drain_notifications(timeout=timeout, expected_types=expected)
    state.last_notifications = notes


def show_notifications(state: AppState) -> None:
    '''마지막으로 수집한 알림을 출력한다.'''
    if not state.last_notifications:
        print('최근 수집된 알림이 없습니다.')
        return
    print('\n=== 최근 알림 목록 ===')
    for note in state.last_notifications:
        print(json.dumps(note, ensure_ascii=False, indent=2))


async def send_custom(client: MainServiceClient) -> None:
    '''임의의 메시지를 전송한다.'''
    msg_type = prompt_text('메시지 타입 입력', 'custom_type')
    raw = prompt_text('data JSON 입력', '{}')
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        print('⚠ JSON 파싱에 실패했습니다. 빈 객체로 전송합니다.')
        data = {}
    await client.send_request(msg_type, data)


async def interactive_loop(client: MainServiceClient) -> None:
    '''메인 인터랙티브 루프.'''
    state = AppState()
    commands = {
        'login': send_login,
        'search': send_product_search,
        'order': send_order_create,
        'stream_on': lambda c, s: send_video_stream(c, s, True),
        'stream_off': lambda c, s: send_video_stream(c, s, False),
        'select': lambda c, s: send_product_selection(c, s, False),
        'select_text': lambda c, s: send_product_selection(c, s, True),
        'end': send_shopping_end,
        'drain': drain_notifications,
        'show': lambda _c, s: show_notifications(s),
        'custom': lambda c, _s: send_custom(c),
    }

    print('\n╔══════════════════════════════════════════════════════════╗')
    print('║          Mock App Interactive Client                    ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print('\n명령어 목록:')
    print('  login       - 사용자 로그인')
    print('  search      - 상품 검색')
    print('  order       - 주문 생성')
    print('  stream_on   - 영상 스트림 시작')
    print('  stream_off  - 영상 스트림 중지')
    print('  select      - 상품 선택')
    print('  select_text - 음성 기반 상품 선택')
    print('  end         - 쇼핑 종료')
    print('  drain       - 알림 조회')
    print('  show        - 최근 알림 표시')
    print('  custom      - 커스텀 메시지 전송')
    print('  quit        - 종료')
    print()

    while True:
        cmd = input('\n명령 입력 > ').strip().lower()
        if cmd in ('quit', 'exit'):
            break
        handler = commands.get(cmd)
        if not handler:
            print('⚠ 지원하지 않는 명령입니다.')
            continue
        result = handler(client, state)
        if asyncio.iscoroutine(result):
            await result


def main() -> None:
    '''콘솔 스크립트 진입점'''
    parser = argparse.ArgumentParser(description='Shopee App 인터랙티브 Mock')
    parser.add_argument('--host', default='localhost', help='Main Service 호스트')
    parser.add_argument('--port', type=int, default=5000, help='Main Service 포트')
    args = parser.parse_args()

    async def run():
        client = MainServiceClient(host=args.host, port=args.port)
        await client.connect()
        try:
            await interactive_loop(client)
        finally:
            await client.close()

    asyncio.run(run())


if __name__ == '__main__':
    main()
