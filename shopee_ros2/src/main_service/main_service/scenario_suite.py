"""
시퀀스 다이어그램 기반 시나리오 실행 유틸리티

SequenceDiagram 디렉터리에 정의된 주요 SC 시나리오를 자동화하기 위한
비동기 실행 함수를 제공합니다.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .client_utils import MainServiceClient


async def run_sc_01_1_login(host: str = 'localhost', port: int = 5000, user_id: str = 'admin',
                            password: str = 'admin123') -> bool:
    """
    SC_01_1 (로그인 플로우) 시나리오를 실행합니다.

    Returns:
        로그인 성공 여부
    """
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        response = await client.send_request(
            'user_login',
            {'user_id': user_id, 'password': password},
        )
        success = response.get('result', False)
        if success:
            print('✓ SC_01_1: 로그인 시나리오 성공')
        else:
            print('✗ SC_01_1: 로그인 시나리오 실패')
        return success
    finally:
        await client.close()


async def run_sc_01_2_product_search(host: str = 'localhost', port: int = 5000, query: str = '비건 사과',
                                     user_id: str = 'admin', password: Optional[str] = None) -> bool:
    """
    SC_01_2 (상품 검색) 시나리오를 실행합니다.

    필요 시 로그인 과정을 선행합니다.

    Returns:
        검색 응답 성공 여부
    """
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        if password is not None:
            login_response = await client.send_request(
                'user_login',
                {'user_id': user_id, 'password': password},
            )
            if not login_response.get('result', False):
                print('✗ SC_01_2: 로그인 실패로 검색을 진행할 수 없습니다.')
                return False
            await asyncio.sleep(0.2)

        response = await client.send_request('product_search', {'query': query})
        success = response.get('result', False)
        if success:
            print('✓ SC_01_2: 상품 검색 시나리오 성공')
        else:
            print('✗ SC_01_2: 상품 검색 시나리오 실패')
        return success
    finally:
        await client.close()


async def run_sc_01_3_order_create(host: str = 'localhost', port: int = 5000, user_id: str = 'admin',
                                   password: str = 'admin123', cart_items: Optional[list[dict[str, int]]] = None) -> bool:
    """
    SC_01_3 (주문 생성 및 Pickee 작업 시작) 시나리오를 실행합니다.

    Returns:
        주문 생성 성공 여부
    """
    if cart_items is None:
        cart_items = [
            {'product_id': 1, 'quantity': 2},
            {'product_id': 2, 'quantity': 1},
        ]

    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        login_response = await client.send_request(
            'user_login',
            {'user_id': user_id, 'password': password},
        )
        if not login_response.get('result', False):
            print('✗ SC_01_3: 로그인 실패로 주문을 진행할 수 없습니다.')
            return False

        await asyncio.sleep(0.2)
        response = await client.send_request(
            'order_create',
            {'user_id': user_id, 'cart_items': cart_items},
        )
        success = response.get('result', False)
        if success:
            data = response.get('data', {})
            order_id = data.get('order_id')
            robot_id = data.get('robot_id')
            print(f'✓ SC_01_3: 주문 생성 성공 (order_id={order_id}, robot_id={robot_id})')
        else:
            print('✗ SC_01_3: 주문 생성 시나리오 실패')
        return success
    finally:
        await client.close()


def _resolve_cart_items(cart_items: Optional[list[dict[str, int]]]) -> list[dict[str, int]]:
    """장바구니 입력이 없을 때 시퀀스 다이어그램의 기본 구성을 반환합니다."""
    if cart_items:
        return [
            {
                'product_id': int(item.get('product_id', 0)),
                'quantity': int(item.get('quantity', 1)),
            }
            for item in cart_items
        ]
    return [
        {'product_id': 1, 'quantity': 2},
        {'product_id': 2, 'quantity': 1},
    ]


async def _prepare_order_context(
    client: MainServiceClient,
    user_id: str,
    password: str,
    cart_items: list[dict[str, int]],
) -> Optional[tuple[int, int]]:
    """로그인 후 주문을 생성하여 order_id와 robot_id를 획득합니다."""
    login_response = await client.send_request(
        'user_login',
        {'user_id': user_id, 'password': password},
    )
    if not login_response.get('result', False):
        print('✗ 주문 준비: 로그인에 실패했습니다.')
        return None

    order_response = await client.send_request(
        'order_create',
        {'user_id': user_id, 'cart_items': cart_items},
    )
    if not order_response.get('result', False):
        print('✗ 주문 준비: 주문 생성에 실패했습니다.')
        return None

    data = order_response.get('data') or {}
    order_id = data.get('order_id')
    robot_id = data.get('robot_id')
    if order_id is None or robot_id is None:
        print('✗ 주문 준비: order_id 또는 robot_id를 확인할 수 없습니다.')
        return None
    return int(order_id), int(robot_id)


async def _collect_notifications(
    client: MainServiceClient,
    scenario_code: str,
    expected_types: set[str],
    timeout: float,
) -> tuple[bool, list[dict[str, object]]]:
    """필수 비동기 알림 수신 여부를 확인합니다."""
    if not expected_types:
        return True, []

    notifications = await client.drain_notifications(timeout=timeout, expected_types=expected_types)
    received_types = {note.get('type') for note in notifications if isinstance(note, dict)}
    missing = expected_types - received_types

    if missing:
        missing_names = ', '.join(sorted(missing))
        print(f'✗ {scenario_code}: 다음 알림을 수신하지 못했습니다. ({missing_names})')
        return False, notifications

    expected_names = ', '.join(sorted(expected_types))
    print(f'✓ {scenario_code}: {expected_names} 알림 수신 완료')
    return True, notifications


def _generate_temp_product_id() -> int:
    """임시 상품 생성을 위한 고유 ID를 생성합니다."""
    return int(time.time_ns() % 1_000_000) + 1_000_000


def _build_temp_product_payload(base_product: Optional[dict[str, object]]) -> dict[str, object]:
    """기존 상품 정보를 참고하여 임시 상품 입력값을 구성합니다."""
    product_id = _generate_temp_product_id()
    base_name = base_product.get('name', '시나리오 상품') if base_product else '시나리오 상품'
    payload = {
        'product_id': product_id,
        'barcode': f'TEST-{product_id}',
        'name': f'{base_name}-테스트',
        'quantity': int(base_product.get('quantity', 5)) if base_product else 5,
        'price': int(base_product.get('price', 5000)) if base_product else 5000,
        'discount_rate': 0,
        'category': base_product.get('category', 'general') if base_product else 'general',
        'allergy_info_id': int(base_product.get('allergy_info_id', 1)) if base_product else 1,
        'is_vegan_friendly': bool(base_product.get('is_vegan_friendly', True)) if base_product else True,
        'section_id': int(base_product.get('section_id', 1)) if base_product else 1,
    }
    return payload


async def _fetch_reference_product(client: MainServiceClient) -> Optional[dict[str, object]]:
    """임시 상품 생성을 위해 참고할 수 있는 첫 번째 상품을 조회합니다."""
    response = await client.send_request('inventory_search', {})
    if not response.get('result', False):
        return None
    data = response.get('data') or {}
    products = data.get('products') or []
    if not products:
        return None
    return products[0]


async def run_sc_02_4_product_selection(
    host: str = 'localhost',
    port: int = 5000,
    user_id: str = 'admin',
    password: str = 'admin123',
    cart_items: Optional[list[dict[str, int]]] = None,
    bbox_number: int = 1,
    product_id: Optional[int] = None,
) -> bool:
    """
    SC_02_4 (상품 담기) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_02_4.md
    """
    resolved_cart = _resolve_cart_items(cart_items)
    target_product_id = product_id or resolved_cart[0]['product_id']

    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        context = await _prepare_order_context(client, user_id, password, resolved_cart)
        if context is None:
            return False
        order_id, robot_id = context

        required_types = {'robot_moving_notification', 'robot_arrived_notification', 'product_selection_start'}
        events_ok, _ = await _collect_notifications(client, 'SC_02_4', required_types, timeout=3.0)
        if not events_ok:
            return False

        response = await client.send_request(
            'product_selection',
            {
                'order_id': order_id,
                'robot_id': robot_id,
                'bbox_number': bbox_number,
                'product_id': target_product_id,
            },
        )
        if not response.get('result', False):
            print('✗ SC_02_4: 상품 선택 요청이 실패했습니다.')
            return False

        cart_notifications = await client.drain_notifications(timeout=1.5, expected_types={'cart_update_notification'})
        received_cart = any(note.get('type') == 'cart_update_notification' for note in cart_notifications)
        if received_cart:
            print(f'✓ SC_02_4: 상품 선택 및 장바구니 반영 완료 (order_id={order_id}, robot_id={robot_id})')
            return True

        print('✗ SC_02_4: 장바구니 알림을 수신하지 못했습니다.')
        return False
    finally:
        await client.close()


async def run_sc_02_5_shopping_end(
    host: str = 'localhost',
    port: int = 5000,
    user_id: str = 'admin',
    password: str = 'admin123',
    cart_items: Optional[list[dict[str, int]]] = None,
) -> bool:
    """
    SC_02_5 (쇼핑 종료) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_02_5.md
    """
    resolved_cart = _resolve_cart_items(cart_items)
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        context = await _prepare_order_context(client, user_id, password, resolved_cart)
        if context is None:
            return False
        order_id, robot_id = context

        required_types = {'robot_moving_notification', 'robot_arrived_notification', 'product_selection_start'}
        events_ok, _ = await _collect_notifications(client, 'SC_02_5', required_types, timeout=3.0)
        if not events_ok:
            return False

        target_product_id = resolved_cart[0]['product_id']
        selection_response = await client.send_request(
            'product_selection',
            {
                'order_id': order_id,
                'robot_id': robot_id,
                'bbox_number': 1,
                'product_id': target_product_id,
            },
        )
        if not selection_response.get('result', False):
            print('✗ SC_02_5: 상품 선택 단계가 실패했습니다.')
            return False

        cart_notifications = await client.drain_notifications(timeout=1.5, expected_types={'cart_update_notification'})
        if not any(note.get('type') == 'cart_update_notification' for note in cart_notifications):
            print('✗ SC_02_5: 장바구니 반영 알림을 수신하지 못했습니다.')
            return False

        response = await client.send_request(
            'shopping_end',
            {
                'user_id': user_id,
                'order_id': order_id,
            },
        )
        if response.get('result', False):
            summary = response.get('data') or {}
            total_items = summary.get('total_items')
            total_price = summary.get('total_price')
            print(
                f'✓ SC_02_5: 쇼핑 종료 성공 (order_id={order_id}, total_items={total_items}, total_price={total_price})'
            )
            return True

        print('✗ SC_02_5: 쇼핑 종료 요청이 실패했습니다.')
        return False
    finally:
        await client.close()


async def run_sc_05_2_1_inventory_search(
    host: str = 'localhost',
    port: int = 5000,
    filters: Optional[dict[str, object]] = None,
) -> bool:
    """
    SC_05_2_1 (재고 정보 조회) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_05_2_1.md
    """
    payload = filters or {'name': '사과'}
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        response = await client.send_request('inventory_search', payload)
        if response.get('result', False):
            products = (response.get('data') or {}).get('products') or []
            print(f'✓ SC_05_2_1: 재고 조회 성공 (총 {len(products)}건)')
            return True

        print('✗ SC_05_2_1: 재고 조회 요청이 실패했습니다.')
        return False
    finally:
        await client.close()


async def run_sc_05_2_2_inventory_update(
    host: str = 'localhost',
    port: int = 5000,
    product_id: Optional[int] = None,
    quantity_delta: int = 1,
) -> bool:
    """
    SC_05_2_2 (재고 정보 수정) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_05_2_2.md
    """
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        target_id = product_id
        base_product: Optional[dict[str, object]] = None
        if target_id is None:
            search_response = await client.send_request('inventory_search', {})
            if not search_response.get('result', False):
                print('✗ SC_05_2_2: 재고 조회에 실패하여 수정 대상을 찾지 못했습니다.')
                return False
            data = search_response.get('data') or {}
            products = data.get('products') or []
            if not products:
                print('✗ SC_05_2_2: 수정 가능한 상품이 존재하지 않습니다.')
                return False
            base_product = products[0]
            target_id = int(base_product.get('product_id', 0))
        detail_response = await client.send_request('inventory_search', {'product_id': target_id})
        if not detail_response.get('result', False):
            print('✗ SC_05_2_2: 대상 상품 세부 조회에 실패했습니다.')
            return False
        detail_data = detail_response.get('data') or {}
        detail_products = detail_data.get('products') or []
        if not detail_products:
            print('✗ SC_05_2_2: 대상 상품을 찾을 수 없습니다.')
            return False
        product_info = detail_products[0]
        original_quantity = int(product_info.get('quantity', 0))
        new_quantity = max(original_quantity + quantity_delta, 0)

        response = await client.send_request(
            'inventory_update',
            {
                'product_id': target_id,
                'quantity': new_quantity,
            },
        )
        if not response.get('result', False):
            print('✗ SC_05_2_2: 재고 수정 요청이 실패했습니다.')
            return False

        # 테스트 이후 원복하여 실제 데이터 영향을 최소화
        await client.send_request(
            'inventory_update',
            {
                'product_id': target_id,
                'quantity': original_quantity,
            },
        )
        print(f'✓ SC_05_2_2: 재고 수정 시나리오 성공 (product_id={target_id})')
        return True
    finally:
        await client.close()


async def run_sc_05_2_3_inventory_create(
    host: str = 'localhost',
    port: int = 5000,
    cleanup: bool = True,
) -> bool:
    """
    SC_05_2_3 (재고 정보 추가) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_05_2_3.md
    """
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        reference = await _fetch_reference_product(client)
        payload = _build_temp_product_payload(reference)
        response = await client.send_request('inventory_create', payload)
        if not response.get('result', False):
            print('✗ SC_05_2_3: 재고 추가 요청이 실패했습니다.')
            return False

        temp_product_id = payload['product_id']
        print(f'✓ SC_05_2_3: 재고 추가 시나리오 성공 (product_id={temp_product_id})')

        if cleanup:
            cleanup_response = await client.send_request(
                'inventory_delete',
                {'product_id': temp_product_id},
            )
            if not cleanup_response.get('result', False):
                print('⚠️ SC_05_2_3: 임시 상품 삭제에 실패했습니다. 수동 점검이 필요합니다.')
        return True
    finally:
        await client.close()


async def run_sc_05_2_4_inventory_delete(
    host: str = 'localhost',
    port: int = 5000,
) -> bool:
    """
    SC_05_2_4 (재고 정보 삭제) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_05_2_4.md
    """
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        reference = await _fetch_reference_product(client)
        payload = _build_temp_product_payload(reference)
        create_response = await client.send_request('inventory_create', payload)
        if not create_response.get('result', False):
            print('✗ SC_05_2_4: 삭제 테스트용 상품 생성에 실패했습니다.')
            return False

        product_id = payload['product_id']
        delete_response = await client.send_request('inventory_delete', {'product_id': product_id})
        if delete_response.get('result', False):
            print(f'✓ SC_05_2_4: 재고 삭제 시나리오 성공 (product_id={product_id})')
            return True

        print('✗ SC_05_2_4: 재고 삭제 요청이 실패했습니다.')
        return False
    finally:
        await client.close()


async def run_sc_05_3_robot_history_search(
    host: str = 'localhost',
    port: int = 5000,
    filters: Optional[dict[str, object]] = None,
) -> bool:
    """
    SC_05_3 (관리자 작업 이력 조회) 시나리오를 실행합니다.

    참조: docs/SequenceDiagram/SC_05_3.md
    """
    payload = filters or {}
    client = MainServiceClient(host=host, port=port)
    await client.connect()
    try:
        response = await client.send_request('robot_history_search', payload)
        if response.get('result', False):
            histories = (response.get('data') or {}).get('histories') or []
            print(f'✓ SC_05_3: 작업 이력 조회 성공 (총 {len(histories)}건)')
            return True

        print('✗ SC_05_3: 작업 이력 조회 요청이 실패했습니다.')
        return False
    finally:
        await client.close()
