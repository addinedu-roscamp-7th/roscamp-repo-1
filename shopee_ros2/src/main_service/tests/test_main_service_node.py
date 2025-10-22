'''
MainServiceApp API 핸들러에 대한 단위 테스트입니다.
'''

import pytest
from unittest.mock import MagicMock, AsyncMock

from main_service.main_service_node import MainServiceApp
from shopee_interfaces.srv import PickeeMainVideoStreamStart, PickeeMainVideoStreamStop

# 이 파일의 테스트는 asyncio 기반으로 실행한다
pytestmark = pytest.mark.asyncio


@pytest.fixture
def app() -> MainServiceApp:
    '''모든 의존성을 주입한 MainServiceApp 인스턴스를 생성합니다.'''
    # 모든 의존성에 대한 목 객체를 생성한다
    mock_robot = MagicMock()
    mock_robot.dispatch_video_stream_start = AsyncMock()
    mock_robot.dispatch_video_stream_stop = AsyncMock()

    mock_streamer = MagicMock()
    mock_db = MagicMock()
    mock_llm = AsyncMock()

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    mock_inventory = AsyncMock()
    mock_robot_history = AsyncMock()

    # 목 객체를 생성자에 주입한다
    app_instance = MainServiceApp(
        db=mock_db,
        llm=mock_llm,
        robot=mock_robot,
        event_bus=mock_bus,
        streaming_service=mock_streamer,
        inventory_service=mock_inventory,
        robot_history_service=mock_robot_history,
    )
    app_instance._install_handlers()
    return app_instance


class TestVideoStreamHandlers:
    '''영상 스트림 API 핸들러 테스트 모음입니다.'''

    async def test_handle_video_stream_start(self, app: MainServiceApp):
        '''스트림 시작 핸들러가 올바르게 호출되는지 검증합니다.'''
        # 준비 단계
        handler = app._handlers['video_stream_start']
        
        # 목 객체 반환값 구성
        app._robot.dispatch_video_stream_start.return_value = MagicMock(success=True, message='Stream started')

        request_data = {'robot_id': 1, 'user_id': 'admin', 'user_type': 'admin'}
        peer_address = ('192.168.1.100', 12345)
        APP_UDP_PORT = 6000

        # 실행 단계
        response = await handler(request_data, peer_address)

        # 검증 단계
        app._streaming_service.start_relay.assert_called_once_with(
            robot_id=request_data['robot_id'], 
            user_id=request_data['user_id'], 
            app_ip=peer_address[0], 
            app_port=APP_UDP_PORT
        )
        app._robot.dispatch_video_stream_start.assert_awaited_once()
        assert response['result'] is True


class TestInventoryHandlers:
    '''재고 관련 핸들러 테스트 모음입니다.'''

    async def test_inventory_search_success(self, app: MainServiceApp):
        handler = app._handlers['inventory_search']
        app._inventory_service.search_products.return_value = ([{'product_id': 1}], 1)

        filters = {'name': '사과'}
        response = await handler(filters)

        app._inventory_service.search_products.assert_awaited_once_with(filters)
        assert response['type'] == 'inventory_search_response'
        assert response['result'] is True
        assert response['data']['total_count'] == 1

    async def test_inventory_create_failure(self, app: MainServiceApp):
        handler = app._handlers['inventory_create']
        app._inventory_service.create_product.side_effect = ValueError('Product already exists.')

        response = await handler({'product_id': 1})

        app._inventory_service.create_product.assert_awaited_once()
        assert response['result'] is False
        assert response['error_code'] == 'PROD_003'

    async def test_inventory_update_not_found(self, app: MainServiceApp):
        handler = app._handlers['inventory_update']
        app._inventory_service.update_product.return_value = False

        response = await handler({'product_id': 999})

        app._inventory_service.update_product.assert_awaited_once()
        assert response['result'] is False
        assert response['error_code'] == 'PROD_001'

    async def test_inventory_delete_success(self, app: MainServiceApp):
        handler = app._handlers['inventory_delete']
        app._inventory_service.delete_product.return_value = True

        response = await handler({'product_id': 10})

        app._inventory_service.delete_product.assert_awaited_once_with(10)
        assert response['result'] is True
        assert response['message'] == '재고 정보를 삭제하였습니다.'


class TestRobotHistoryHandlers:
    '''로봇 히스토리 조회 핸들러 테스트 모음입니다.'''

    async def test_robot_history_search_success(self, app: MainServiceApp):
        handler = app._handlers['robot_history_search']
        history_payload = [{'robot_history_id': 1}]
        app._robot_history_service.search_histories.return_value = (history_payload, 1)

        response = await handler({'robot_id': 1})

        app._robot_history_service.search_histories.assert_awaited_once_with({'robot_id': 1})
        assert response['result'] is True
        assert response['data']['total_count'] == 1

    async def test_handle_video_stream_stop(self, app: MainServiceApp):
        '''스트림 중지 핸들러가 올바르게 호출되는지 검증합니다.'''
        # 준비 단계
        handler = app._handlers['video_stream_stop']
        
        # 목 객체 반환값 구성
        app._robot.dispatch_video_stream_stop.return_value = MagicMock(success=True, message='Stream stopped')
        # 마지막 시청자 조건을 시뮬레이션
        app._streaming_service._sessions.get.return_value = []

        request_data = {'robot_id': 1, 'user_id': 'admin', 'user_type': 'admin'}

        # 실행 단계
        response = await handler(request_data, None)

        # 검증 단계
        app._streaming_service.stop_relay.assert_called_once()
        app._robot.dispatch_video_stream_stop.assert_awaited_once()
        assert response['result'] is True
