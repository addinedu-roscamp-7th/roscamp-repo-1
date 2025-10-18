"""
StreamingService 단위 테스트

UDP 영상 스트리밍 중계 기능을 테스트합니다.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main_service.streaming_service import StreamingService, UdpRelayProtocol

pytestmark = pytest.mark.asyncio


@pytest.fixture
def streaming_service() -> StreamingService:
    """StreamingService 인스턴스 생성"""
    return StreamingService(host='0.0.0.0', port=6000)


class TestStreamingServiceLifecycle:
    """StreamingService 생명주기 테스트"""

    async def test_start_creates_udp_endpoint(
        self, streaming_service: StreamingService
    ):
        """UDP 엔드포인트 시작"""
        # Arrange
        mock_transport = MagicMock()
        mock_protocol = MagicMock()

        with patch.object(asyncio, 'get_running_loop') as mock_get_loop:
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.create_datagram_endpoint = AsyncMock(
                return_value=(mock_transport, mock_protocol)
            )

            # Act
            await streaming_service.start()

        # Assert
        assert streaming_service._transport == mock_transport
        assert streaming_service._protocol == mock_protocol
        mock_loop.create_datagram_endpoint.assert_called_once()
        call_args = mock_loop.create_datagram_endpoint.call_args
        assert call_args.kwargs['local_addr'] == ('0.0.0.0', 6000)

    def test_start_relay_sets_app_address(
        self, streaming_service: StreamingService
    ):
        """앱 목적지 주소 설정"""
        # Arrange
        app_ip = '192.168.1.100'
        app_port = 7000

        # Act
        streaming_service.start_relay(app_ip, app_port)

        # Assert
        assert streaming_service.app_address == (app_ip, app_port)

    def test_stop_relay_clears_app_address(
        self, streaming_service: StreamingService
    ):
        """중계 중지 시 앱 주소 초기화"""
        # Arrange
        streaming_service.app_address = ('192.168.1.100', 7000)

        # Act
        streaming_service.stop_relay()

        # Assert
        assert streaming_service.app_address is None

    def test_stop_relay_when_no_relay_active(
        self, streaming_service: StreamingService
    ):
        """중계가 없을 때 stop_relay 호출"""
        # Arrange
        streaming_service.app_address = None

        # Act (예외 발생하지 않음)
        streaming_service.stop_relay()

        # Assert
        assert streaming_service.app_address is None

    def test_stop_closes_transport(
        self, streaming_service: StreamingService
    ):
        """UDP 서버 종료 시 transport 닫기"""
        # Arrange
        mock_transport = MagicMock()
        streaming_service._transport = mock_transport

        # Act
        streaming_service.stop()

        # Assert
        mock_transport.close.assert_called_once()

    def test_stop_when_no_transport(
        self, streaming_service: StreamingService
    ):
        """transport가 없을 때 stop 호출"""
        # Arrange
        streaming_service._transport = None

        # Act (예외 발생하지 않음)
        streaming_service.stop()

        # Assert (정상 종료)
        pass


class TestUdpRelayProtocol:
    """UdpRelayProtocol 테스트"""

    def test_connection_made(self):
        """연결 생성 시 transport 저장"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        protocol = UdpRelayProtocol(streaming_service)
        mock_transport = MagicMock()
        mock_transport.get_extra_info.return_value = ('0.0.0.0', 6000)

        # Act
        protocol.connection_made(mock_transport)

        # Assert
        assert protocol.transport == mock_transport
        mock_transport.get_extra_info.assert_called_once_with('sockname')

    def test_datagram_received_relays_to_app(self):
        """데이터그램 수신 시 앱으로 전달"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        streaming_service.app_address = ('192.168.1.100', 7000)

        protocol = UdpRelayProtocol(streaming_service)
        mock_transport = MagicMock()
        protocol.transport = mock_transport

        data = b'test video packet'
        robot_addr = ('10.0.0.1', 8000)

        # Act
        protocol.datagram_received(data, robot_addr)

        # Assert
        mock_transport.sendto.assert_called_once_with(data, ('192.168.1.100', 7000))

    def test_datagram_received_no_app_address(self):
        """앱 주소가 없을 때 데이터그램 수신"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        streaming_service.app_address = None

        protocol = UdpRelayProtocol(streaming_service)
        mock_transport = MagicMock()
        protocol.transport = mock_transport

        data = b'test video packet'
        robot_addr = ('10.0.0.1', 8000)

        # Act (패킷이 전달되지 않음)
        protocol.datagram_received(data, robot_addr)

        # Assert
        mock_transport.sendto.assert_not_called()

    def test_datagram_received_no_transport(self):
        """transport가 없을 때 데이터그램 수신"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        streaming_service.app_address = ('192.168.1.100', 7000)

        protocol = UdpRelayProtocol(streaming_service)
        protocol.transport = None

        data = b'test video packet'
        robot_addr = ('10.0.0.1', 8000)

        # Act (예외 발생하지 않음)
        protocol.datagram_received(data, robot_addr)

        # Assert (정상 처리)
        pass

    def test_error_received(self):
        """에러 수신 시 로그 기록"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        protocol = UdpRelayProtocol(streaming_service)
        error = OSError("Network error")

        # Act (예외 발생하지 않음)
        with patch('main_service.streaming_service.logger') as mock_logger:
            protocol.error_received(error)

        # Assert
        mock_logger.error.assert_called_once()

    def test_connection_lost(self):
        """연결 종료 시 로그 기록"""
        # Arrange
        streaming_service = StreamingService(host='0.0.0.0', port=6000)
        protocol = UdpRelayProtocol(streaming_service)

        # Act
        with patch('main_service.streaming_service.logger') as mock_logger:
            protocol.connection_lost(None)

        # Assert
        mock_logger.info.assert_called_once()


class TestStreamingServiceIntegration:
    """통합 시나리오 테스트"""

    async def test_full_streaming_workflow(self):
        """전체 스트리밍 워크플로우"""
        # Arrange
        streaming_service = StreamingService(host='127.0.0.1', port=6000)

        with patch.object(asyncio, 'get_running_loop') as mock_get_loop:
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop
            mock_transport = MagicMock()
            mock_protocol = UdpRelayProtocol(streaming_service)
            mock_loop.create_datagram_endpoint = AsyncMock(
                return_value=(mock_transport, mock_protocol)
            )

            # Act 1: 서버 시작
            await streaming_service.start()
            assert streaming_service._transport == mock_transport

            # Act 2: 앱으로 중계 시작
            streaming_service.start_relay('192.168.1.100', 7000)
            assert streaming_service.app_address == ('192.168.1.100', 7000)

            # Act 3: 패킷 전달 시뮬레이션
            mock_protocol.transport = mock_transport
            test_data = b'video frame 1'
            mock_protocol.datagram_received(test_data, ('10.0.0.1', 8000))
            mock_transport.sendto.assert_called_with(test_data, ('192.168.1.100', 7000))

            # Act 4: 중계 중지
            streaming_service.stop_relay()
            assert streaming_service.app_address is None

            # Act 5: 서버 종료
            streaming_service.stop()
            mock_transport.close.assert_called_once()

    async def test_relay_without_start(self):
        """서버 시작 없이 중계만 설정"""
        # Arrange
        streaming_service = StreamingService(host='127.0.0.1', port=6000)

        # Act
        streaming_service.start_relay('192.168.1.100', 7000)

        # Assert
        assert streaming_service.app_address == ('192.168.1.100', 7000)
        # 실제 전달은 transport가 없으므로 불가능

    async def test_multiple_relay_sessions(self):
        """여러 세션 중계"""
        # Arrange
        streaming_service = StreamingService(host='127.0.0.1', port=6000)

        # Act: 첫 번째 세션
        streaming_service.start_relay('192.168.1.100', 7000)
        assert streaming_service.app_address == ('192.168.1.100', 7000)

        # Act: 두 번째 세션 (덮어쓰기)
        streaming_service.start_relay('192.168.1.200', 8000)
        assert streaming_service.app_address == ('192.168.1.200', 8000)

        # Act: 중지
        streaming_service.stop_relay()
        assert streaming_service.app_address is None
