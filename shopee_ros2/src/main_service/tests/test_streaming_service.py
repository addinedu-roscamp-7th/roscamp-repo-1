"""
StreamingService 단위 테스트

UDP 영상 스트리밍 중계 기능을 테스트합니다.
세션 기반 멀티플렉싱을 지원합니다.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main_service.streaming_service import StreamingService, StreamingSession, UdpRelayProtocol

pytestmark = pytest.mark.asyncio


@pytest.fixture
def streaming_service() -> StreamingService:
    """StreamingService 인스턴스 생성"""
    return StreamingService(host='0.0.0.0', port=6000)


class TestStreamingSession:
    """StreamingSession 클래스 테스트"""

    def test_session_creation(self):
        """세션 생성"""
        session = StreamingSession(
            robot_id=1,
            user_id='user1',
            app_ip='192.168.1.100',
            app_port=6000
        )

        assert session.robot_id == 1
        assert session.user_id == 'user1'
        assert session.app_address == ('192.168.1.100', 6000)
        assert session.last_frame_id == -1
        assert len(session.frame_buffer) == 0

    def test_session_expiration(self):
        """세션 만료 확인"""
        session = StreamingSession(1, 'user1', '192.168.1.100', 6000)

        # 방금 생성된 세션은 만료되지 않음
        assert not session.is_expired(timeout=30.0)

        # 타임스탬프를 60초 전으로 설정
        session.last_packet_time = time.time() - 60
        assert session.is_expired(timeout=30.0)

    def test_update_activity(self):
        """활동 시간 갱신"""
        session = StreamingSession(1, 'user1', '192.168.1.100', 6000)
        old_time = session.last_packet_time

        time.sleep(0.01)
        session.update_activity()

        assert session.last_packet_time > old_time

    def test_add_chunk_buffer_limit(self):
        """프레임 버퍼 크기 제한"""
        session = StreamingSession(1, 'user1', '192.168.1.100', 6000)
        session.buffer_max_size = 3  # 테스트용으로 작게 설정

        # 4개 프레임 추가 (버퍼 크기 3 초과)
        for frame_id in range(4):
            session.add_chunk(frame_id, 0, b'data')

        # 가장 오래된 프레임(0)이 제거되고 1, 2, 3만 남음
        assert 0 not in session.frame_buffer
        assert 1 in session.frame_buffer
        assert 2 in session.frame_buffer
        assert 3 in session.frame_buffer
        assert len(session.frame_buffer) == 3


class TestStreamingServiceLifecycle:
    """StreamingService 생명주기 테스트"""

    async def test_start_creates_udp_endpoint(self, streaming_service: StreamingService):
        """UDP 엔드포인트 및 cleanup task 시작"""
        mock_transport = MagicMock()
        mock_protocol = MagicMock()

        with patch.object(asyncio, 'get_running_loop') as mock_get_loop:
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.create_datagram_endpoint = AsyncMock(
                return_value=(mock_transport, mock_protocol)
            )

            await streaming_service.start()

        assert streaming_service._transport == mock_transport
        assert streaming_service._protocol == mock_protocol
        assert streaming_service._cleanup_task is not None

    def test_start_relay_creates_new_session(self, streaming_service: StreamingService):
        """새 세션 생성"""
        streaming_service.start_relay(
            robot_id=1,
            user_id='user1',
            app_ip='192.168.1.100',
            app_port=6000
        )

        assert 1 in streaming_service._sessions
        assert len(streaming_service._sessions[1]) == 1
        assert streaming_service._sessions[1][0].user_id == 'user1'

    def test_start_relay_multiple_users_same_robot(self, streaming_service: StreamingService):
        """한 로봇을 여러 사용자가 시청"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(1, 'user2', '192.168.1.101', 6000)

        assert len(streaming_service._sessions[1]) == 2

    def test_start_relay_duplicate_session(self, streaming_service: StreamingService):
        """중복 세션 방지"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)

        # 중복 세션은 생성되지 않음
        assert len(streaming_service._sessions[1]) == 1

    def test_stop_relay_removes_session(self, streaming_service: StreamingService):
        """세션 종료"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.stop_relay(1, 'user1')

        assert 1 not in streaming_service._sessions

    def test_stop_relay_keeps_other_sessions(self, streaming_service: StreamingService):
        """한 세션 종료 시 다른 세션 유지"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(1, 'user2', '192.168.1.101', 6000)

        streaming_service.stop_relay(1, 'user1')

        assert len(streaming_service._sessions[1]) == 1
        assert streaming_service._sessions[1][0].user_id == 'user2'

    def test_stop_closes_transport_and_cleanup_task(self, streaming_service: StreamingService):
        """서비스 종료 시 transport와 cleanup task 정리"""
        mock_transport = MagicMock()
        mock_cleanup_task = MagicMock()
        streaming_service._transport = mock_transport
        streaming_service._cleanup_task = mock_cleanup_task

        streaming_service.stop()

        mock_transport.close.assert_called_once()
        mock_cleanup_task.cancel.assert_called_once()


class TestStreamingServiceRelay:
    """패킷 중계 테스트"""

    def test_relay_packet_with_robot_id(self, streaming_service: StreamingService):
        """robot_id가 포함된 패킷 중계"""
        # 세션 등록
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)

        # Mock transport 설정
        mock_transport = MagicMock()
        streaming_service._transport = mock_transport

        # 가짜 패킷 생성 (robot_id 포함)
        header = {
            'type': 'video_frame',
            'robot_id': 1,
            'frame_id': 1,
            'chunk_idx': 0,
            'total_chunks': 1,
            'data_size': 10,
            'timestamp': 123456,
            'width': 640,
            'height': 480,
            'format': 'jpeg'
        }
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8').ljust(200, b'\x00')
        packet = header_bytes + b'1234567890'

        # 패킷 중계
        streaming_service.relay_packet(packet)

        # 검증
        mock_transport.sendto.assert_called_once_with(packet, ('192.168.1.100', 6000))

    def test_relay_packet_to_multiple_users(self, streaming_service: StreamingService):
        """한 로봇의 패킷을 여러 사용자에게 중계"""
        # 세션 등록
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(1, 'user2', '192.168.1.101', 6000)

        # Mock transport 설정
        mock_transport = MagicMock()
        streaming_service._transport = mock_transport

        # 가짜 패킷 생성
        header = {
            'type': 'video_frame',
            'robot_id': 1,
            'frame_id': 1,
            'chunk_idx': 0,
            'total_chunks': 1,
            'data_size': 10,
            'timestamp': 123456,
            'width': 640,
            'height': 480,
            'format': 'jpeg'
        }
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8').ljust(200, b'\x00')
        packet = header_bytes + b'1234567890'

        # 패킷 중계
        streaming_service.relay_packet(packet)

        # 두 사용자 모두에게 전송됨
        assert mock_transport.sendto.call_count == 2
        calls = mock_transport.sendto.call_args_list
        assert calls[0][0] == (packet, ('192.168.1.100', 6000))
        assert calls[1][0] == (packet, ('192.168.1.101', 6000))

    def test_relay_packet_without_robot_id(self, streaming_service: StreamingService):
        """robot_id가 없는 패킷은 무시"""
        mock_transport = MagicMock()
        streaming_service._transport = mock_transport

        # robot_id 없는 패킷
        header = {
            'type': 'video_frame',
            'frame_id': 1,
            'chunk_idx': 0,
            'total_chunks': 1,
            'data_size': 10,
            'timestamp': 123456,
            'width': 640,
            'height': 480,
            'format': 'jpeg'
        }
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8').ljust(200, b'\x00')
        packet = header_bytes + b'1234567890'

        # 패킷 중계 시도
        with patch('main_service.streaming_service.logger') as mock_logger:
            streaming_service.relay_packet(packet)

        # 에러 로그만 기록되고 전송되지 않음
        mock_logger.error.assert_called_once()
        mock_transport.sendto.assert_not_called()

    def test_relay_packet_no_active_sessions(self, streaming_service: StreamingService):
        """활성 세션이 없는 로봇의 패킷"""
        mock_transport = MagicMock()
        streaming_service._transport = mock_transport

        # robot_id=2인 패킷 (세션 없음)
        header = {
            'type': 'video_frame',
            'robot_id': 2,
            'frame_id': 1,
            'chunk_idx': 0,
            'total_chunks': 1,
            'data_size': 10,
            'timestamp': 123456,
            'width': 640,
            'height': 480,
            'format': 'jpeg'
        }
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8').ljust(200, b'\x00')
        packet = header_bytes + b'1234567890'

        # 패킷 중계 시도
        streaming_service.relay_packet(packet)

        # 전송되지 않음
        mock_transport.sendto.assert_not_called()


class TestStreamingServiceIntegration:
    """통합 시나리오 테스트"""

    async def test_single_robot_single_user(self, streaming_service: StreamingService):
        """단일 로봇, 단일 사용자"""
        with patch.object(asyncio, 'get_running_loop') as mock_get_loop:
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop
            mock_transport = MagicMock()
            mock_protocol = MagicMock()
            mock_loop.create_datagram_endpoint = AsyncMock(
                return_value=(mock_transport, mock_protocol)
            )

            await streaming_service.start()

        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        assert 1 in streaming_service._sessions
        assert len(streaming_service._sessions[1]) == 1

        streaming_service.stop_relay(1, 'user1')
        assert 1 not in streaming_service._sessions

    async def test_single_robot_multiple_users(self, streaming_service: StreamingService):
        """단일 로봇, 복수 사용자"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(1, 'user2', '192.168.1.101', 6000)

        assert len(streaming_service._sessions[1]) == 2

        # 한 명만 중지
        streaming_service.stop_relay(1, 'user1')
        assert len(streaming_service._sessions[1]) == 1

        # 나머지도 중지
        streaming_service.stop_relay(1, 'user2')
        assert 1 not in streaming_service._sessions

    async def test_multiple_robots_multiple_users(self, streaming_service: StreamingService):
        """복수 로봇, 복수 사용자"""
        # User1 → Robot1, User2 → Robot2, User3 → Robot1
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)
        streaming_service.start_relay(2, 'user2', '192.168.1.101', 6000)
        streaming_service.start_relay(1, 'user3', '192.168.1.102', 6000)

        assert len(streaming_service._sessions[1]) == 2  # Robot1: 2명
        assert len(streaming_service._sessions[2]) == 1  # Robot2: 1명

    async def test_session_cleanup(self, streaming_service: StreamingService):
        """만료된 세션 자동 정리"""
        streaming_service.start_relay(1, 'user1', '192.168.1.100', 6000)

        # 세션의 타임스탬프를 60초 전으로 설정
        session = streaming_service._sessions[1][0]
        session.last_packet_time = time.time() - 60

        # cleanup 로직 수동 실행
        expired_count = 0
        for robot_id in list(streaming_service._sessions.keys()):
            active_sessions = [
                s for s in streaming_service._sessions[robot_id]
                if not s.is_expired(timeout=30.0)
            ]

            removed = len(streaming_service._sessions[robot_id]) - len(active_sessions)
            expired_count += removed

            if active_sessions:
                streaming_service._sessions[robot_id] = active_sessions
            else:
                del streaming_service._sessions[robot_id]

        # 세션이 제거되었는지 확인
        assert expired_count == 1
        assert 1 not in streaming_service._sessions
