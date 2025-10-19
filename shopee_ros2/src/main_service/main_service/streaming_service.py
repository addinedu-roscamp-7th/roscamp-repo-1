'''
UDP 영상 중계 서비스

로봇에서 수신한 UDP 영상 패킷을 앱으로 전달합니다.
복수의 로봇-사용자 세션을 동시에 지원합니다.
'''
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StreamingSession:
    '''
    한 사용자가 한 로봇의 영상을 시청하는 세션

    각 세션은 robot_id, user_id, 앱 주소를 저장하며,
    프레임 재조립 버퍼와 타임아웃 관리 기능을 제공합니다.
    '''

    def __init__(self, robot_id: int, user_id: str, app_ip: str, app_port: int):
        self.robot_id = robot_id
        self.user_id = user_id
        self.app_address = (app_ip, app_port)

        # 청크 재조립용
        self.last_frame_id = -1
        self.frame_buffer: Dict[int, Dict[int, bytes]] = {}  # frame_id -> {chunk_idx: data}
        self.buffer_max_size = 10  # 최대 10개 프레임만 보관 (메모리 누수 방지)

        # 타임아웃 관리
        self.last_packet_time = time.time()
        self.created_at = time.time()

    def is_expired(self, timeout: float = 30.0) -> bool:
        '''마지막 패킷 수신 후 timeout 초 경과 시 만료'''
        return time.time() - self.last_packet_time > timeout

    def update_activity(self):
        '''활동 시간 갱신'''
        self.last_packet_time = time.time()

    def add_chunk(self, frame_id: int, chunk_idx: int, data: bytes):
        '''프레임 청크 추가 및 버퍼 크기 관리'''
        # 오래된 프레임 제거 (메모리 누수 방지)
        if len(self.frame_buffer) > self.buffer_max_size:
            oldest_frame = min(self.frame_buffer.keys())
            del self.frame_buffer[oldest_frame]
            logger.debug(f'Removed old frame {oldest_frame} from buffer')

        # 청크 추가
        if frame_id not in self.frame_buffer:
            self.frame_buffer[frame_id] = {}
        self.frame_buffer[frame_id][chunk_idx] = data


class UdpRelayProtocol(asyncio.DatagramProtocol):
    '''
    UDP 패킷 중계를 담당하는 asyncio DatagramProtocol 구현입니다.
    '''

    def __init__(self, streaming_service: 'StreamingService'):
        self.service = streaming_service
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        '''소켓이 준비되었을 때 호출됩니다.'''
        self.transport = transport
        self.service._transport = transport  # StreamingService에 전달
        logger.info('UDP Relay server listening on %s', transport.get_extra_info('sockname'))

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        '''
        로봇에서 UDP 데이터그램을 수신했을 때 호출됩니다.
        '''
        # StreamingService에 위임
        self.service.relay_packet(data)

    def error_received(self, exc: Exception) -> None:
        '''송·수신 과정에서 OSError가 발생했을 때 호출됩니다.'''
        logger.error('UDP server error: %s', exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        '''소켓이 닫혔을 때 호출됩니다.'''
        logger.info('UDP Relay server closed.')


class StreamingService:
    '''
    UDP 영상 스트림 중계를 관리합니다.
    복수의 로봇-사용자 세션을 동시에 지원합니다.
    '''

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

        # robot_id -> List[StreamingSession]
        self._sessions: Dict[int, List[StreamingSession]] = {}

        # 세션 정리용 타이머
        self._cleanup_task: Optional[asyncio.Task] = None

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UdpRelayProtocol] = None

        # 대시보드 콜백 (패킷을 대시보드로 전달)
        self._dashboard_callback = None

    async def start(self) -> None:
        '''
        UDP 서버 엔드포인트를 시작합니다.
        '''
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: UdpRelayProtocol(self),
            local_addr=(self._host, self._port)
        )

        # 세션 정리 태스크 시작
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        logger.info('StreamingService started with session cleanup task')

    def start_relay(self, robot_id: int, user_id: str, app_ip: str, app_port: int) -> None:
        '''특정 로봇-사용자 세션 시작'''
        # 기존 세션 중복 확인
        if robot_id in self._sessions:
            for session in self._sessions[robot_id]:
                if session.user_id == user_id:
                    logger.warning(f'Session already exists: robot={robot_id}, user={user_id}')
                    session.update_activity()
                    return

        # 새 세션 생성
        session = StreamingSession(robot_id, user_id, app_ip, app_port)

        if robot_id not in self._sessions:
            self._sessions[robot_id] = []
        self._sessions[robot_id].append(session)

        logger.info(f'Started streaming session: robot={robot_id}, user={user_id}, app={app_ip}:{app_port}')

    def stop_relay(self, robot_id: int, user_id: str) -> None:
        '''특정 로봇-사용자 세션 종료'''
        if robot_id in self._sessions:
            before_count = len(self._sessions[robot_id])
            self._sessions[robot_id] = [
                s for s in self._sessions[robot_id]
                if s.user_id != user_id
            ]
            after_count = len(self._sessions[robot_id])

            if before_count > after_count:
                logger.info(f'Stopped streaming session: robot={robot_id}, user={user_id}')

            # 세션이 모두 종료되면 robot_id 제거
            if not self._sessions[robot_id]:
                del self._sessions[robot_id]
                logger.info(f'All sessions closed for robot {robot_id}')

    def set_dashboard_callback(self, callback):
        '''대시보드 콜백 등록'''
        self._dashboard_callback = callback

    def relay_packet(self, data: bytes) -> None:
        '''UDP 패킷을 파싱하여 해당 로봇을 시청 중인 모든 앱에 전달'''
        try:
            # JSON 헤더 파싱 (첫 200바이트)
            header_bytes = data[:200]
            # null 문자 제거 후 파싱
            header_str = header_bytes.decode('utf-8').rstrip('\x00').strip()
            header = json.loads(header_str)

            robot_id = header.get('robot_id')
            if robot_id is None:
                logger.error('Packet missing robot_id field')
                return

            # 대시보드로 패킷 전달 (콜백 방식)
            if self._dashboard_callback:
                try:
                    self._dashboard_callback(data)
                except Exception as e:
                    logger.error(f'Failed to send to dashboard: {e}')

            # 이 로봇을 시청 중인 세션들에게 전달
            if robot_id in self._sessions:
                for session in self._sessions[robot_id]:
                    try:
                        self._transport.sendto(data, session.app_address)
                        session.update_activity()
                    except Exception as e:
                        logger.error(f'Failed to send to {session.app_address}: {e}')
            # else:
            #     logger.debug(f'No active sessions for robot {robot_id}')

        except Exception as e:
            logger.error(f'Failed to relay packet: {e}')

    async def _cleanup_expired_sessions(self):
        '''주기적으로 만료된 세션 정리 (30초 타임아웃)'''
        while True:
            try:
                await asyncio.sleep(10)  # 10초마다 확인

                expired_count = 0
                for robot_id in list(self._sessions.keys()):
                    active_sessions = [
                        s for s in self._sessions[robot_id]
                        if not s.is_expired(timeout=30.0)
                    ]

                    removed = len(self._sessions[robot_id]) - len(active_sessions)
                    if removed > 0:
                        logger.info(f'Removed {removed} expired session(s) for robot {robot_id}')
                        expired_count += removed

                    if active_sessions:
                        self._sessions[robot_id] = active_sessions
                    else:
                        del self._sessions[robot_id]

                if expired_count > 0:
                    logger.info(f'Cleanup completed: {expired_count} session(s) removed')

            except asyncio.CancelledError:
                logger.info('Session cleanup task cancelled')
                break
            except Exception as e:
                logger.error(f'Cleanup task error: {e}')

    def stop(self) -> None:
        '''
        UDP 서버를 종료합니다.
        '''
        if self._cleanup_task:
            self._cleanup_task.cancel()

        if self._transport:
            self._transport.close()

        self._sessions.clear()
        logger.info('StreamingService stopped')
