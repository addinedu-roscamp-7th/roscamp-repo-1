'''
UDP 영상 중계 서비스

로봇에서 수신한 UDP 영상 패킷을 앱으로 전달합니다.
'''
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class UdpRelayProtocol(asyncio.DatagramProtocol):
    '''
    UDP 패킷 중계를 담당하는 asyncio DatagramProtocol 구현입니다.
    '''

    def __init__(self, streaming_service: "StreamingService"):
        self.service = streaming_service
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        '''소켓이 준비되었을 때 호출됩니다.'''
        self.transport = transport
        logger.info("UDP Relay server listening on %s", transport.get_extra_info('sockname'))

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        '''
        로봇에서 UDP 데이터그램을 수신했을 때 호출됩니다.
        '''
        if self.service.app_address and self.transport:
            # 수신한 패킷을 앱으로 전달
            self.transport.sendto(data, self.service.app_address)
        # else: logger.debug("UDP packet received but no app address to relay to.")

    def error_received(self, exc: Exception) -> None:
        '''송·수신 과정에서 OSError가 발생했을 때 호출됩니다.'''
        logger.error("UDP server error: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        '''소켓이 닫혔을 때 호출됩니다.'''
        logger.info("UDP Relay server closed.")


class StreamingService:
    '''
    UDP 영상 스트림 중계를 관리합니다.
    '''

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self.app_address: Optional[Tuple[str, int]] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UdpRelayProtocol] = None

    async def start(self) -> None:
        '''
        UDP 서버 엔드포인트를 시작합니다.
        '''
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: UdpRelayProtocol(self),
            local_addr=(self._host, self._port)
        )

    def start_relay(self, app_ip: str, app_port: int) -> None:
        '''
        앱 목적지 주소를 설정하고 중계를 시작합니다.
        '''
        self.app_address = (app_ip, app_port)
        logger.info("Started UDP relay to %s:%d", app_ip, app_port)

    def stop_relay(self) -> None:
        '''
        앱 목적지 정보를 초기화하고 중계를 중단합니다.
        '''
        if self.app_address:
            logger.info("Stopped UDP relay to %s", self.app_address)
            self.app_address = None

    def stop(self) -> None:
        '''
        UDP 서버를 종료합니다.
        '''
        if self._transport:
            self._transport.close()
