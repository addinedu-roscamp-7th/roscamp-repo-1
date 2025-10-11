"""
UDP Video Streaming Relay Service

This module handles the relay of UDP video packets from the robot to the app.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class UdpRelayProtocol(asyncio.DatagramProtocol):
    """
    An asyncio DatagramProtocol implementation for relaying UDP packets.
    """

    def __init__(self, streaming_service: "StreamingService"):
        self.service = streaming_service
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Called when the socket is set up."""
        self.transport = transport
        logger.info("UDP Relay server listening on %s", transport.get_extra_info('sockname'))

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """
        Called when a UDP datagram is received from the robot.
        """
        if self.service.app_address and self.transport:
            # Forward the received packet to the app
            self.transport.sendto(data, self.service.app_address)
        # else: logger.debug("UDP packet received but no app address to relay to.")

    def error_received(self, exc: Exception) -> None:
        """Called when a send or receive operation raises an OSError."""
        logger.error("UDP server error: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Called when the socket is closed."""
        logger.info("UDP Relay server closed.")


class StreamingService:
    """
    Manages the UDP relay for video streaming.
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self.app_address: Optional[Tuple[str, int]] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UdpRelayProtocol] = None

    async def start(self) -> None:
        """
        Starts the UDP server endpoint.
        """
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: UdpRelayProtocol(self),
            local_addr=(self._host, self._port)
        )

    def start_relay(self, app_ip: str, app_port: int) -> None:
        """
        Sets the destination app address and starts relaying.
        """
        self.app_address = (app_ip, app_port)
        logger.info("Started UDP relay to %s:%d", app_ip, app_port)

    def stop_relay(self) -> None:
        """
        Clears the destination app address and stops relaying.
        """
        if self.app_address:
            logger.info("Stopped UDP relay to %s", self.app_address)
            self.app_address = None

    def stop(self) -> None:
        """
        Stops the UDP server.
        """
        if self._transport:
            self._transport.close()
