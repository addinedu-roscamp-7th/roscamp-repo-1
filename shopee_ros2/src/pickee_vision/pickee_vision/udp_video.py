import asyncio
import json
import time
import math
import logging
import threading
from typing import Optional
import cv2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 명세서 기반 설정
HEADER_SIZE = 200
CHUNK_DATA_SIZE = 1400
JPEG_QUALITY = 90

class UdpStreamer:
    """
    별도의 스레드에서 asyncio 이벤트 루프를 실행하여,
    메인 스레드를 차단하지 않고 이미지 프레임을 UDP로 비동기 전송합니다.
    큐를 사용하지 않고 'fire-and-forget' 방식으로 동작합니다.
    """
    def __init__(self, host: str, port: int, robot_id: int = 1):
        self._host = host
        self._port = port
        self._robot_id = robot_id
        
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        # 스트리밍 스레드를 시작합니다.
        if self.is_running:
            logging.warning("UdpStreamer is already running.")
            return
        
        self.is_running = True
        # asyncio 이벤트 루프를 위한 백그라운드 스레드 시작
        self.thread = threading.Thread(target=self._run_async_loop, name="UdpStreamerThread")
        self.thread.daemon = True  # 메인 프로그램이 종료될 때 스레드도 함께 종료되도록 설정
        self.thread.start()
        logging.info("UdpStreamer thread started.")

    def stop(self):
        # 스트리밍 스레드를 종료합니다.
        if not self.is_running or self._loop is None:
            return
        
        self.is_running = False
        
        # 메인 스레드에서 이벤트 루프 종료를 예약
        self._loop.call_soon_threadsafe(self._loop.stop)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0) # 스레드가 완전히 종료될 때까지 대기
        logging.info("UdpStreamer thread stopped.")

    def send_frame(self, frame):
        """
        메인 스레드에서 프레임의 비동기 전송을 예약합니다.
        이 작업은 논블로킹(non-blocking) 입니다.
        """
        if not self.is_running or self._loop is None or not self._loop.is_running():
            return
        
        # 백그라운드 이벤트 루프에서 _send_frame 코루틴을 실행하도록 스케줄링
        asyncio.run_coroutine_threadsafe(self._send_frame(frame), self._loop)

    def _run_async_loop(self):
        # 백그라운드 스레드의 진입점. asyncio 이벤트 루프를 설정하고 실행합니다.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # 루프 시작 전에 UDP 엔드포인트 설정
            setup_task = self._loop.create_task(self._setup_udp())
            self._loop.run_until_complete(setup_task)

            if not self.transport:
                logging.error("UDP transport setup failed. Exiting thread.")
                self.is_running = False
                return

            # stop()이 호출될 때까지 이벤트 루프를 계속 실행
            self._loop.run_forever()

        except Exception as e:
            logging.error(f"Exception in UdpStreamer event loop: {e}")
        finally:
            logging.info("UdpStreamer event loop shutting down.")
            if self.transport:
                self.transport.close()
            
            # 남은 태스크 정리
            tasks = asyncio.all_tasks(loop=self._loop)
            for task in tasks:
                task.cancel()
            group = asyncio.gather(*tasks, return_exceptions=True)
            self._loop.run_until_complete(group)
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            self._loop = None
            logging.info("UdpStreamer event loop closed.")

    async def _setup_udp(self):
        # 비동기적으로 UDP 엔드포인트를 생성합니다.
        if not self._loop:
            return
        try:
            self.transport, _ = await self._loop.create_datagram_endpoint(
                lambda: asyncio.DatagramProtocol(),
                remote_addr=(self._host, self._port)
            )
            logging.info(f"UDP endpoint created for {self._host}:{self._port}")
        except Exception as e:
            logging.error(f"Failed to create UDP endpoint: {e}")
            self.transport = None

    async def _send_frame(self, frame):
        # 하나의 프레임을 인코딩하고 청크로 나누어 전송합니다.
        if not self.transport:
            logging.warning("UDP transport not available, cannot send frame.")
            return

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        result, encimg = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            logging.warning("Failed to encode frame to JPEG.")
            return

        image_data = encimg.tobytes()
        image_size = len(image_data)
        total_chunks = math.ceil(image_size / CHUNK_DATA_SIZE)
        
        frame_id = int(time.time() * 1000) # 간단한 프레임 ID
        
        for i in range(total_chunks):
            start = i * CHUNK_DATA_SIZE
            end = start + CHUNK_DATA_SIZE
            chunk_data = image_data[start:end]
            
            header = {
                "type": "video_frame",
                "robot_id": self._robot_id,
                "frame_id": frame_id,
                "chunk_idx": i,
                "total_chunks": total_chunks,
                "data_size": len(chunk_data),
                "timestamp": int(time.time() * 1000),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "format": "jpeg"
            }
            
            header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
            if len(header_bytes) > HEADER_SIZE:
                logging.error(f'Header too large: {len(header_bytes)} bytes')
                continue
            
            padded_header = header_bytes.ljust(HEADER_SIZE, b'\x00')
            packet = padded_header + chunk_data
            
            self.transport.sendto(packet)
