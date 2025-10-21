import asyncio
import json
import time
import math
import logging
import threading
import queue
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
    별도 스레드에서 동작하며, Queue를 통해 받은 이미지 프레임을
    UDP로 비동기적으로 스트리밍하는 클래스.
    """
    def __init__(self, host: str, port: int, robot_id: int = 1):
        self._host = host
        self._port = port
        self._robot_id = robot_id
        
        self.frame_queue = queue.Queue(maxsize=5) # 프레임 보관함
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.transport: Optional[asyncio.DatagramTransport] = None

    def start(self):
        """스트리밍 스레드를 시작합니다."""
        if self.is_running:
            logging.warning("UdpStreamer is already running.")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_async_loop, name="UdpStreamerThread")
        self.thread.start()
        logging.info("UdpStreamer thread started.")

    def stop(self):
        """스트리밍 스레드를 종료합니다."""
        if not self.is_running:
            return
        
        self.is_running = False
        # 큐에 None을 넣어 블로킹된 get()을 해제하고 루프를 종료시킴
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass # 큐가 꽉 찼으면 어차피 소비자 스레드가 바쁘다는 의미

        if self.thread and self.thread.is_alive():
            self.thread.join()
        logging.info("UdpStreamer thread stopped.")

    def queue_frame(self, frame):
        """메인 스레드에서 스트리밍할 프레임을 큐에 추가합니다."""
        if not self.is_running:
            return
        try:
            # 큐가 꽉 찼으면 오래된 프레임은 버리고 새 프레임을 넣기 위해 non-blocking 사용
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            logging.warning("Frame queue is full, dropping a frame.")

    def _run_async_loop(self):
        """새로운 스레드의 진입점으로, asyncio 이벤트 루프를 설정하고 실행합니다."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_worker())
        finally:
            loop.close()

    async def _async_worker(self):
        """실제 UDP 통신을 담당하는 비동기 워커"""
        loop = asyncio.get_running_loop()
        try:
            self.transport, _ = await loop.create_datagram_endpoint(
                lambda: asyncio.DatagramProtocol(),
                remote_addr=(self._host, self._port)
            )
        except Exception as e:
            logging.error(f"Failed to create UDP endpoint: {e}")
            self.is_running = False
            return

        logging.info(f"UDP endpoint created for {self._host}:{self._port}")

        while self.is_running:
            try:
                # run_in_executor를 사용해 블로킹 I/O인 queue.get()을 비동기적으로 실행
                frame = await loop.run_in_executor(None, self.frame_queue.get)
                
                if frame is None: # 종료 신호
                    break

                await self._send_frame(frame)

            except Exception as e:
                logging.error(f"Error during frame sending: {e}")

        if self.transport:
            self.transport.close()

    async def _send_frame(self, frame):
        """하나의 프레임을 인코딩하고 청크로 나누어 전송합니다."""
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
            
            header_bytes = json.dumps(header).encode('utf-8')
            padded_header = header_bytes.ljust(HEADER_SIZE)
            packet = padded_header + chunk_data
            
            if self.transport:
                self.transport.sendto(packet)