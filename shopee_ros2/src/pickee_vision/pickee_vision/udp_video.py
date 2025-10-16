import asyncio
import json
import time
import os
import math
import logging
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 명세서 기반 설정
HOST = '0.0.0.0'
PORT = 6000
HEADER_SIZE = 72
CHUNK_DATA_SIZE = 1400

# 테스트용 이미지 파일 경로
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'test_image.jpg')

class VisionStreamingClient:
    """
    UDP를 통해 비동기적으로 영상 프레임을 스트리밍하는 클라이언트입니다.
    streaming_service.py의 구조를 참고하여 송신자 역할을 수행합니다.
    """
    def __init__(self, host: str, port: int, image_path: str):
        self._host = host
        self._port = port
        self._image_path = image_path
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._image_data: bytes = b''
        self._total_chunks: int = 0

    def _load_image_data(self):
        """테스트용 이미지 데이터를 로드하거나, 없을 경우 가짜 데이터를 생성합니다."""
        try:
            with open(self._image_path, 'rb') as f:
                self._image_data = f.read()
            logging.info(f"테스트 이미지 로드 완료: {self._image_path} ({len(self._image_data)} bytes)")
        except FileNotFoundError:
            logging.warning(f"테스트 이미지 파일을 찾을 수 없어, 가짜 데이터를 생성합니다. ({self._image_path})")
            self._image_data = b'\xAA' * (100 * 1024) # 100KB 더미 데이터
        
        image_size = len(self._image_data)
        self._total_chunks = math.ceil(image_size / CHUNK_DATA_SIZE)
        logging.info(f"이미지 크기: {image_size} bytes, 총 청크 수: {self._total_chunks}")

    async def start(self):
        """UDP 클라이언트(Endpoint)를 시작하고 전송을 준비합니다."""
        self._load_image_data()
        loop = asyncio.get_running_loop()
        
        # DatagramEndpoint를 생성합니다. 송신자는 별도의 프로토콜 클래스가 필요 없는 경우가 많습니다.
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: asyncio.DatagramProtocol(), # 기본 프로토콜 사용
            remote_addr=(self._host, self._port)
        )
        logging.info(f"UDP 클라이언트 시작. 서버 주소: {self._host}:{self._port}")

    async def stream_frames(self):
        """이미지 프레임을 무한 루프로 스트리밍합니다."""
        if not self._transport:
            logging.error("클라이언트가 시작되지 않았습니다. start()를 먼저 호출하세요.")
            return

        frame_id = 0
        while True:
            logging.info(f"--- Frame ID: {frame_id} 전송 시작 ---")
            for i in range(self._total_chunks):
                start = i * CHUNK_DATA_SIZE
                end = start + CHUNK_DATA_SIZE
                chunk_data = self._image_data[start:end]
                
                header = {
                    "type": "video_frame",
                    "frame_id": frame_id,
                    "chunk_idx": i,
                    "total_chunks": self._total_chunks,
                    "data_size": len(chunk_data),
                    "timestamp": int(time.time() * 1000),
                    "width": 640,
                    "height": 480,
                    "format": "jpeg"
                }
                
                header_bytes = json.dumps(header).encode('utf-8')
                padded_header = header_bytes.ljust(HEADER_SIZE)
                packet = padded_header + chunk_data
                
                self._transport.sendto(packet)
                # 비동기 환경에서는 아주 짧은 sleep이라도 이벤트 루프에 제어권을 넘겨주는 것이 좋습니다.
                await asyncio.sleep(0.001)

            logging.info(f"  - Frame ID: {frame_id}의 모든 청크 ({self._total_chunks}개) 전송 완료.")
            frame_id += 1
            await asyncio.sleep(1) # 1초에 한 프레임

    def stop(self):
        """UDP 클라이언트를 종료합니다."""
        if self._transport:
            self._transport.close()
            logging.info("UDP 클라이언트 종료.")

    async def start_and_stream(self):
        """start()와 stream_frames()를 순차적으로 실행하는 헬퍼 메서드"""
        await self.start()
        await self.stream_frames()

async def main():
    """메인 실행 함수"""
    client = VisionStreamingClient(HOST, PORT, IMAGE_PATH)
    try:
        await client.start()
        await client.stream_frames()
    except KeyboardInterrupt:
        logging.info("사용자에 의해 중단됨.")
    finally:
        client.stop()

if __name__ == '__main__':
    asyncio.run(main())
