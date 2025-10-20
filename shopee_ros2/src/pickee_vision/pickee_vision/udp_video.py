import asyncio
import json
import time
import os
import math
import logging
from typing import Optional
import cv2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 명세서 기반 설정
HOST = '192.168.0.25'  # 수신자(Main Service)의 IP 주소. 로컬 테스트를 위해 127.0.0.1로 설정
PORT = 6000
HEADER_SIZE = 200
CHUNK_DATA_SIZE = 1400
JPEG_QUALITY = 90  # JPEG 압축 품질 (0-100)

class VisionStreamingClient:
    """
    UDP를 통해 비동기적으로 영상 프레임을 스트리밍하는 클라이언트입니다.
    """
    def __init__(self, host: str, port: int, robot_id: int = 1, camera_index: int = 0):
        self._host = host
        self._port = port
        self._robot_id = robot_id
        self._camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._transport: Optional[asyncio.DatagramTransport] = None

    async def start(self):
        """UDP 클라이언트(Endpoint)를 시작하고 카메라를 준비합니다."""
        loop = asyncio.get_running_loop()
        
        self.cap = cv2.VideoCapture(self._camera_index)
        if not self.cap.isOpened():
            logging.error(f"카메라 인덱스 {self._camera_index}를 열 수 없습니다.")
            raise IOError(f"Cannot open camera {self._camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logging.info(f"카메라 {self._camera_index} 열기 성공 (640x480)")

        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: asyncio.DatagramProtocol(),
            remote_addr=(self._host, self._port)
        )
        logging.info(f"UDP 클라이언트 시작. 서버 주소: {self._host}:{self._port}")

    async def stream_frames(self):
        """웹캠 프레임을 캡처하여 무한 루프로 스트리밍합니다."""
        if not self._transport or not self.cap:
            logging.error("클라이언트가 시작되지 않았습니다. start()를 먼저 호출하세요.")
            return

        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("웹캠에서 프레임을 읽는 데 실패했습니다. 1초 후 재시도합니다.")
                await asyncio.sleep(1)
                continue

            # 프레임을 JPEG로 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                logging.warning("프레임을 JPEG로 인코딩하는 데 실패했습니다.")
                continue
            
            image_data = encimg.tobytes()
            image_size = len(image_data)
            total_chunks = math.ceil(image_size / CHUNK_DATA_SIZE)

            logging.info(f"--- Frame ID: {frame_id} (size: {image_size} bytes, chunks: {total_chunks}) 전송 시작 ---")
            
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
                    "width": 640,
                    "height": 480,
                    "format": "jpeg"
                }
                
                header_bytes = json.dumps(header).encode('utf-8')
                padded_header = header_bytes.ljust(HEADER_SIZE)
                packet = padded_header + chunk_data

                # 비동기 환경에서는 아주 짧은 sleep이라도 이벤트 루프에 제어권을 넘겨주는 것이 좋습니다.
                await asyncio.sleep(0.001)
                
                self._transport.sendto(packet)

            frame_id += 1
            # 약 30 FPS로 전송하기 위한 대기 시간
            await asyncio.sleep(1/60)

    def stop(self):
        """카메라와 UDP 클라이언트를 종료합니다."""
        if self.cap:
            self.cap.release()
            logging.info("카메라 자원 해제.")
        if self._transport:
            self._transport.close()
            logging.info("UDP 클라이언트 종료.")

    async def start_and_stream(self):
        """start()와 stream_frames()를 순차적으로 실행하는 헬퍼 메서드"""
        await self.start()
        await self.stream_frames()

async def main():
    """메인 실행 함수"""
    # 사용할 카메라 인덱스(보통 0)를 지정합니다.
    client = VisionStreamingClient(host=HOST, port=PORT, robot_id=1, camera_index=0)
    try:
        await client.start_and_stream()
    except (IOError, KeyboardInterrupt) as e:
        logging.info(f"실행 중단: {e}")
    finally:
        client.stop()

if __name__ == '__main__':
    asyncio.run(main())
