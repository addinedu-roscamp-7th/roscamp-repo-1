#!/usr/bin/env python3
'''
Mock Pickee Vision 노드

실제 Pickee Vision Service를 대신하여 UDP 영상 패킷을 송출합니다.
robot_id 필드를 포함하여 멀티 로봇 시뮬레이션을 지원합니다.
'''
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    print('Error: rclpy is not available. Make sure ROS2 is sourced.')
    sys.exit(1)

logger = logging.getLogger('mock_pickee_vision_node')


class MockPickeeVisionNode(Node):
    '''Mock Pickee Vision Node - UDP 영상 패킷 송출 시뮬레이션'''

    def __init__(self, robot_id: int = 1, main_service_host: str = '127.0.0.1', main_service_port: int = 6000):
        super().__init__('mock_pickee_vision_node')

        self.robot_id = robot_id
        self.main_service_host = main_service_host
        self.main_service_port = main_service_port

        # UDP 영상 송출 관련
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.is_streaming = False
        self.frame_id = 0

        # 영상 설정
        self.width = 640
        self.height = 480
        self.format = 'jpeg'
        self.chunk_size = 1400  # 데이터 청크 크기
        self.fps = 10  # 초당 프레임 수

        self.get_logger().info(
            f'Mock Pickee Vision Node initialized (robot_id={robot_id}, target={main_service_host}:{main_service_port})'
        )

    async def start_streaming(self) -> None:
        '''UDP 영상 스트리밍 시작'''
        if self.is_streaming:
            self.get_logger().warning('Already streaming')
            return

        # UDP 전송용 엔드포인트 생성
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: asyncio.DatagramProtocol(),
            remote_addr=(self.main_service_host, self.main_service_port)
        )

        self.is_streaming = True
        self.get_logger().info(f'Started streaming for robot {self.robot_id}')

        # 비동기 스트리밍 태스크 시작
        asyncio.create_task(self._streaming_loop())

    def stop_streaming(self) -> None:
        '''UDP 영상 스트리밍 중지'''
        if not self.is_streaming:
            return

        self.is_streaming = False
        if self.transport:
            self.transport.close()
            self.transport = None

        self.get_logger().info(f'Stopped streaming for robot {self.robot_id}')

    def _generate_test_image(self) -> np.ndarray:
        '''테스트용 JPEG 이미지 생성 (640x480)'''
        # 배경 생성 (그라디언트)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 색상 그라디언트 배경
        for y in range(self.height):
            color_value = int((y / self.height) * 255)
            image[y, :] = [color_value, 128, 255 - color_value]
        
        # 로봇 ID 텍스트 표시
        text = f'Robot {self.robot_id}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # 프레임 ID 텍스트 표시
        frame_text = f'Frame: {self.frame_id}'
        cv2.putText(image, frame_text, (10, 30), font, 0.7, (255, 255, 0), 2)
        
        # 타임스탬프 표시
        timestamp_text = time.strftime('%H:%M:%S', time.localtime())
        cv2.putText(image, timestamp_text, (10, 60), font, 0.7, (255, 255, 0), 2)
        
        # 랜덤 원 그리기 (동적 효과)
        for _ in range(5):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            radius = random.randint(10, 30)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(image, (x, y), radius, color, -1)
        
        return image

    async def _streaming_loop(self) -> None:
        '''영상 프레임 송출 루프'''
        frame_interval = 1.0 / self.fps

        while self.is_streaming:
            try:
                await self._send_frame()
                await asyncio.sleep(frame_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.get_logger().error(f'Streaming error: {e}')

    async def _send_frame(self) -> None:
        '''한 프레임을 청크로 분할하여 전송'''
        # 실제 JPEG 이미지 생성 (OpenCV 사용)
        image = self._generate_test_image()
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85% 품질
        success, encoded_image = cv2.imencode('.jpg', image, encode_param)
        
        if not success:
            self.get_logger().error('Failed to encode image to JPEG')
            return
        
        jpeg_data = encoded_image.tobytes()

        # 청크 분할
        total_chunks = (len(jpeg_data) + self.chunk_size - 1) // self.chunk_size
        timestamp = int(time.time() * 1000)

        for chunk_idx in range(total_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(jpeg_data))
            chunk_data = jpeg_data[start:end]

            # JSON 헤더 생성 (72바이트 고정)
            header = {
                'type': 'video_frame',
                'robot_id': self.robot_id,  # robot_id 필드 포함 ★
                'frame_id': self.frame_id,
                'chunk_idx': chunk_idx,
                'total_chunks': total_chunks,
                'data_size': len(chunk_data),
                'timestamp': timestamp,
                'width': self.width,
                'height': self.height,
                'format': self.format
            }
            # JSON을 인코딩하고 200바이트로 null 패딩
            header_json = json.dumps(header, separators=(',', ':'))  # 공백 제거
            header_bytes = header_json.encode('utf-8')
            if len(header_bytes) > 200:
                self.get_logger().error(f'Header too large: {len(header_bytes)} bytes')
                continue
            header_bytes = header_bytes.ljust(200, b'\x00')  # null 문자로 패딩

            # 패킷 전송
            packet = header_bytes + chunk_data
            if self.transport:
                self.transport.sendto(packet)

        self.frame_id += 1
        if self.frame_id % 100 == 0:
            self.get_logger().info(f'Sent frame {self.frame_id} (robot_id={self.robot_id})')


async def run_mock_vision_async(robot_id: int, main_service_host: str, main_service_port: int) -> None:
    '''Mock Vision 노드 비동기 실행'''
    rclpy.init()
    node = MockPickeeVisionNode(
        robot_id=robot_id,
        main_service_host=main_service_host,
        main_service_port=main_service_port
    )

    # 스트리밍 시작
    await node.start_streaming()

    # ROS2 spin을 비동기로 실행
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        logger.info('Mock Pickee Vision Node stopped by user')
    finally:
        node.stop_streaming()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    '''콘솔 스크립트 진입점'''
    parser = argparse.ArgumentParser(description='Mock Pickee Vision Node')
    parser.add_argument(
        '--robot-id',
        type=int,
        default=1,
        help='로봇 ID (기본값: 1)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='192.168.0.22',
        help='Main Service 호스트 (기본값: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6000,
        help='Main Service UDP 포트 (기본값: 6000)'
    )
    args, ros_args = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    banner = [
        '╔══════════════════════════════════════════════════════════╗',
        '║       Mock Pickee Vision Node Starting                  ║',
        '╚══════════════════════════════════════════════════════════╝',
        '',
        f'Robot ID: {args.robot_id}',
        f'Target: {args.host}:{args.port}',
        f'Frame Rate: 10 FPS',
        '',
        'Press Ctrl+C to stop',
    ]
    banner_text = '\n'.join(banner)
    print(f'\n{banner_text}\n')

    # asyncio 이벤트 루프에서 실행
    asyncio.run(run_mock_vision_async(args.robot_id, args.host, args.port))


if __name__ == '__main__':
    main()
