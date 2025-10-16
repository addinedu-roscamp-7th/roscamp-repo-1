
import rclpy
from rclpy.node import Node
import asyncio
import threading

from shopee_interfaces.srv import (
    PickeeVisionVideoStreamStart, PickeeVisionVideoStreamStop
)

# 이전에 만든 UDP 비디오 스트리밍 클라이언트를 import 합니다.
# 실제로는 별도의 파일로 분리하고 패키지 의존성을 설정해야 합니다.
from .udp_video import VisionStreamingClient

class CameraServiceNode(Node):
    """
    영상 스트리밍 시작/종료 및 장바구니 존재 여부 확인 서비스를 제공합니다.
    """
    def __init__(self):
        super().__init__('camera_service_node')

        # 서비스 서버 생성
        self.create_service(PickeeVisionVideoStreamStart, '/pickee/vision/video_stream_start', self.video_stream_start_callback)
        self.create_service(PickeeVisionVideoStreamStop, '/pickee/vision/video_stream_stop', self.video_stream_stop_callback)

        # UDP 스트리밍 클라이언트 인스턴스화
        # HOST는 실제 Main Service의 IP로 변경해야 합니다.
        self.udp_client = VisionStreamingClient(host='127.0.0.1', port=6000, image_path='')
        self.streaming_task = None
        self.streaming_loop = None

        self.get_logger().info('Camera Service Node has been started.')

    def video_stream_start_callback(self, request, response):
        self.get_logger().info(f'Video stream start request from user {request.user_id}')
        if self.streaming_task and not self.streaming_task.done():
            response.success = False
            response.message = "Streaming is already in progress."
            self.get_logger().warning('Streaming is already in progress.')
            return response

        # 별도의 스레드에서 asyncio 이벤트 루프를 실행하여 스트리밍 처리
        self.streaming_thread = threading.Thread(target=self.run_streaming_loop)
        self.streaming_thread.start()

        response.success = True
        response.message = "Video streaming started"
        return response

    def run_streaming_loop(self):
        """새로운 이벤트 루프를 생성하고 스트리밍 작업을 실행하는 스레드 함수"""
        self.streaming_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.streaming_loop)
        try:
            self.streaming_task = self.streaming_loop.create_task(self.udp_client.start_and_stream())
            self.streaming_loop.run_until_complete(self.streaming_task)
        except Exception as e:
            self.get_logger().error(f'Error in streaming loop: {e}')
        finally:
            self.streaming_loop.close()

    def video_stream_stop_callback(self, request, response):
        self.get_logger().info(f'Video stream stop request from user {request.user_id}')
        if self.streaming_loop and self.streaming_loop.is_running():
            self.get_logger().info('Stopping streaming task...')
            self.udp_client.stop()
            self.streaming_loop.call_soon_threadsafe(self.streaming_loop.stop)
            self.streaming_thread.join() # 스레드가 끝날 때까지 대기
            self.get_logger().info('Streaming stopped.')
            response.success = True
            response.message = "Video streaming stopped"
        else:
            self.get_logger().warning('Streaming is not running.')
            response.success = False
            response.message = "Streaming is not running."
        return response


    def destroy_node(self):
        # 노드 종료 시 스트리밍도 함께 종료
        self.video_stream_stop_callback(None, None)
        super().destroy_node()

# udp_video.py 파일도 일부 수정이 필요합니다.
# start_and_stream 메서드를 추가하여 start와 stream_frames를 함께 호출하도록 합니다.

async def start_and_stream(self):
    await self.start()
    await self.stream_frames()

# VisionStreamingClient 클래스에 위 메서드를 추가해야 합니다.
VisionStreamingClient.start_and_stream = start_and_stream

def main(args=None):
    rclpy.init(args=args)
    node = CameraServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
