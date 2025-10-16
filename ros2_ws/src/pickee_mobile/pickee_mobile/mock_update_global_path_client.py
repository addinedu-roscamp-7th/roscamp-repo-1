import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose2D
from shopee_interfaces.srv import PickeeMobileUpdateGlobalPath
import time

class MockUpdateGlobalPathClient(Node):
    '''
    PickeeMobileUpdateGlobalPath 서비스 클라이언트 Mock 노드.
    '''

    def __init__(self):
        super().__init__('mock_update_global_path_client')
        self.get_logger().info('Mock Update Global Path Client 노드가 시작되었습니다.')

        self.update_global_path_client = self.create_client(
            PickeeMobileUpdateGlobalPath,
            '/pickee/mobile/update_global_path'
        )

        while not self.update_global_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('update_global_path 서비스 대기 중...')

        self.get_logger().info('service available')

        self.timer = self.create_timer(10.0, self.timer_callback) # 20초마다 실행

    def timer_callback(self):
        self.send_update_global_path_request()

    def send_update_global_path_request(self):
        
        
        print('send request')
        request = PickeeMobileUpdateGlobalPath.Request()
        request.robot_id = 1
        request.order_id = 123
        request.location_id = 456

        request.global_path = [
            Pose2D(x=99.0, y=0.0, theta=0.0),
            Pose2D(x=1.0, y=99.5, theta=0.3),
            Pose2D(x=2.0, y=1.0, theta=99.6),
        ]

        self.future = self.update_global_path_client.call_async(request)
        self.future.add_done_callback(self.update_global_path_response_callback)

    def update_global_path_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'전역 경로 업데이트 성공: {response.message}')
            else:
                self.get_logger().warn(f'전역 경로 업데이트 실패: {response.message}')
        except Exception as e:
            self.get_logger().error(f'전역 경로 업데이트 서비스 호출 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MockUpdateGlobalPathClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
