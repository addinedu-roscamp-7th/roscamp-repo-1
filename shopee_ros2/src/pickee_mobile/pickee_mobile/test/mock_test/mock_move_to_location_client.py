import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose2D
from shopee_interfaces.srv import PickeeMobileMoveToLocation
import time

class MockMoveToLocationClient(Node):
    '''
    PickeeMobileMoveToLocation 서비스 클라이언트 Mock 노드.
    '''

    def __init__(self):
        super().__init__('mock_move_to_location_client')
        self.get_logger().info('Mock Move To Location Client 노드가 시작되었습니다.')

        self.move_to_location_client = self.create_client(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location'
        )

        while not self.move_to_location_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move_to_location 서비스 대기 중...')

        self.get_logger().info('service available')

        self.timer = self.create_timer(5.0, self.timer_callback) # 10초마다 실행

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0

    def timer_callback(self):
        print('timer')
        self.send_move_to_location_request()

    def send_move_to_location_request(self):
        print('send request')
        request = PickeeMobileMoveToLocation.Request()
        request.robot_id = 1
        request.order_id = 123
        request.location_id = 456
        request.target_pose = Pose2D(x=0.7430041301250458, y=-0.31415562995672226, theta=0.0)
    #     x: 0.4130041301250458
#   y: -0.08875562995672226
        self.future = self.move_to_location_client.call_async(request)
        self.future.add_done_callback(self.move_to_location_response_callback)
        
    def move_to_location_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'이동 명령 성공: {response.message}')
            else:
                self.get_logger().warn(f'이동 명령 실패: {response.message}')
        except Exception as e:
            self.get_logger().error(f'이동 명령 서비스 호출 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MockMoveToLocationClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
