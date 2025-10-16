import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileArrival, PickeeMoveStatus

class MockArrivalAndMoveStatusSubscriber(Node):
    '''
    PickeeMobileArrival 및 PickeeMoveStatus 토픽을 구독하는 Mock 노드.
    '''

    def __init__(self):
        super().__init__('mock_arrival_and_move_status_subscriber')
        self.get_logger().info('Mock Arrival and Move Status Subscriber 노드가 시작되었습니다.')

        self.arrival_subscriber = self.create_subscription(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            self.arrival_callback,
            1
        )
        self.move_status_subscriber = self.create_subscription(
            PickeeMoveStatus,
            '/pickee/mobile/local_path',
            self.move_status_callback,
            10
        )

    def arrival_callback(self, arrival_msg):
        print('reading arrival message')  # 디버그 출력 추가
        self.get_logger().info(f'도착 메시지 수신: robot_id={arrival_msg.robot_id}, order_id={arrival_msg.order_id}, location_id={arrival_msg.location_id}, message="{arrival_msg.message}"')

    def move_status_callback(self, msg: PickeeMoveStatus):
        self.get_logger().info(f'Move Status 수신: target_x={msg.target_x:.2f}, dist={msg.distance_to_target:.2f}, arrived={msg.is_arrived}')

def main(args=None):
    rclpy.init(args=args)
    node = MockArrivalAndMoveStatusSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
