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
            10
        )

    def arrival_callback(self, arrival_msg):
        print('reading arrival message')  # 디버그 출력 추가

        self.get_logger().info(
            f"\n📩 [도착 메시지 수신]\n"
            f"  robot_id      : {arrival_msg.robot_id}\n"
            f"  order_id      : {arrival_msg.order_id}\n"
            f"  location_id   : {arrival_msg.location_id}\n"
            f"  final_pose    : (x={arrival_msg.final_pose.x:.3f}, "
            f"y={arrival_msg.final_pose.y:.3f}, θ={arrival_msg.final_pose.theta:.3f})\n"
            f"  position_error: (x={arrival_msg.position_error.x:.3f}, "
            f"y={arrival_msg.position_error.y:.3f}, θ={arrival_msg.position_error.theta:.3f})\n"
            f"  travel_time   : {arrival_msg.travel_time:.2f} sec\n"
            f"  message       : {arrival_msg.message}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MockArrivalAndMoveStatusSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
