import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileSpeedControl

class MockSpeedControlPublisher(Node):
    '''
    PickeeMobileSpeedControl 토픽을 발행하는 Mock 노드.
    '''

    def __init__(self):
        super().__init__('mock_speed_control_publisher')
        self.get_logger().info('Mock Speed Control Publisher 노드가 시작되었습니다.')

        self.speed_control_publisher = self.create_publisher(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            10
        )
        self.timer = self.create_timer(1.0, self.timer_callback) # 1초마다 실행

    def timer_callback(self):
        speed_control_msg = PickeeMobileSpeedControl()
        speed_control_msg.speed_mode = 'normal'
        speed_control_msg.target_speed = 0.5
        self.speed_control_publisher.publish(speed_control_msg)
        self.get_logger().info(f'Speed Control 발행: mode={speed_control_msg.speed_mode}, speed={speed_control_msg.target_speed}')

def main(args=None):
    rclpy.init(args=args)
    node = MockSpeedControlPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
