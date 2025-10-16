import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose

class MockPoseSubscriber(Node):
    '''
    PickeeMobilePose 토픽을 구독하는 Mock 노드.
    '''

    def __init__(self):
        super().__init__('mock_pose_subscriber')
        self.get_logger().info('Mock Pose Subscriber 노드가 시작되었습니다.')

        self.pose_subscriber = self.create_subscription(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            self.pose_callback,
            10
        )

    def pose_callback(self, msg: PickeeMobilePose):
        self.get_logger().info(f'Pose 수신: x={msg.current_pose.x:.2f}, y={msg.current_pose.y:.2f}, theta={msg.current_pose.theta:.2f}, battery={msg.battery_level:.1f}%, status={msg.status}')

def main(args=None):
    rclpy.init(args=args)
    node = MockPoseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
