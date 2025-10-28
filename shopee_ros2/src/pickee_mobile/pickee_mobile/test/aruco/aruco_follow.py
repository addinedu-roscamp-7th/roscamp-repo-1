from pickee_mobile.module.module_go_strait import run
from pickee_mobile.module.module_rotate import rotate

import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import ArucoPose
import math

class MockArrivalAndMoveStatusSubscriber(Node):
    '''
    PickeeMobileArrival 및 PickeeMoveStatus 토픽을 구독하는 Mock 노드.
    '''

    def __init__(self):
        super().__init__('ArucoPose_Subscriber')
        self.get_logger().info('ArucoPose Subscriber 노드가 시작되었습니다.')

        self.arrival_subscriber = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_callback,
            10
        )

    def aruco_callback(self, arrival_msg):
        print('reading aruco message')  # 디버그 출력 추가


        # Step 1: rotate(theta_pitch)
        rotate(self, arrival_msg.pitch)
        self.get_logger().info('첫 번째 회전 완료.')

        # Step 2: run( sqrt(x^2 + z^2) / (2 * cos(theta_pitch)) )
        distance = math.sqrt(arrival_msg.x**2 + arrival_msg.z**2) / (2 * math.cos(arrival_msg.pitch))
        run(self, distance)
        self.get_logger().info('직진 주행 완료.')
        self.get_logger().info(f'이동 거리: {distance:.3f} mm')

        # Step 3: rotate(-2 * theta_pitch)
        rotate(self, -2 * arrival_msg.pitch)
        self.get_logger().info('두 번째 회전 완료.')

        self.get_logger().info(
            f"\n📩 [도착 메시지 수신]\n"
            f"  aruco_id      : {arrival_msg.aruco_id}\n"
            f"  x   : {arrival_msg.x:.3f}\n"
            f"  y   : {arrival_msg.y:.3f}\n"
            f"  z   : {arrival_msg.z:.3f}\n"
            f"  roll  : {arrival_msg.roll:.3f}\n"
            f"  pitch : {arrival_msg.pitch:.3f}\n"
            f"  yaw   : {arrival_msg.yaw:.3f}\n"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MockArrivalAndMoveStatusSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
