import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobileArrival, Pose2D
import random
import time


class MockArrivalPublisher(Node):
    """
    PickeeMobileArrival 토픽에 임의의 도착 메시지를 publish 하는 Mock 노드
    """

    def __init__(self):
        super().__init__('mock_arrival_publisher')
        self.get_logger().info("🚀 Mock Arrival Publisher Started")

        self.publisher = self.create_publisher(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            10
        )

        # 1초마다 메시지 발행
        self.timer = self.create_timer(4.0, self.publish_mock_arrival)

        # 테스트용 로봇/위치 ID
        self.robot_id = 1
        self.order_id = 101
        self.location_id = 5

    def publish_mock_arrival(self):
        arrival_msg = PickeeMobileArrival()

        # 기본 정보 설정
        arrival_msg.robot_id = self.robot_id
        arrival_msg.order_id = self.order_id
        arrival_msg.location_id = self.location_id

        # 최종 위치 (예: 임의 오차 포함)
        final_pose = Pose2D()
        final_pose.x = round(1.0 + random.uniform(-0.05, 0.05), 3)
        final_pose.y = round(2.0 + random.uniform(-0.05, 0.05), 3)
        final_pose.theta = round(0.0 + random.uniform(-0.05, 0.05), 3)
        arrival_msg.final_pose = final_pose

        # 위치 오차 (임의로 생성)
        pos_err = Pose2D()
        pos_err.x = round(random.uniform(-0.02, 0.02), 3)
        pos_err.y = round(random.uniform(-0.02, 0.02), 3)
        pos_err.theta = round(random.uniform(-0.05, 0.05), 3)
        arrival_msg.position_error = pos_err

        # 이동 시간 (샘플)
        arrival_msg.travel_time = round(random.uniform(3.0, 8.0), 2)

        arrival_msg.message = "Mock arrival success"

        self.publisher.publish(arrival_msg)

        self.get_logger().info(
            f"📤 Mock Arrival Published | "
            f"(x={final_pose.x}, y={final_pose.y}, θ={final_pose.theta}, travel={arrival_msg.travel_time}s)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MockArrivalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
