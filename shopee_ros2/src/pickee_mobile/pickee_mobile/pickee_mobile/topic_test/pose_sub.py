import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math

class AmclPoseSubscriber(Node):
    def __init__(self):
        super().__init__('amcl_pose_subscriber')

        # /amcl_pose 토픽 구독 (QoS 기본값)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.listener_callback,
            10  # queue size
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info('✅ Subscribed to /amcl_pose topic')

    def listener_callback(self, msg):
        # 위치 정보
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 쿼터니언 -> yaw(라디안) 변환
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y**2 + q.z**2))

        # 로그 출력
        self.get_logger().info(
            f'📍 Position: (x={x:.3f}, y={y:.3f}), Yaw={math.degrees(yaw):.2f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = AmclPoseSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()