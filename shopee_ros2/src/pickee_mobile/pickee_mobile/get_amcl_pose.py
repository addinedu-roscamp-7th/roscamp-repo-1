import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose2D
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


class LocalizationComponent(Node):
    """
    Pickee Mobile의 위치 추정 노드.
    AMCL로부터 PoseWithCovarianceStamped 메시지를 받아
    x, y, yaw(θ)을 계산해 출력합니다.
    """

    def __init__(self):
        super().__init__('amcl_pose_listener')
        self.get_logger().info('📡 LocalizationComponent 초기화 중...')

        # 현재 pose 데이터 저장용
        self.current_pose = Pose2D()
        self.current_pose.x = 0.0
        self.current_pose.y = 0.0
        self.current_pose.theta = 0.0

        # AMCL Pose 구독
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.get_current_pose,
            10
        )

        self.get_logger().info('✅ Localization Component 초기화 완료.')

    def get_current_pose(self, msg: PoseWithCovarianceStamped):
        """
        AMCL에서 전달받은 PoseWithCovarianceStamped 메시지를 기반으로
        로봇의 현재 위치(x, y, theta)를 업데이트합니다.
        """
        # 메시지에서 위치 좌표 추출
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 쿼터니언을 yaw(θ)로 변환
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = math.atan2(2.0 * qz * qw, 1.0 - 2.0 * (qz ** 2))

        # 내부 상태 업데이트
        self.current_pose.x = x
        self.current_pose.y = y
        self.current_pose.theta = theta

        # 로그 출력
        self.get_logger().info(
            f'📍 AMCL Pose 업데이트 → x={x:.3f}, y={y:.3f}, θ={math.degrees(theta):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationComponent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('🛑 AMCL Pose Listener 종료')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
