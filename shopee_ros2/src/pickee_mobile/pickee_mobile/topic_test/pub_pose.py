import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose
from shopee_interfaces.msg import Pose2D
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
import math


class GetAmclPose(Node):
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
        self.robot_id = 0
        self.order_id = 0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_state = 'IDLE'
        self.current_battery_level = 100.0

        # AMCL Pose 구독
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.get_current_pose,
            10
        )

        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.modify_cmd_vel_callback,
            10
        )

        self.pose_publisher = self.node.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )

        self.get_logger().info('✅ Localization Component 초기화 완료.')
    
    def modify_cmd_vel_callback(self, msg: Twist):
        """현재 선형 및 각속도 업데이트"""
        self.current_velocity = Twist()
        # 모든 선형 속도에 적용
        
        self.current_velocity.linear.x = msg.linear.x
        self.current_velocity.linear.y = msg.linear.y

        self.current_linear_velocity = math.sqrt(msg.linear.x**2 + msg.linear.y**2)

        # 각속도 읽기
        self.current_angular_velocity = msg.angular.z

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


        pose_msg = PickeeMobilePose()
        # pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.robot_id = self.robot_id
        pose_msg.order_id = self.order_id
        pose_msg.current_pose = self.current_pose
        pose_msg.linear_velocity = self.current_linear_velocity
        pose_msg.angular_velocity = self.current_angular_velocity
        pose_msg.battery_level = self.current_battery_level
        pose_msg.status = self.current_state # 현재 로봇 상태를 메시지에 포함

        self.pose_publisher.publish(pose_msg)

        self.get_logger().info(
            f'🚀 Published PickeeMobilePose: RobotID={pose_msg.robot_id}, OrderID={pose_msg.order_id}, '
            f'Pose=({pose_msg.current_pose.x:.3f}, {pose_msg.current_pose.y:.3f}, {math.degrees(pose_msg.current_pose.theta):.1f}°), '
            f'LinearVel={pose_msg.linear_velocity:.3f} m/s, AngularVel={pose_msg.angular_velocity:.3f} rad/s, '
            f'Battery={pose_msg.battery_level:.1f}%, Status={pose_msg.status}'
        )
        


def main(args=None):
    rclpy.init(args=args)
    node = GetAmclPose()
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
