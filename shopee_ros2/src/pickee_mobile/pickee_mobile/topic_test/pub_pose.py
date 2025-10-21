import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose, Pose2D
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
import math
import time


class GetAmclPose(Node):
    """
    Pickee Mobile의 위치 추정 노드.
    AMCL로부터 PoseWithCovarianceStamped 메시지를 받아
    x, y, yaw(θ)을 계산해 /pickee/mobile/pose 토픽으로 발행합니다.
    """

    def __init__(self):
        super().__init__('amcl_pose_listener')
        self.get_logger().info('📡 LocalizationComponent 초기화 중...')

        # 현재 pose 및 상태 변수 초기화
        

        self.current_pose = Pose2D()
        self.current_pose.x = 0.0
        self.current_pose.y = 0.0
        self.current_pose.theta = 0.0
        self.prev_pose = Pose2D()  # 이전 pose 저장용
        self.robot_id = 0
        self.order_id = 0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_state = 'IDLE'
        self.current_battery_level = 100.0

        self.pose_msg = PickeeMobilePose()
        self.pose_msg.robot_id = self.robot_id
        self.pose_msg.order_id = self.order_id
        self.pose_msg.current_pose = self.current_pose
        self.pose_msg.linear_velocity = self.current_linear_velocity
        self.pose_msg.angular_velocity = self.current_angular_velocity
        self.pose_msg.battery_level = self.current_battery_level
        self.pose_msg.status = self.current_state

        self.last_pose_update_time = time.time()  # 마지막 pose 갱신 시각
        self.pose_msg = None
        self.moving = 0

        # 📩 구독자 설정
        self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.get_current_pose,
            10
        )

        self.create_subscription(
            Twist,
            'cmd_vel_modified',
            self.modify_cmd_vel_callback,
            10
        )

        # 📤 발행자 설정
        self.pose_publisher = self.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )

        # 🕒 0.5초마다 위치 변화 감시
        self.create_timer(0.5, self.check_pose_stability)

        self.get_logger().info('✅ Localization Component 초기화 완료.')

    # -------------------------------------------------------------
    # cmd_vel 구독 콜백
    # -------------------------------------------------------------
    def modify_cmd_vel_callback(self, msg: Twist):
        """현재 선형 및 각속도 업데이트"""
        self.current_linear_velocity = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.current_angular_velocity = msg.angular.z
        self.moving = 1

    # -------------------------------------------------------------
    # amcl_pose 구독 콜백
    # -------------------------------------------------------------
    def get_current_pose(self, msg: PoseWithCovarianceStamped):
        """AMCL Pose 메시지를 기반으로 로봇 위치 추정"""
        # 위치 추출
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 쿼터니언 → yaw(θ) 변환
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = math.atan2(2.0 * qz * qw, 1.0 - 2.0 * (qz ** 2))

        # 위치 변화 감지
        dx = x - self.prev_pose.x
        dy = y - self.prev_pose.y
        dtheta = abs(theta - self.prev_pose.theta)

        # pose가 변했으면 시간 갱신
        if abs(dx) > 0.001 or abs(dy) > 0.001 or dtheta > 0.001:
            self.last_pose_update_time = time.time()

        # 현재 pose 갱신
        self.current_pose.x = x
        self.current_pose.y = y
        self.current_pose.theta = theta

        # PickeeMobilePose 메시지 생성 및 발행
        
        print(type(self.pose_msg))
        self.pose_msg.robot_id = self.robot_id
        self.pose_msg.order_id = self.order_id
        self.pose_msg.current_pose = self.current_pose
        self.pose_msg.linear_velocity = self.current_linear_velocity
        self.pose_msg.angular_velocity = self.current_angular_velocity
        self.pose_msg.battery_level = self.current_battery_level
        self.pose_msg.status = self.current_state

        self.pose_publisher.publish(self.pose_msg)

        # 이전 pose 갱신
        self.prev_pose.x = x
        self.prev_pose.y = y
        self.prev_pose.theta = theta

        # 로그 출력
        self.get_logger().info(
            f'📍 AMCL Pose 업데이트 → x={x:.3f}, y={y:.3f}, θ={math.degrees(theta):.1f}°'
        )

    # -------------------------------------------------------------
    # 2초간 pose 변화가 없으면 속도 초기화
    # -------------------------------------------------------------
    def check_pose_stability(self):
        elapsed = time.time() - self.last_pose_update_time
        if elapsed > 2.0 and self.moving == 1:
            if self.current_linear_velocity != 0.0 or self.current_angular_velocity != 0.0:
                self.current_linear_velocity = 0.0
                self.current_angular_velocity = 0.0
                self.pose_msg.linear_velocity = self.current_linear_velocity
                self.pose_msg.angular_velocity = self.current_angular_velocity
                self.pose_publisher.publish(self.pose_msg)
                self.get_logger().warn(f'⏸️ 2초간 pose 변화 없음 → 속도 초기화됨 (linear=0, angular=0)')
        else:
            
            self.last_pose_update_time = time.time()

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
