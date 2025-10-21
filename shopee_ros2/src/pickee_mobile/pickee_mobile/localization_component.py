import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose, Pose2D
import math
from geometry_msgs.msg import PoseWithCovarianceStamped

class LocalizationComponent:
    '''
    Pickee Mobile의 위치 추정 컴포넌트.
    센서 데이터를 기반으로 로봇의 현재 위치를 추정하고,
    추정된 위치 정보를 Pickee Main Controller에 보고합니다.
    '''

    def __init__(self, node: Node):
        print('LocalizationComponent init')
        self.node = node
        self.pose_publisher = self.node.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )

        self.get_current_pose_subscriber = self.node.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.get_current_pose,
            10
        )
        self.node.get_logger().info('Localization Component 초기화 완료.')

        # 임시 위치 데이터 (실제 구현 시 센서 데이터로 대체)
        self.robot_id = 0
        self.order_id = 0
        self.current_pose = Pose2D()
        self.current_pose.x = 0.0
        self.current_pose.y = 0.0
        self.current_pose.theta = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_battery_level = 100.0
        self.current_state = 'IDLE'

    def update_pose(self, current_state: str):
        '''
        로봇의 현재 위치를 업데이트하고 발행합니다.
        실제 구현 시 센서 데이터를 처리하여 위치를 추정합니다.
        '''
        # 임시 로직: 간단한 이동 시뮬레이션
        # 실제 구현 시 센서 융합 및 위치 추정 알고리즘 적용

        self.node.get_logger().info(f'Publishing pose')

        if current_state == 'MOVING':
            self.current_pose.x += 0.01 * math.cos(self.current_pose.theta)
            self.current_pose.y += 0.01 * math.sin(self.current_pose.theta)
            self.current_pose.theta += 0.005
            self.current_linear_velocity = 0.1
            self.current_angular_velocity = 0.05
            self.current_battery_level -= 0.01 # 배터리 소모 시뮬레이션
        else:
            self.current_linear_velocity = 0.0
            self.current_angular_velocity = 0.0

        if self.current_battery_level < 0:
            self.current_battery_level = 0.0

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
        self.node.get_logger().info(f'robot_id: {self.robot_id}, order_id: {self.order_id}, pose: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}, {self.current_pose.theta:.2f}), linear_velocity: {self.current_linear_velocity:.2f}, angular_velocity: {self.current_angular_velocity:.2f}, battery: {self.current_battery_level:.2f}%, state: {self.current_state}')

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
        self.node.get_logger().info(
            f'📡 AMCL Pose 업데이트: x={x:.3f}, y={y:.3f}, theta={math.degrees(theta):.1f}°'
        )

    def get_battery_level(self):
        '''
        현재 배터리 잔량을 반환합니다.
        '''
        return self.battery_level
    