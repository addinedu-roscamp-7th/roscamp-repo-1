import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose
import math

class LocalizationComponent:
    '''
    Pickee Mobile의 위치 추정 컴포넌트.
    센서 데이터를 기반으로 로봇의 현재 위치를 추정하고,
    추정된 위치 정보를 Pickee Main Controller에 보고합니다.
    '''

    def __init__(self, node: Node):
        self.node = node
        self.pose_publisher = self.node.create_publisher(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )
        self.node.get_logger().info('Localization Component 초기화 완료.')

        # 임시 위치 데이터 (실제 구현 시 센서 데이터로 대체)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_linear_speed = 0.0
        self.current_angular_speed = 0.0
        self.battery_percentage = 100.0

    def update_pose(self, robot_state: str):
        '''
        로봇의 현재 위치를 업데이트하고 발행합니다.
        실제 구현 시 센서 데이터를 처리하여 위치를 추정합니다.
        '''
        # 임시 로직: 간단한 이동 시뮬레이션
        # 실제 구현 시 센서 융합 및 위치 추정 알고리즘 적용
        if robot_state == 'MOVING':
            self.current_x += 0.01 * math.cos(self.current_theta)
            self.current_y += 0.01 * math.sin(self.current_theta)
            self.current_theta += 0.005
            self.current_linear_speed = 0.1
            self.current_angular_speed = 0.05
            self.battery_percentage -= 0.01 # 배터리 소모 시뮬레이션
        else:
            self.current_linear_speed = 0.0
            self.current_angular_speed = 0.0

        if self.battery_percentage < 0:
            self.battery_percentage = 0.0

        pose_msg = PickeeMobilePose()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.x = self.current_x
        pose_msg.y = self.current_y
        pose_msg.theta = self.current_theta
        pose_msg.linear_speed = self.current_linear_speed
        pose_msg.angular_speed = self.current_angular_speed
        pose_msg.battery_percentage = self.battery_percentage
        pose_msg.robot_state = robot_state # 현재 로봇 상태를 메시지에 포함

        self.pose_publisher.publish(pose_msg)
        # self.node.get_logger().info(f'Pose 발행: x={pose_msg.x:.2f}, y={pose_msg.y:.2f}, theta={pose_msg.theta:.2f}, battery={pose_msg.battery_percentage:.1f}%')

    def get_current_pose(self):
        '''
        현재 추정된 로봇의 위치 정보를 반환합니다.
        '''
        return self.current_x, self.current_y, self.current_theta

    def get_battery_percentage(self):
        '''
        현재 배터리 잔량을 반환합니다.
        '''
        return self.battery_percentage
