import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from shopee_interfaces.msg import PickeeMobileSpeedControl, PickeeMoveStatus
import math

class MotionControlComponent:
    '''
    Pickee Mobile의 모션 제어 컴포넌트.
    경로 계획 컴포넌트로부터 지역 경로를 수신하고, 로봇의 선속도 및 각속도를 계산하여
    Motor Driver 및 Steering Actuator에 전달합니다.
    '''

    def __init__(self, node: Node):
        self.node = node
        self.cmd_vel_subscriber = self.node.create_subscription(
            Twist,
            '/cmd_vel',
            self.speed_publisher_callback,
            10
        )
        self.cmd_vel_modified_publisher = self.node.create_publisher(
            Twist,
            '/cmd_vel_modified',
            10
        )
        self.speed_control_subscriber = self.node.create_subscription(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            self.speed_control_callback,
            10
        )
        self.local_path_subscriber = self.node.create_subscription(
            PickeeMoveStatus,
            '/pickee/mobile/local_path',
            self.local_path_callback,
            10
        )
        self.node.get_logger().info('Motion Control Component 초기화 완료.')

        self.current_speed_mode = 'normal' # normal, decelerate, stop
        self.target_speed = 1.0
        self.current_move_status = None
    

    def speed_control_callback(self, msg: PickeeMobileSpeedControl):
        '''
        Pickee Main Controller로부터 속도 제어 명령을 수신합니다.
        '''
        self.current_speed_mode = msg.speed_mode
        
        self.node.get_logger().info(f'속도 제어 명령 수신: mode={self.current_speed_mode}, target_speed={self.target_speed}')

        if self.current_speed_mode == 'stop':
            self.node.get_logger().info('로봇 정지 모드로 전환됨.')
            self.target_speed = 0.0
            self.current_move_status = 'stop'

        elif self.current_speed_mode == 'normal':
            self.node.get_logger().info('로봇 정상 주행 모드로 전환됨.')
            self.target_speed = 1.0
            self.current_move_status = 'moving'

        elif self.current_speed_mode == 'decelerate':
            self.node.get_logger().info('로봇 감속 모드로 전환됨.')
            self.target_speed = msg.target_speed
            self.current_move_status = 'decelerate moving'

        else:
            self.node.get_logger().warn(f'알 수 없는 속도 모드 수신: {self.current_speed_mode}')
            self.target_speed = msg.target_speed
            self.current_move_status = 'unknown mode received'

    def speed_publisher_callback(self, msg: Twist):
        '''
        /cmd_vel 토픽을 수신하여 /cmd_vel_modified로 발행합니다.
        '''
        modified_twist = Twist()
        scale = self.target_speed

        modified_twist.linear.x = msg.linear.x * scale
        modified_twist.linear.y = msg.linear.y * scale
        modified_twist.linear.z = msg.linear.z * scale

        modified_twist.angular.x = msg.angular.x * scale
        modified_twist.angular.y = msg.angular.y * scale
        modified_twist.angular.z = msg.angular.z * scale

        self.cmd_vel_modified_publisher.publish(modified_twist)
        # self.node.get_logger().info(f'Cmd_vel Modified 발행: linear.x={modified_twist.linear.x:.2f}, angular.z={modified_twist.angular.z:.2f}')

    def local_path_callback(self, msg: PickeeMoveStatus):
        '''
        Path Planning Component로부터 지역 경로 (이동 상태)를 수신합니다.
        '''
        self.current_move_status = msg

    def control_robot(self):
        '''
        로봇의 선속도 및 각속도를 계산하여 발행합니다.
        '''
        twist_msg = Twist()

        if self.current_speed_mode == 'stop':
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
        elif self.current_speed_mode == 'decelerate':
            # 감속 로직 (예: 현재 속도에서 target_speed까지 점진적으로 감소)
            twist_msg.linear.x = min(0.1, self.target_speed) # 임시 감속
            twist_msg.angular.z = 0.0
        else: # normal
            if self.current_move_status and not self.current_move_status.is_arrived:
                # 임시 제어 로직: 목표 방향으로 회전 후 직진
                # 실제 구현 시 PID, MPC 등의 제어 알고리즘 적용
                current_x = self.current_move_status.current_x
                current_y = self.current_move_status.current_y
                current_theta = self.current_move_status.current_theta
                target_x = self.current_move_status.target_x
                target_y = self.current_move_status.target_y

                angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
                angle_diff = angle_to_target - current_theta

                # 각도 차이를 -pi ~ pi 범위로 정규화
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                if abs(angle_diff) > 0.1: # 목표 방향과 10도 이상 차이 나면 회전
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.5 * angle_diff # 비례 제어
                else: # 목표 방향과 일치하면 직진
                    twist_msg.linear.x = 0.2 # 임시 선속도
                    twist_msg.angular.z = 0.0
            else:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

        self.cmd_vel_publisher.publish(twist_msg)
        # self.node.get_logger().info(f'Cmd_vel 발행: linear.x={twist_msg.linear.x:.2f}, angular.z={twist_msg.angular.z:.2f}')

    def get_current_speed_mode(self):
        '''
        현재 속도 제어 모드를 반환합니다.
        '''
        return self.current_speed_mode
