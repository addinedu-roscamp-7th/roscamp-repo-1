import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMoveStatus, Pose2D, PickeeMobilePose
from shopee_interfaces.srv import PickeeMobileMoveToLocation, PickeeMobileUpdateGlobalPath
from geometry_msgs.msg import Pose2D # ROS2 표준 메시지 사용
import math # math 모듈 추가
from pickee_mobile.states import MovingState # MovingState 임포트

class PathPlanningComponent:
    '''
    Pickee Mobile의 경로 계획 컴포넌트.
    전역 경로를 수신하고, 실시간 장애물 정보를 반영하여 지역 경로를 계획합니다.
    '''

    def __init__(self, node: Node):
        self.node = node
        self.local_path_publisher = self.node.create_publisher(
            PickeeMoveStatus,
            '/pickee/mobile/local_path', # 내부 토픽으로 지역 경로 발행
            10
        )
        self.node.get_logger().info('Path Planning Component 초기화 완료.')

        self.global_path = [] # 전역 경로 (Pose2D 리스트)
        self.target_pose = None # 최종 목표 포즈 (Pose2D)

        # 서비스 서버 초기화 (Pickee Main Controller로부터 명령 수신)
        self.move_to_location_service = self.node.create_service(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location',
            self.move_to_location_callback
        )
        self.update_global_path_service = self.node.create_service(
            PickeeMobileUpdateGlobalPath,
            '/pickee/mobile/update_global_path',
            self.update_global_path_callback
        )

        # 파라미터 선언 (arrival_position_tolerance)
        self.node.declare_parameter('arrival_position_tolerance', 0.05)

    def move_to_location_callback(self, request, response):
        '''
        Pickee Main Controller로부터 이동 명령을 수신합니다.
        '''

        self.robot_id = request.robot_id
        self.order_id = request.order_id
        self.location_id = request.location_id
        self.target_pose = request.target_pose
        self.global_path = request.global_path # 전역 경로 저장
        self.navigation_mode = request.navigation_mode

        self.node.get_logger().info(f'이동 명령 수신: robot_id={self.robot_id}, order_id={self.order_id}, location_id={self.location_id}, target_pose=({self.target_pose.x}, {self.target_pose.y}, {self.target_pose.theta}), navigation_mode={self.navigation_mode}')
        self.node.get_logger().info(f'전역 경로 길이: {len(self.global_path)}')
        self.node.get_logger().info(f'전역 경로: {[ (pose.x, pose.y, pose.theta) for pose in self.global_path ]}')

        # self.node.get_logger().info(f'이동 명령 수신: target_pose=({request.target_pose.x}, {request.target_pose.y}, {request.target_pose.theta})')
        # self.target_pose = request.target_pose
        # self.global_path = request.global_path # 전역 경로 저장

        

        # # mobile_controller의 상태 기계를 MOVING 상태로 전환
        # self.node.state_machine.transition_to(MovingState(self.node)) # self.node.state_machine에 접근

        response.success = True
        response.message = '이동 명령 수신 완료.'
        return response

    def update_global_path_callback(self, request, response):
        '''
        Pickee Main Controller로부터 전역 경로 업데이트 명령을 수신합니다.
        '''

        self.robot_id = request.robot_id
        self.order_id = request.order_id
        self.location_id = request.location_id
        self.global_path = request.global_path # 전역 경로 저장

        self.node.get_logger().info(f'이동 명령 수신: robot_id={self.robot_id}, order_id={self.order_id}, location_id={self.location_id}, {self.target_pose.y}, {self.target_pose.theta})')
        self.node.get_logger().info(f'전역 경로 길이: {len(self.global_path)}')
        self.node.get_logger().info(f'전역 경로: {[ (pose.x, pose.y, pose.theta) for pose in self.global_path ]}')


        self.response.success = True
        self.response.message = '이동 명령 수신 완료.'

        # self.node.get_logger().info('전역 경로 업데이트 명령 수신.')
        # self.global_path = request.new_global_path # 전역 경로 업데이트

        response.success = True
        response.message = '전역 경로 업데이트 완료.'
        return response

    def plan_local_path(self, current_pose: tuple, obstacles: list = []) -> PickeeMoveStatus: # 반환 타입 명시
        '''
        현재 위치와 장애물 정보를 기반으로 지역 경로를 계획하고 발행합니다.
        '''
        if not self.target_pose:
            return None # 목표 포즈가 없으면 None 반환

        # 임시 로직: 현재 위치에서 목표 포즈까지의 간단한 이동 상태 발행
        # 실제 구현 시 DWA, TEB 등의 알고리즘 적용
        move_status_msg = PickeeMoveStatus()
        # move_status_msg.header.stamp = self.node.get_clock().now().to_msg()
        move_status_msg.current_x = current_pose[0]
        move_status_msg.current_y = current_pose[1]
        move_status_msg.current_theta = current_pose[2]
        move_status_msg.target_x = self.target_pose.x
        move_status_msg.target_y = self.target_pose.y
        move_status_msg.target_theta = self.target_pose.theta
        move_status_msg.distance_to_target = ((self.target_pose.x - current_pose[0])**2 + \
                                              (self.target_pose.y - current_pose[1])**2)**0.5
        move_status_msg.is_arrived = move_status_msg.distance_to_target < self.node.get_parameter('arrival_position_tolerance').get_parameter_value().double_value

        self.local_path_publisher.publish(move_status_msg)
        # self.node.get_logger().info(f'Local Path 발행: target_x={move_status_msg.target_x:.2f}, dist={move_status_msg.distance_to_target:.2f}')
        return move_status_msg # move_status_msg 반환

    def get_target_pose(self):
        '''
        현재 목표 포즈를 반환합니다.
        '''
        return self.target_pose
