import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import PickeeMobilePose, PickeeMobileSpeedControl, PickeeMoveStatus
from shopee_interfaces.srv import PickeeMobileMoveToLocation, PickeeMobileUpdateGlobalPath
from geometry_msgs.msg import Pose2D
import time

class MockMainController(Node):
    '''
    Pickee Mobile Controller와 통신하는 Mock Main Controller 노드.
    테스트 및 시뮬레이션 목적으로 사용됩니다.
    '''

    def __init__(self):
        super().__init__('mock_main_controller')
        self.get_logger().info('Mock Main Controller 노드가 시작되었습니다.')

        # Publisher 초기화
        self.speed_control_publisher = self.create_publisher(
            PickeeMobileSpeedControl,
            '/pickee/mobile/speed_control',
            10
        )
        self.pose_publisher = self.create_publisher( # PickeeMobilePose 발행 (요청에 따라 추가)
            PickeeMobilePose,
            '/pickee/mobile/pose',
            10
        )

        # Subscriber 초기화
        self.move_status_subscriber = self.create_subscription(
            PickeeMoveStatus,
            '/pickee/mobile/local_path',
            self.move_status_callback,
            10
        )

        # Service Client 초기화
        self.move_to_location_client = self.create_client(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location'
        )
        self.update_global_path_client = self.create_client(
            PickeeMobileUpdateGlobalPath,
            '/pickee/mobile/update_global_path'
        )

        # 서비스 서버가 준비될 때까지 대기
        while not self.move_to_location_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move_to_location 서비스 대기 중...')
        while not self.update_global_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('update_global_path 서비스 대기 중...')

        # 타이머 설정
        self.timer = self.create_timer(1.0, self.timer_callback) # 1초마다 실행

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.robot_state = 'IDLE'

    def move_status_callback(self, msg: PickeeMoveStatus):
        '''
        PickeeMoveStatus 메시지 수신 시 호출됩니다.
        '''
        self.get_logger().info(f'Move Status 수신: target_x={msg.target_x:.2f}, dist={msg.distance_to_target:.2f}, arrived={msg.is_arrived}')

    def timer_callback(self):
        '''
        주기적으로 실행되는 타이머 콜백.
        '''
        # PickeeMobileSpeedControl 발행 예시
        speed_control_msg = PickeeMobileSpeedControl()
        speed_control_msg.speed_mode = 'normal'
        speed_control_msg.target_speed = 0.5
        self.speed_control_publisher.publish(speed_control_msg)
        self.get_logger().info(f'Speed Control 발행: mode={speed_control_msg.speed_mode}, speed={speed_control_msg.target_speed}')

        # PickeeMobilePose 발행 예시 (요청에 따라 추가)
        pose_msg = PickeeMobilePose()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.current_pose.x = self.current_x
        pose_msg.current_pose.y = self.current_y
        pose_msg.current_pose.theta = self.current_theta
        pose_msg.linear_velocity = 0.0
        pose_msg.angular_velocity = 0.0
        pose_msg.battery_level = 90.0
        pose_msg.status = self.robot_state
        self.pose_publisher.publish(pose_msg)
        self.get_logger().info(f'Mock Pose 발행: x={pose_msg.current_pose.x:.2f}, y={pose_msg.current_pose.y:.2f}')


        # PickeeMobileMoveToLocation 서비스 요청 예시 (10초마다 한 번)
        if self.get_clock().now().nanoseconds % (10 * 10**9) < 10**9: # 대략 10초마다
            self.send_move_to_location_request(1.0, 2.0, 0.0)

        # PickeeMobileUpdateGlobalPath 서비스 요청 예시 (20초마다 한 번)
        if self.get_clock().now().nanoseconds % (20 * 10**9) < 10**9: # 대략 20초마다
            new_global_path = [Pose2D(x=0.0, y=0.0, theta=0.0), Pose2D(x=3.0, y=4.0, theta=1.57)]
            self.send_update_global_path_request(new_global_path)

    def send_move_to_location_request(self, x, y, theta):
        '''
        PickeeMobileMoveToLocation 서비스 요청을 보냅니다.
        '''
        request = PickeeMobileMoveToLocation.Request()
        request.target_pose.x = x
        request.target_pose.y = y
        request.target_pose.theta = theta
        # 전역 경로 예시 (간단하게 시작점과 목표점만 포함)
        request.global_path = [Pose2D(x=self.current_x, y=self.current_y, theta=self.current_theta), Pose2D(x=x, y=y, theta=theta)]

        self.future = self.move_to_location_client.call_async(request)
        self.future.add_done_callback(self.move_to_location_response_callback)

    def move_to_location_response_callback(self, future):
        '''
        PickeeMobileMoveToLocation 서비스 응답 처리 콜백.
        '''
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'이동 명령 성공: {response.message}')
            else:
                self.get_logger().warn(f'이동 명령 실패: {response.message}')
        except Exception as e:
            self.get_logger().error(f'이동 명령 서비스 호출 실패: {e}')

    def send_update_global_path_request(self, new_global_path: list):
        '''
        PickeeMobileUpdateGlobalPath 서비스 요청을 보냅니다.
        '''
        request = PickeeMobileUpdateGlobalPath.Request()
        request.new_global_path = new_global_path

        self.future = self.update_global_path_client.call_async(request)
        self.future.add_done_callback(self.update_global_path_response_callback)

    def update_global_path_response_callback(self, future):
        '''
        PickeeMobileUpdateGlobalPath 서비스 응답 처리 콜백.
        '''
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'전역 경로 업데이트 성공: {response.message}')
            else:
                self.get_logger().warn(f'전역 경로 업데이트 실패: {response.message}')
        except Exception as e:
            self.get_logger().error(f'전역 경로 업데이트 서비스 호출 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    mock_main_controller = MockMainController()
    rclpy.spin(mock_main_controller)
    mock_main_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
