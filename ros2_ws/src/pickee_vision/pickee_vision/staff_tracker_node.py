
import rclpy
from rclpy.node import Node
import time
import random

from shopee_interfaces.srv import (
    PickeeVisionSetMode, PickeeVisionRegisterStaff, PickeeVisionTrackStaff, PickeeTtsRequest
)
from shopee_interfaces.msg import PickeeVisionStaffRegister, PickeeVisionStaffLocation, Point2D

class StaffTrackerNode(Node):
    """
    직원 등록, 추종, 모드 변경 등 복합적인 시나리오를 처리하는 노드입니다.
    """
    def __init__(self):
        super().__init__('staff_tracker_node')
        self.current_mode = "navigation"
        self.is_tracking = False

        # 서비스 서버
        self.create_service(PickeeVisionSetMode, '/pickee/vision/set_mode', self.set_mode_callback)
        self.create_service(PickeeVisionRegisterStaff, '/pickee/vision/register_staff', self.register_staff_callback)
        self.create_service(PickeeVisionTrackStaff, '/pickee/vision/track_staff', self.track_staff_callback)

        # 퍼블리셔
        self.reg_result_pub = self.create_publisher(PickeeVisionStaffRegister, '/pickee/vision/register_staff_result', 10)
        self.location_pub = self.create_publisher(PickeeVisionStaffLocation, '/pickee/vision/staff_location', 10)

        # 서비스 클라이언트 (다른 노드의 서비스를 호출하기 위함)
        self.tts_client = self.create_client(PickeeTtsRequest, '/pickee/tts_request')

        # 추종 위치 발행을 위한 타이머
        self.track_timer = self.create_timer(1.0, self.publish_staff_location)

        self.get_logger().info('Staff Tracker Node has been started.')

    def set_mode_callback(self, request, response):
        self.current_mode = request.mode
        self.get_logger().info(f'Vision mode switched to {self.current_mode}')
        response.success = True
        response.message = f"Vision mode switched to {self.current_mode}"
        return response

    def register_staff_callback(self, request, response):
        self.get_logger().info('Staff registration process started.')
        
        # 음성 안내 요청 (비동기 처리는 생략하고 간단히 구현)
        self.call_tts_service("카메라 정면을 봐주세요. 3초 후 정면을 등록합니다.")
        time.sleep(3)
        self.get_logger().info('Front view registered.')

        self.call_tts_service("뒤로 돌아주세요. 3초 후 후면을 등록합니다.")
        time.sleep(3)
        self.get_logger().info('Back view registered.')

        # 등록 결과 발행
        result_msg = PickeeVisionStaffRegister()
        result_msg.robot_id = request.robot_id
        result_msg.success = True
        result_msg.message = "Staff registration successful."
        self.reg_result_pub.publish(result_msg)
        self.get_logger().info('Published staff registration result.')

        response.accepted = True
        response.message = "Staff registration process accepted."
        return response

    def track_staff_callback(self, request, response):
        self.is_tracking = request.track
        status = "Started" if self.is_tracking else "Stopped"
        self.get_logger().info(f'{status} tracking staff.')
        response.success = True
        response.message = f"{status} tracking staff."
        return response

    def publish_staff_location(self):
        if self.is_tracking and self.current_mode == "track_staff":
            msg = PickeeVisionStaffLocation()
            msg.robot_id = 1
            msg.relative_position = Point2D(x=random.uniform(2.0, 3.0), y=random.uniform(-0.5, 0.5))
            msg.distance = float(msg.relative_position.x)
            msg.is_tracking = True
            self.location_pub.publish(msg)
            self.get_logger().info(f'Publishing staff location: dist={msg.distance:.2f}')

    def call_tts_service(self, text):
        if not self.tts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('TTS service not available!')
            return
        
        request = PickeeTtsRequest.Request()
        request.text_to_speak = text
        self.tts_client.call_async(request)
        self.get_logger().info(f'Requested TTS: "{text}"')


def main(args=None):
    rclpy.init(args=args)
    node = StaffTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
