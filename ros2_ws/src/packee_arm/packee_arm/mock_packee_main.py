from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.task import Future

from shopee_interfaces.msg import ArmPoseStatus
from shopee_interfaces.msg import BBox
from shopee_interfaces.msg import PackeeArmTaskStatus
from shopee_interfaces.msg import Point3D
from shopee_interfaces.srv import PackeeArmMoveToPose
from shopee_interfaces.srv import PackeeArmPickProduct
from shopee_interfaces.srv import PackeeArmPlaceProduct


class MockPackeeMain(Node):
    # Packee Main을 모의하여 Packee Arm과 통신을 검증하는 노드

    def __init__(self):
        super().__init__('mock_packee_main')

        # 테스트 파라미터 설정
        self.declare_parameter('robot_id', 1)
        self.declare_parameter('order_id', 100)
        self.declare_parameter('product_id', 501)
        self.declare_parameter('arm_side', 'left')

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        self.order_id = self.get_parameter('order_id').get_parameter_value().integer_value
        self.product_id = self.get_parameter('product_id').get_parameter_value().integer_value
        self.arm_side = self.get_parameter('arm_side').get_parameter_value().string_value

        # 서비스 클라이언트 생성
        self.move_client = self.create_client(PackeeArmMoveToPose, '/packee/arm/move_to_pose')
        self.pick_client = self.create_client(PackeeArmPickProduct, '/packee/arm/pick_product')
        self.place_client = self.create_client(PackeeArmPlaceProduct, '/packee/arm/place_product')

        # 상태 토픽 구독
        self.create_subscription(ArmPoseStatus, '/packee/arm/pose_status', self.pose_status_callback, 10)
        self.create_subscription(PackeeArmTaskStatus, '/packee/arm/pick_status', self.pick_status_callback, 10)
        self.create_subscription(PackeeArmTaskStatus, '/packee/arm/place_status', self.place_status_callback, 10)

        # 내부 상태 변수
        self.state = 'wait_services'
        self.current_future: Optional[Future] = None

        # 주기적으로 상태를 확인하여 다음 단계 진행
        self.timer = self.create_timer(0.2, self.process_steps)

        self.get_logger().info(
            f'Packee Arm 통신 모의 테스트를 시작합니다. robot_id={self.robot_id}, order_id={self.order_id}, '
            f'product_id={self.product_id}, arm_side={self.arm_side}'
        )

    def pose_status_callback(self, status_msg: ArmPoseStatus):
        # 자세 상태 수신 로그
        self.get_logger().info(
            f'[PoseStatus] pose_type={status_msg.pose_type}, status={status_msg.status}, '
            f'progress={status_msg.progress:.2f}, message={status_msg.message}'
        )

    def pick_status_callback(self, status_msg: PackeeArmTaskStatus):
        # 픽업 상태 수신 로그
        self.get_logger().info(
            f'[PickStatus] product_id={status_msg.product_id}, phase={status_msg.current_phase}, '
            f'status={status_msg.status}, progress={status_msg.progress:.2f}, message={status_msg.message}'
        )

    def place_status_callback(self, status_msg: PackeeArmTaskStatus):
        # 담기 상태 수신 로그
        self.get_logger().info(
            f'[PlaceStatus] product_id={status_msg.product_id}, phase={status_msg.current_phase}, '
            f'status={status_msg.status}, progress={status_msg.progress:.2f}, message={status_msg.message}'
        )

    def process_steps(self):
        # 상태 기계 기반 테스트 진행
        if self.state == 'wait_services':
            if self.move_client.service_is_ready() and self.pick_client.service_is_ready() and self.place_client.service_is_ready():
                self.get_logger().info('Packee Arm 서비스가 준비되었습니다. 자세 변경을 요청합니다.')
                self.send_move_request()
                self.state = 'await_move'
        elif self.state == 'await_move':
            self.handle_future('자세 변경')
        elif self.state == 'request_pick':
            self.send_pick_request()
            self.state = 'await_pick'
        elif self.state == 'await_pick':
            self.handle_future('상품 픽업')
        elif self.state == 'request_place':
            self.send_place_request()
            self.state = 'await_place'
        elif self.state == 'await_place':
            self.handle_future('상품 담기')
        elif self.state == 'completed':
            self.get_logger().info('모의 통신 테스트를 완료했습니다. 노드를 종료합니다.')
            self.state = 'stopping'
            rclpy.shutdown()

    def handle_future(self, action_name: str):
        # 비동기 서비스 호출 결과 처리
        if self.current_future is None or not self.current_future.done():
            return
        try:
            result = self.current_future.result()
        except Exception as error:
            self.get_logger().error(f'{action_name} 서비스 호출 중 예외 발생: {error}')
            self.state = 'completed'
            return

        if not getattr(result, 'accepted', True):
            self.get_logger().error(f'{action_name} 명령이 Arm 컨트롤러에서 거부되었습니다: message={result.message}')
            self.state = 'completed'
            return

        self.get_logger().info(f'{action_name} 서비스 응답 수신: accepted={result.accepted}, message={result.message}')
        self.current_future = None

        if action_name == '자세 변경':
            self.state = 'request_pick'
        elif action_name == '상품 픽업':
            self.state = 'request_place'
        elif action_name == '상품 담기':
            self.state = 'completed'

    def send_move_request(self):
        # 자세 변경 서비스 요청
        request = PackeeArmMoveToPose.Request()
        request.robot_id = self.robot_id
        request.order_id = self.order_id
        request.pose_type = 'cart_view'
        self.current_future = self.move_client.call_async(request)

    def send_pick_request(self):
        # 상품 픽업 서비스 요청
        request = PackeeArmPickProduct.Request()
        request.robot_id = self.robot_id
        request.order_id = self.order_id
        request.product_id = self.product_id
        request.arm_side = self.arm_side
        request.target_position = self.create_point3d(0.3, 0.1, 0.75)
        request.bbox = self.create_bbox(120, 180, 250, 320)
        self.current_future = self.pick_client.call_async(request)

    def send_place_request(self):
        # 상품 담기 서비스 요청
        request = PackeeArmPlaceProduct.Request()
        request.robot_id = self.robot_id
        request.order_id = self.order_id
        request.product_id = self.product_id
        request.arm_side = self.arm_side
        request.box_position = self.create_point3d(0.5, 0.2, 0.2)
        self.current_future = self.place_client.call_async(request)

    def create_point3d(self, x_value: float, y_value: float, z_value: float) -> Point3D:
        # Point3D 메시지 생성 유틸리티
        point = Point3D()
        point.x = x_value
        point.y = y_value
        point.z = z_value
        return point

    def create_bbox(self, x1_value: int, y1_value: int, x2_value: int, y2_value: int) -> BBox:
        # BBox 메시지 생성 유틸리티
        bbox = BBox()
        bbox.x1 = x1_value
        bbox.y1 = y1_value
        bbox.x2 = x2_value
        bbox.y2 = y2_value
        return bbox


def main():
    rclpy.init()
    node = MockPackeeMain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Mock Packee Main 노드를 종료합니다.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
