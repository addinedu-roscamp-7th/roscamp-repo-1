# typing 모듈의 List 타입 사용을 위해 임포트
from typing import List
# typing 모듈의 Optional 타입 사용을 위해 임포트
from typing import Optional

# rclpy 패키지를 사용하기 위해 임포트
import rclpy
# ROS 2 노드 기반 클래스를 사용하기 위해 임포트
from rclpy.node import Node
# 비동기 서비스 응답을 처리하기 위한 Future 타입 임포트
from rclpy.task import Future

# 팔 자세 상태 메시지 타입을 불러옴
from shopee_interfaces.msg import ArmPoseStatus
# 이미지 좌표 박스 메시지 타입을 불러옴
from shopee_interfaces.msg import BBox
# 팔 작업 상태 메시지 타입을 불러옴
from shopee_interfaces.msg import PackeeArmTaskStatus
# 3차원 점 메시지 타입을 불러옴
from shopee_interfaces.msg import Point3D
# 자세 이동 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmMoveToPose
# 상품 픽업 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmPickProduct
# 상품 담기 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmPlaceProduct


class MockPackeeMain(Node):
    # Packee Main을 모의하여 Packee Arm과 통신을 검증하는 노드

    def __init__(self):
        # 상위 Node 초기화와 노드 이름 설정
        super().__init__('mock_packee_main')

        # 테스트용 로봇 ID 파라미터 선언
        self.declare_parameter('robot_id', 1)
        # 테스트용 주문 ID 파라미터 선언
        self.declare_parameter('order_id', 100)
        # 테스트용 기본 상품 ID 파라미터 선언
        self.declare_parameter('product_id', 501)
        # 기본 팔 구분 파라미터 선언
        self.declare_parameter('arm_side', 'left')
        # 여러 팔 구분 파라미터 선언
        self.declare_parameter('arm_sides', 'left,right')

        # 선언된 로봇 ID 파라미터 값을 정수로 추출
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        # 선언된 주문 ID 파라미터 값을 정수로 추출
        self.order_id = self.get_parameter('order_id').get_parameter_value().integer_value
        # 선언된 기본 상품 ID 파라미터 값을 정수로 추출
        self.base_product_id = self.get_parameter('product_id').get_parameter_value().integer_value
        # 선언된 기본 팔 구분 파라미터 값을 문자열로 추출
        self.arm_side = self.get_parameter('arm_side').get_parameter_value().string_value

        # 다중 팔 구분 파라미터 값을 문자열로 추출
        arm_sides_param = self.get_parameter('arm_sides').get_parameter_value().string_value
        # 콤마로 구분된 팔 구분 목록을 정제하여 리스트로 변환
        parsed_arm_sides = [side.strip() for side in arm_sides_param.split(',') if side.strip()]
        # 변환 결과가 비어 있으면 기본 팔 구분으로 대체
        if not parsed_arm_sides:
            parsed_arm_sides = [self.arm_side]
        # 팔 구분 리스트를 속성으로 저장
        self.arm_sides: List[str] = parsed_arm_sides
        # 현재 테스트 중인 팔 인덱스 초기화
        self.current_arm_index = 0

        # 자세 이동 서비스 클라이언트 생성
        self.move_client = self.create_client(PackeeArmMoveToPose, '/packee/arm/move_to_pose')
        # 상품 픽업 서비스 클라이언트 생성
        self.pick_client = self.create_client(PackeeArmPickProduct, '/packee/arm/pick_product')
        # 상품 담기 서비스 클라이언트 생성
        self.place_client = self.create_client(PackeeArmPlaceProduct, '/packee/arm/place_product')

        # 자세 상태 토픽 구독자 생성
        self.create_subscription(ArmPoseStatus, '/packee/arm/pose_status', self.pose_status_callback, 10)
        # 픽업 상태 토픽 구독자 생성
        self.create_subscription(PackeeArmTaskStatus, '/packee/arm/pick_status', self.pick_status_callback, 10)
        # 담기 상태 토픽 구독자 생성
        self.create_subscription(PackeeArmTaskStatus, '/packee/arm/place_status', self.place_status_callback, 10)

        # 상태 기계 상태를 초기값으로 설정
        self.state = 'wait_services'
        # 현재 진행 중인 Future 객체를 저장할 변수 초기화
        self.current_future: Optional[Future] = None

        # 주기적으로 상태를 처리할 타이머 생성
        self.timer = self.create_timer(0.2, self.process_steps)

        # 테스트 시작 정보를 로그로 출력
        self.get_logger().info(
            f'Packee Arm 통신 모의 테스트를 시작합니다. robot_id={self.robot_id}, order_id={self.order_id}, '
            f'product_id 시작값={self.base_product_id}, arm_sides={self.arm_sides}'
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
            # 모든 서비스 클라이언트가 준비되었는지 확인
            if self.move_client.service_is_ready() and self.pick_client.service_is_ready() and self.place_client.service_is_ready():
                # 서비스 준비 완료 로그 출력
                self.get_logger().info('Packee Arm 서비스가 준비되었습니다. 자세 변경을 요청합니다.')
                # 자세 이동 요청 전송
                self.send_move_request()
                # 다음 상태로 전환
                self.state = 'await_move'
        elif self.state == 'await_move':
            # 자세 이동 서비스 응답을 확인
            self.handle_future('자세 변경')
        elif self.state == 'request_pick':
            # 모든 팔 테스트를 완료했는지 확인
            if self.current_arm_index >= len(self.arm_sides):
                # 전체 테스트 완료 상태로 전환
                self.state = 'completed'
            else:
                # 픽업 요청 전송
                self.send_pick_request()
                # 픽업 응답 대기 상태로 전환
                self.state = 'await_pick'
        elif self.state == 'await_pick':
            # 픽업 서비스 응답을 확인
            self.handle_future('상품 픽업')
        elif self.state == 'request_place':
            # 담기 요청 전송
            self.send_place_request()
            # 담기 응답 대기 상태로 전환
            self.state = 'await_place'
        elif self.state == 'await_place':
            # 담기 서비스 응답을 확인
            self.handle_future('상품 담기')
        elif self.state == 'completed':
            # 테스트 완료 로그 출력
            self.get_logger().info('모의 통신 테스트를 완료했습니다. 노드를 종료합니다.')
            # 종료 진행 상태로 전환
            self.state = 'stopping'
            # ROS 2 세션 종료 요청
            rclpy.shutdown()

    def handle_future(self, action_name: str):
        # 비동기 서비스 호출 결과 처리
        if self.current_future is None or not self.current_future.done():
            # Future가 준비되지 않았다면 대기
            return
        try:
            # Future 결과를 추출
            result = self.current_future.result()
        except Exception as error:
            # 예외 발생 시 로그 출력 후 테스트 종료
            self.get_logger().error(f'{action_name} 서비스 호출 중 예외 발생: {error}')
            self.state = 'completed'
            return

        # 서비스 응답이 거부인지 확인
        if not getattr(result, 'accepted', True):
            # 거부 응답 로그 출력 후 테스트 종료
            self.get_logger().error(f'{action_name} 명령이 Arm 컨트롤러에서 거부되었습니다: message={result.message}')
            self.state = 'completed'
            return

        # 성공 응답 로그 출력
        self.get_logger().info(f'{action_name} 서비스 응답 수신: accepted={result.accepted}, message={result.message}')
        # Future 상태 초기화
        self.current_future = None

        # 다음 상태 전환 로직 처리
        if action_name == '자세 변경':
            self.state = 'request_pick'
        elif action_name == '상품 픽업':
            self.state = 'request_place'
        elif action_name == '상품 담기':
            # 담기 완료 후 다음 팔로 이동
            self.current_arm_index += 1
            if self.current_arm_index < len(self.arm_sides):
                # 다음 팔 테스트 정보를 로그로 출력
                self.get_logger().info(
                    f'다음 팔 테스트 준비: arm_side={self.current_arm_side()}, '
                    f'product_id={self.current_product_id()}'
                )
                # 다음 팔 픽업 요청을 준비
                self.state = 'request_pick'
            else:
                # 모든 테스트가 끝나면 완료 상태로 전환
                self.state = 'completed'

    def send_move_request(self):
        # 자세 변경 서비스 요청
        request = PackeeArmMoveToPose.Request()
        # 로봇 ID 설정
        request.robot_id = self.robot_id
        # 주문 ID 설정
        request.order_id = self.order_id
        # 목표 자세 타입 설정
        request.pose_type = 'cart_view'
        # 비동기 서비스 호출 실행
        self.current_future = self.move_client.call_async(request)

    def send_pick_request(self):
        # 상품 픽업 서비스 요청
        request = PackeeArmPickProduct.Request()
        # 로봇 ID 설정
        request.robot_id = self.robot_id
        # 주문 ID 설정
        request.order_id = self.order_id
        # 테스트할 상품 ID 설정
        request.product_id = self.current_product_id()
        # 테스트할 팔 구분 설정
        request.arm_side = self.current_arm_side()
        # 목표 위치 좌표 설정
        request.target_position = self.create_point3d(0.3, 0.1, 0.75)
        # 상품 박스 위치 정보 설정
        request.bbox = self.create_bbox(120, 180, 250, 320)
        # 비동기 서비스 호출 실행
        self.current_future = self.pick_client.call_async(request)

    def send_place_request(self):
        # 상품 담기 서비스 요청
        request = PackeeArmPlaceProduct.Request()
        # 로봇 ID 설정
        request.robot_id = self.robot_id
        # 주문 ID 설정
        request.order_id = self.order_id
        # 테스트할 상품 ID 설정
        request.product_id = self.current_product_id()
        # 테스트할 팔 구분 설정
        request.arm_side = self.current_arm_side()
        # 박스 위치 좌표 설정
        request.box_position = self.create_point3d(0.5, 0.2, 0.2)
        # 비동기 서비스 호출 실행
        self.current_future = self.place_client.call_async(request)

    def current_arm_side(self) -> str:
        # 현재 테스트할 팔 구분 반환
        if 0 <= self.current_arm_index < len(self.arm_sides):
            # 인덱스가 유효할 때 해당 팔 구분 반환
            return self.arm_sides[self.current_arm_index]
        # 기본 팔 구분 반환
        return self.arm_side

    def current_product_id(self) -> int:
        # 현재 테스트 대상 상품 ID 반환
        return self.base_product_id + self.current_arm_index

    def create_point3d(self, x_value: float, y_value: float, z_value: float) -> Point3D:
        # Point3D 메시지 생성 유틸리티
        point = Point3D()
        # X 좌표 설정
        point.x = x_value
        # Y 좌표 설정
        point.y = y_value
        # Z 좌표 설정
        point.z = z_value
        # 생성된 메시지 반환
        return point

    def create_bbox(self, x1_value: int, y1_value: int, x2_value: int, y2_value: int) -> BBox:
        # BBox 메시지 생성 유틸리티
        bbox = BBox()
        # 좌상단 X 좌표 설정
        bbox.x1 = x1_value
        # 좌상단 Y 좌표 설정
        bbox.y1 = y1_value
        # 우하단 X 좌표 설정
        bbox.x2 = x2_value
        # 우하단 Y 좌표 설정
        bbox.y2 = y2_value
        # 생성된 메시지 반환
        return bbox


def main():
    # rclpy 라이브러리를 초기화
    rclpy.init()
    # MockPackeeMain 노드를 생성
    node = MockPackeeMain()
    try:
        # 노드를 스핀하여 콜백을 처리
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 사용자 인터럽트 시 종료 로그 출력
        node.get_logger().info('Mock Packee Main 노드를 종료합니다.')
    finally:
        # 노드 자원 정리
        node.destroy_node()
        # rclpy가 아직 실행 중인지 확인 후 종료
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    # 메인 진입점에서 main 함수를 호출
    main()
