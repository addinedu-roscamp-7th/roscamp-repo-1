# rclpy 패키지 기능 사용을 위해 임포트
import rclpy
# ROS 2 노드 기본 클래스를 이용하기 위해 임포트
from rclpy.node import Node

# 로봇 팔 자세 상태 메시지 타입을 불러옴
from shopee_interfaces.msg import ArmPoseStatus
# 로봇 팔 작업 상태 메시지 타입을 불러옴
from shopee_interfaces.msg import PackeeArmTaskStatus
# 로봇 팔 자세 이동 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmMoveToPose
# 로봇 팔 픽업 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmPickProduct
# 로봇 팔 담기 서비스 타입을 불러옴
from shopee_interfaces.srv import PackeeArmPlaceProduct


# 허용된 자세 타입 집합 정의
VALID_POSE_TYPES = {'cart_view', 'standby'}
# 허용된 팔 구분 집합 정의
VALID_ARM_SIDES = {'left', 'right'}


class PackeeArmController(Node):
    # Packee 양팔 제어 서비스와 상태 토픽을 담당하는 노드

    def __init__(self):
        # 상위 Node 초기화와 노드 이름 설정
        super().__init__('packee_arm_controller')

        # 명세에 정의된 자세 상태 퍼블리셔 생성
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/packee/arm/pose_status', 10)
        # 명세에 정의된 픽업 상태 퍼블리셔 생성
        self.pick_status_pub = self.create_publisher(PackeeArmTaskStatus, '/packee/arm/pick_status', 10)
        # 명세에 정의된 담기 상태 퍼블리셔 생성
        self.place_status_pub = self.create_publisher(PackeeArmTaskStatus, '/packee/arm/place_status', 10)

        # 자세 이동 서비스 서버를 생성하여 요청 처리 등록
        self.create_service(PackeeArmMoveToPose, '/packee/arm/move_to_pose', self.handle_move_to_pose)
        # 상품 픽업 서비스 서버를 생성하여 요청 처리 등록
        self.create_service(PackeeArmPickProduct, '/packee/arm/pick_product', self.handle_pick_product)
        # 상품 담기 서비스 서버를 생성하여 요청 처리 등록
        self.create_service(PackeeArmPlaceProduct, '/packee/arm/place_product', self.handle_place_product)

        # 노드 초기화 완료 로그 출력
        self.get_logger().info('Packee Arm Controller 노드가 초기화되었습니다.')

    def handle_move_to_pose(self, request, response):
        # Packee Main의 자세 변경 요청 처리
        if request.pose_type not in VALID_POSE_TYPES:
            # 지원되지 않는 자세 타입일 때 실패 상태를 발행
            self.publish_pose_status(
                robot_id=request.robot_id,
                order_id=request.order_id,
                pose_type=request.pose_type,
                status='failed',
                progress=0.0,
                message='알 수 없는 포즈 타입입니다.'
            )
            # 서비스 응답을 거부로 설정
            response.accepted = False
            response.message = '지원되지 않는 포즈 타입입니다.'
            # 거부 응답 반환
            return response

        # 유효한 요청이 수신되었음을 로그로 출력
        self.get_logger().info(
            f'자세 변경 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, pose_type={request.pose_type}'
        )

        # 작업 진행 중 상태를 퍼블리시
        self.publish_pose_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            pose_type=request.pose_type,
            status='in_progress',
            progress=0.5,
            message='자세 이동을 진행 중입니다.'
        )
        # 작업 완료 상태를 퍼블리시
        self.publish_pose_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            pose_type=request.pose_type,
            status='completed',
            progress=1.0,
            message='자세 이동을 완료했습니다.'
        )

        # 서비스 응답을 성공으로 설정
        response.accepted = True
        response.message = '자세 변경 명령이 처리되었습니다.'
        # 성공 응답 반환
        return response

    def handle_pick_product(self, request, response):
        # Packee Main의 상품 픽업 요청 처리
        if request.arm_side not in VALID_ARM_SIDES:
            # 지원되지 않는 팔 구분일 때 실패 상태를 발행
            self.publish_pick_status(
                robot_id=request.robot_id,
                order_id=request.order_id,
                product_id=request.product_id,
                arm_side=request.arm_side,
                status='failed',
                current_phase='planning',
                progress=0.0,
                message='알 수 없는 팔 구분입니다.'
            )
            # 서비스 응답을 거부로 설정
            response.accepted = False
            response.message = '지원되지 않는 팔 구분입니다.'
            # 거부 응답 반환
            return response

        # 유효한 픽업 요청이 들어왔음을 로그로 출력
        self.get_logger().info(
            f'상품 픽업 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, '
            f'product_id={request.product_id}, arm_side={request.arm_side}'
        )

        # 픽업 계획 단계 상태를 퍼블리시
        self.publish_pick_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            product_id=request.product_id,
            arm_side=request.arm_side,
            status='in_progress',
            current_phase='planning',
            progress=0.3,
            message='픽업 경로를 계획 중입니다.'
        )
        # 픽업 파지 단계 상태를 퍼블리시
        self.publish_pick_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            product_id=request.product_id,
            arm_side=request.arm_side,
            status='in_progress',
            current_phase='grasping',
            progress=0.6,
            message='상품을 파지하고 있습니다.'
        )
        # 픽업 완료 상태를 퍼블리시
        self.publish_pick_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            product_id=request.product_id,
            arm_side=request.arm_side,
            status='completed',
            current_phase='done',
            progress=1.0,
            message='상품 픽업을 완료했습니다.'
        )

        # 서비스 응답을 성공으로 설정
        response.accepted = True
        response.message = '상품 픽업 명령이 처리되었습니다.'
        # 성공 응답 반환
        return response

    def handle_place_product(self, request, response):
        # Packee Main의 상품 담기 요청 처리
        if request.arm_side not in VALID_ARM_SIDES:
            # 지원되지 않는 팔 구분일 때 실패 상태를 발행
            self.publish_place_status(
                robot_id=request.robot_id,
                order_id=request.order_id,
                product_id=request.product_id,
                arm_side=request.arm_side,
                status='failed',
                current_phase='planning',
                progress=0.0,
                message='알 수 없는 팔 구분입니다.'
            )
            # 서비스 응답을 거부로 설정
            response.accepted = False
            response.message = '지원되지 않는 팔 구분입니다.'
            # 거부 응답 반환
            return response

        # 유효한 담기 요청이 들어왔음을 로그로 출력
        self.get_logger().info(
            f'상품 담기 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, '
            f'product_id={request.product_id}, arm_side={request.arm_side}'
        )

        # 담기 접근 단계 상태를 퍼블리시
        self.publish_place_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            product_id=request.product_id,
            arm_side=request.arm_side,
            status='in_progress',
            current_phase='approaching',
            progress=0.4,
            message='포장 박스로 이동 중입니다.'
        )
        # 담기 완료 상태를 퍼블리시
        self.publish_place_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            product_id=request.product_id,
            arm_side=request.arm_side,
            status='completed',
            current_phase='done',
            progress=1.0,
            message='상품 담기를 완료했습니다.'
        )

        # 서비스 응답을 성공으로 설정
        response.accepted = True
        response.message = '상품 담기 명령이 처리되었습니다.'
        # 성공 응답 반환
        return response

    def publish_pose_status(self, robot_id, order_id, pose_type, status, progress, message):
        # 자세 상태 토픽 발행 유틸리티
        status_msg = ArmPoseStatus()
        # 로봇 식별자를 메시지에 설정
        status_msg.robot_id = robot_id
        # 주문 식별자를 메시지에 설정
        status_msg.order_id = order_id
        # 자세 타입을 메시지에 설정
        status_msg.pose_type = pose_type
        # 작업 상태를 메시지에 설정
        status_msg.status = status
        # 진행률을 메시지에 설정
        status_msg.progress = progress
        # 상세 메시지를 메시지에 설정
        status_msg.message = message
        # 퍼블리셔를 통해 메시지를 전송
        self.pose_status_pub.publish(status_msg)

    def publish_pick_status(self, robot_id, order_id, product_id, arm_side, status, current_phase, progress, message):
        # 픽업 상태 토픽 발행 유틸리티
        status_msg = PackeeArmTaskStatus()
        # 로봇 식별자를 메시지에 설정
        status_msg.robot_id = robot_id
        # 주문 식별자를 메시지에 설정
        status_msg.order_id = order_id
        # 상품 식별자를 메시지에 설정
        status_msg.product_id = product_id
        # 팔 구분을 메시지에 설정
        status_msg.arm_side = arm_side
        # 작업 상태를 메시지에 설정
        status_msg.status = status
        # 현재 단계를 메시지에 설정
        status_msg.current_phase = current_phase
        # 진행률을 메시지에 설정
        status_msg.progress = progress
        # 상세 메시지를 메시지에 설정
        status_msg.message = message
        # 퍼블리셔를 통해 메시지를 전송
        self.pick_status_pub.publish(status_msg)

    def publish_place_status(self, robot_id, order_id, product_id, arm_side, status, current_phase, progress, message):
        # 담기 상태 토픽 발행 유틸리티
        status_msg = PackeeArmTaskStatus()
        # 로봇 식별자를 메시지에 설정
        status_msg.robot_id = robot_id
        # 주문 식별자를 메시지에 설정
        status_msg.order_id = order_id
        # 상품 식별자를 메시지에 설정
        status_msg.product_id = product_id
        # 팔 구분을 메시지에 설정
        status_msg.arm_side = arm_side
        # 작업 상태를 메시지에 설정
        status_msg.status = status
        # 현재 단계를 메시지에 설정
        status_msg.current_phase = current_phase
        # 진행률을 메시지에 설정
        status_msg.progress = progress
        # 상세 메시지를 메시지에 설정
        status_msg.message = message
        # 퍼블리셔를 통해 메시지를 전송
        self.place_status_pub.publish(status_msg)


def main():
    # rclpy 클라이언트 라이브러리를 초기화
    rclpy.init()
    # PackeeArmController 노드 인스턴스를 생성
    node = PackeeArmController()
    try:
        # 노드를 스핀하여 콜백을 처리
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 사용자가 인터럽트를 보냈을 때 종료 로그 출력
        node.get_logger().info('Packee Arm Controller 노드가 종료됩니다.')
    finally:
        # 노드를 정리하고 자원 해제
        node.destroy_node()
        # rclpy를 종료하여 ROS 2 세션 정리
        rclpy.shutdown()


if __name__ == '__main__':
    # 메인 진입점에서 main 함수를 호출
    main()
