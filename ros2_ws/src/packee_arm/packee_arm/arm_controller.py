import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import ArmPoseStatus
from shopee_interfaces.msg import PackeeArmTaskStatus
from shopee_interfaces.srv import PackeeArmMoveToPose
from shopee_interfaces.srv import PackeeArmPickProduct
from shopee_interfaces.srv import PackeeArmPlaceProduct


VALID_POSE_TYPES = {'cart_view', 'standby'}
VALID_ARM_SIDES = {'left', 'right'}


class PackeeArmController(Node):
    # Packee 양팔 제어 서비스와 상태 토픽을 담당하는 노드

    def __init__(self):
        super().__init__('packee_arm_controller')

        # 명세에 정의된 토픽 퍼블리셔 생성
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/packee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(PackeeArmTaskStatus, '/packee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(PackeeArmTaskStatus, '/packee/arm/place_status', 10)

        # 명세에 정의된 서비스 서버 생성
        self.create_service(PackeeArmMoveToPose, '/packee/arm/move_to_pose', self.handle_move_to_pose)
        self.create_service(PackeeArmPickProduct, '/packee/arm/pick_product', self.handle_pick_product)
        self.create_service(PackeeArmPlaceProduct, '/packee/arm/place_product', self.handle_place_product)

        self.get_logger().info('Packee Arm Controller 노드가 초기화되었습니다.')

    def handle_move_to_pose(self, request, response):
        # Packee Main의 자세 변경 요청 처리
        if request.pose_type not in VALID_POSE_TYPES:
            self.publish_pose_status(
                robot_id=request.robot_id,
                order_id=request.order_id,
                pose_type=request.pose_type,
                status='failed',
                progress=0.0,
                message='알 수 없는 포즈 타입입니다.'
            )
            response.accepted = False
            response.message = '지원되지 않는 포즈 타입입니다.'
            return response

        self.get_logger().info(
            f'자세 변경 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, pose_type={request.pose_type}'
        )

        self.publish_pose_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            pose_type=request.pose_type,
            status='in_progress',
            progress=0.5,
            message='자세 이동을 진행 중입니다.'
        )
        self.publish_pose_status(
            robot_id=request.robot_id,
            order_id=request.order_id,
            pose_type=request.pose_type,
            status='completed',
            progress=1.0,
            message='자세 이동을 완료했습니다.'
        )

        response.accepted = True
        response.message = '자세 변경 명령이 처리되었습니다.'
        return response

    def handle_pick_product(self, request, response):
        # Packee Main의 상품 픽업 요청 처리
        if request.arm_side not in VALID_ARM_SIDES:
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
            response.accepted = False
            response.message = '지원되지 않는 팔 구분입니다.'
            return response

        self.get_logger().info(
            f'상품 픽업 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, '
            f'product_id={request.product_id}, arm_side={request.arm_side}'
        )

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

        response.accepted = True
        response.message = '상품 픽업 명령이 처리되었습니다.'
        return response

    def handle_place_product(self, request, response):
        # Packee Main의 상품 담기 요청 처리
        if request.arm_side not in VALID_ARM_SIDES:
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
            response.accepted = False
            response.message = '지원되지 않는 팔 구분입니다.'
            return response

        self.get_logger().info(
            f'상품 담기 요청 수신: robot_id={request.robot_id}, order_id={request.order_id}, '
            f'product_id={request.product_id}, arm_side={request.arm_side}'
        )

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

        response.accepted = True
        response.message = '상품 담기 명령이 처리되었습니다.'
        return response

    def publish_pose_status(self, robot_id, order_id, pose_type, status, progress, message):
        # 자세 상태 토픽 발행 유틸리티
        status_msg = ArmPoseStatus()
        status_msg.robot_id = robot_id
        status_msg.order_id = order_id
        status_msg.pose_type = pose_type
        status_msg.status = status
        status_msg.progress = progress
        status_msg.message = message
        self.pose_status_pub.publish(status_msg)

    def publish_pick_status(self, robot_id, order_id, product_id, arm_side, status, current_phase, progress, message):
        # 픽업 상태 토픽 발행 유틸리티
        status_msg = PackeeArmTaskStatus()
        status_msg.robot_id = robot_id
        status_msg.order_id = order_id
        status_msg.product_id = product_id
        status_msg.arm_side = arm_side
        status_msg.status = status
        status_msg.current_phase = current_phase
        status_msg.progress = progress
        status_msg.message = message
        self.pick_status_pub.publish(status_msg)

    def publish_place_status(self, robot_id, order_id, product_id, arm_side, status, current_phase, progress, message):
        # 담기 상태 토픽 발행 유틸리티
        status_msg = PackeeArmTaskStatus()
        status_msg.robot_id = robot_id
        status_msg.order_id = order_id
        status_msg.product_id = product_id
        status_msg.arm_side = arm_side
        status_msg.status = status
        status_msg.current_phase = current_phase
        status_msg.progress = progress
        status_msg.message = message
        self.place_status_pub.publish(status_msg)


def main():
    rclpy.init()
    node = PackeeArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Packee Arm Controller 노드가 종료됩니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
