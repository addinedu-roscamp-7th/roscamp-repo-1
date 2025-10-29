#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time

from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus, Pose6D

from . import arm_poses
from .arm_control import ArmControl

class PickeeArmNode(Node):
    def __init__(self):
        super().__init__('pickee_arm_node')
        self.arm = None

        # 실시간 시각 서보 제어(Visual Servoing) 활성화 여부를 나타내는 상태 플래그
        self.visual_servoing_active = False

        try:
            self.arm = ArmControl(self.get_logger())
            # 초기화: 그리퍼를 열고 대기 자세로 이동
            self.arm.control_gripper(100)
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
        except Exception as e:
            self.get_logger().fatal(f"ArmControl initialization failed: {e}. Shutting down node.")
            return

        self.pose_map = {
            "standby": arm_poses.STANDBY_POSE,
            "lying_down": arm_poses.LYING_DOWN_POSE,
            "shelf_view": arm_poses.CHECK_SHELF_POSE,
        }

        # 서비스 서버 생성 (외부에서 로봇 팔에 명령을 내릴 수 있는 창구)
        self.move_to_pose_srv = self.create_service(ArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        self.pick_product_srv = self.create_service(ArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)
        self.place_product_srv = self.create_service(ArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)

        # pickee_vision으로부터 실시간 제어 명령을 받기 위한 Subscriber (토픽 리스너)
        self.servo_subscriber = self.create_subscription(
            Pose6D,
            '/pickee/arm/move_servo',
            self.servo_callback,
            10)

        # 상태 보고용 퍼블리셔 생성
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Node has been started.')

    def _publish_pose_status(self, request, status, progress, message=""):
        # ... (생략)
        pass # 상태 보고 함수

    def _publish_task_status(self, publisher, request, product_id, status, current_phase, progress, message=""):
        # ... (생략)
        pass # 작업 상태 보고 함수

    def destroy_node(self):
        if self.arm and self.arm.is_connected():
            self.get_logger().info("Moving to lying down pose and holding position.")
            # 노드 종료 시 로봇 팔을 안전한 자세로 이동
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
        # ... (생략)
        pass # 일반적인 관절 이동 처리 함수

    def servo_callback(self, msg):
        # 시각 서보 제어 모드가 아닐 때는 명령을 무시
        if not self.visual_servoing_active:
            return

        # 현재 각도를 읽어옴
        current_angles = self.arm.get_angles()
        if current_angles is None:
            return

        # 현재 각도에 수신된 변화량(move_servo 토픽 값)을 더해 새로운 목표 각도 계산
        # msg.x, msg.y, ... 는 로봇 팔의 6개 관절에 대한 목표 변화량을 의미함
        new_angles = [
            current_angles[0] + msg.x,
            current_angles[1] + msg.y,
            current_angles[2] + msg.z,
            current_angles[3] + msg.rx,
            current_angles[4] + msg.ry,
            current_angles[5] + msg.rz,
        ]

        # 새로운 목표 각도로 로봇 팔을 부드럽게 움직임 (속도 60)
        self.arm.move_to_joints(new_angles, speed=60)

    def pick_product_callback(self, request, response):
        self.get_logger().info(f'Pick product request received. Activating Visual Servoing.')

        # 시각 서보 제어 모드를 활성화
        self.visual_servoing_active = True

        try:
            # 픽업 준비 자세로 이동 (시각 서보잉 시작 지점)
            self.arm.move_to_joints(arm_poses.PRE_PICK_1_POSE)
            self.get_logger().info("Reached pre-pick pose. Waiting for /pickee/arm/move_servo commands...")
            response.success = True
            response.message = "Visual servoing for pick-up activated."
        except Exception as e:
            self.get_logger().error(f"Failed to move to pre-pick pose: {e}")
            response.success = False
            response.message = str(e)

        return response

    def place_product_callback(self, request, response):
        # place 동작이 시작되면 시각 서보 모드를 비활성화 (시각 서보잉 종료)
        self.visual_servoing_active = False
        self.get_logger().info("Visual servoing DEACTIVATED.")

        # 기존의 하드코딩된 place 동작 수행 (바구니에 내려놓는 일련의 동작)
        self.get_logger().info(f'Place product request for "{request.product_id}" received.')
        pub = self.place_status_pub
        product_id = request.product_id
        self._publish_task_status(pub, request, product_id, "in_progress", "starting", 0.05, "Starting place sequence")
        try:
            self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE) # 바구니 위로 이동
            self.arm.move_to_joints(arm_poses.BASKET_PLACE_POSE) # 바구니 안으로 이동
            self.arm.control_gripper(100) # 물건 내려놓기 (그리퍼 열기)
            self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE) # 바구니 위로 다시 올라오기
            self.arm.move_to_joints(arm_poses.STANDBY_POSE) # 대기 자세로 복귀

            response.success = True
            response.message = "Product placed successfully"
            self._publish_task_status(pub, request, product_id, "completed", "done", 1.0, response.message)
        except Exception as e:
            response.success = False
            response.message = f"Place sequence failed: {e}"
            self.get_logger().error(response.message)
            self._publish_task_status(pub, request, product_id, "failed", "error", 0.5, response.message)

        return response

def main(args=None):
    rclpy.init(args=args)
    node = PickeeArmNode()
    if node.arm:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()