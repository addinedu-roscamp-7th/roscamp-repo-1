#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import Bool

from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus, Pose6D

from . import arm_poses
from .arm_control import ArmControl

class PickeeArmNode(Node):
    def __init__(self):
        super().__init__('test_1110_new_arm_node')
        self.arm = None
        self.basket_item_count = 0

        # 실시간 시각 서보 제어(Visual Servoing) 활성화 여부를 나타내는 상태 플래그
        self.visual_servoing_active = True # for test

        try:
            self.arm = ArmControl(self.get_logger())
            self.arm.control_gripper(100)
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
            time.sleep(2)
            self.arm.move_to_coords(arm_poses.TOP_SECOND_GRID)
        except Exception as e:
            self.get_logger().fatal(f"ArmControl initialization failed: {e}. Shutting down node.")
            return

        self.pose_map = {
            "standby": arm_poses.STANDBY_POSE,
            "lying_down": arm_poses.LYING_DOWN_POSE,
            "shelf_view": arm_poses.CHECK_SHELF_POSE,
        }

        # 서비스 서버 생성
        self.move_to_pose_srv = self.create_service(ArmMoveToPose,
        '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        self.pick_product_srv = self.create_service(ArmPickProduct,
        '/pickee/arm/pick_product', self.pick_product_callback)
        self.place_product_srv = self.create_service(ArmPlaceProduct,
        '/pickee/arm/place_product', self.place_product_callback)

        # pickee_vision으로부터 실시간 제어 명령을 받기 위한 Subscriber
        self.servo_subscriber = self.create_subscription(
            Pose6D,
            '/pickee/arm/move_servo',
            self.servo_callback,
            10)

        # 상태 보고용 퍼블리셔 생성
        self.pose_status_pub = self.create_publisher(ArmPoseStatus,
        '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus,
        '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus,
        '/pickee/arm/place_status', 10)

        # pickee_arm <-> pickee_vision test 
        self.test_pub = self.create_publisher(Bool,'/testtest/move_lock', 10)

        # =================================================================
        # [추가된 부분 1] 실시간 좌표 발행을 위한 퍼블리셔 및 타이머
        # =================================================================
        self.real_pose_publisher = self.create_publisher(Pose6D, '/pickee/arm/real_pose', 10)
        self.pose_publish_timer = self.create_timer(0.03, self.publish_real_pose_callback) # 10Hz로 발행
        # =================================================================

        self.bool_msg = Bool()
        self.bool_msg.data = False
        self.test_pub.publish(self.bool_msg)

        self.get_logger().info('Pickee Arm Node has been started.')

    # =================================================================
    # [추가된 부분 2] 실시간 좌표를 주기적으로 발행하는 콜백 함수
    # =================================================================
    def publish_real_pose_callback(self):
        if not self.arm or not self.arm.is_connected():
            return

        # ArmControl을 통해 현재 로봇 좌표를 얻어옴
        current_coords = self.arm.get_coords()

        # 좌표가 유효한지 확인 후 발행
        if current_coords and isinstance(current_coords, list) and len(current_coords) == 6:
            pose_msg = Pose6D()
            pose_msg.x, pose_msg.y, pose_msg.z, pose_msg.rx, pose_msg.ry, pose_msg.rz = [float(c) for c in current_coords]
            self.real_pose_publisher.publish(pose_msg)
        else:
            self.get_logger().debug(f"Could not get valid coordinates to publish. Value: {current_coords}")
    # =================================================================
        
    def _publish_pose_status(self, request, status, progress, message=""):
        msg = ArmPoseStatus()
        msg.robot_id = request.robot_id
        msg.order_id = request.order_id
        msg.pose_type = request.pose_type
        msg.status = status
        msg.progress = progress
        msg.message = message
        self.pose_status_pub.publish(msg)
        self.get_logger().info(f"Published Pose Status: {status} ({progress*100:.0f}%) for pose {request.pose_type}")

    def _publish_task_status(self, publisher, request, product_id, status,
        current_phase, progress, message=""):
        msg = ArmTaskStatus()
        msg.robot_id = request.robot_id
        msg.order_id = request.order_id
        msg.product_id = product_id
        msg.arm_side = ""
        msg.status = status
        msg.current_phase = current_phase
        msg.progress = progress
        msg.message = message
        publisher.publish(msg)
        self.get_logger().info(f"Published Task Status: {status} ({progress*100:.0f}%) - Phase: {current_phase}")

    def destroy_node(self):
        if self.arm and self.arm.is_connected():
            self.get_logger().info("distroy: Moving to lying down pose and holding position.")
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
        self.bool_msg.data = False
        self.test_pub.publish(self.bool_msg)
        
        self.get_logger().info(f'Move to pose request received: {request.pose_type}')
        self._publish_pose_status(request, "in_progress", 0.1, f"Starting to move to {request.pose_type}")

        target_pose = self.pose_map.get(request.pose_type)

        if not target_pose:
            response.success = False
            response.message = f"Unknown pose_type: {request.pose_type}"
            self.get_logger().error(response.message)
            self._publish_pose_status(request, "failed", 0.0, response.message)
            return response

        try:
            self.arm.move_to_joints(target_pose)
            response.success = True
            response.message = f"Successfully moved to {request.pose_type}"
            self._publish_pose_status(request, "completed", 1.0, response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to move to {request.pose_type}: {e}"
            self.get_logger().error(response.message)
            self._publish_pose_status(request, "failed", 0.5, response.message)

        return response

    def servo_callback(self, msg):
        if not self.visual_servoing_active:
            return

        new_angles = [
            msg.x,
            msg.y,
            msg.z,
            msg.rx,
            msg.ry,
            msg.rz,
        ]
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.arm.move_to_coords(new_angles, speed=60)

    def pick_product_callback(self, request, response):
        self.destroy_subscription(self.servo_subscriber)
        self.servo_subscriber = self.create_subscription(
            Pose6D,
            '/pickee/arm/move_servo',
            self.servo_callback,
            10)

        self.bool_msg.data = True
        self.test_pub.publish(self.bool_msg)

        self.get_logger().info(f'Pick product request received. Activating Visual Servoing.')
        self.visual_servoing_active = True

        try:
            temp_list = self.arm.get_coords()
            temp_list[2] -= 60
            self.arm.move_to_coords(temp_list)
            self.get_logger().info("Reached pre-pick pose. Waiting for /pickee/arm/move_servo commands...")
            response.success = True
            response.message = "Visual servoing for pick-up activated."
            time.sleep(0.5)
            self.arm.control_gripper(0)
        except Exception as e:
            self.get_logger().error(f"Failed to move to pre-pick pose: {e}")
            response.success = False
            response.message = str(e)

        return response

    def place_product_callback(self, request, response):
        self.visual_servoing_active = True
        self.get_logger().info("Visual servoing DEACTIVATED.")

        self.get_logger().info(f'Place product request for "{request.product_id}" received.')
        pub = self.place_status_pub
        product_id = request.product_id
        self._publish_task_status(pub, request, product_id, "in_progress",
        "starting", 0.05, "Starting place sequence")

        self.basket_item_count += 1

        if self.basket_item_count > 3:
            self.basket_item_count = 1

        basket_above_pose = getattr(arm_poses, f"BASKET_ABOVE_POSE_{self.basket_item_count}")
        basket_place_pose = getattr(arm_poses, f"BASKET_PLACE_POSE_{self.basket_item_count}")

        try:
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
            time.sleep(2)
            self.get_logger().info("STANDBY_POSE")
            self.arm.move_to_joints(basket_above_pose)
            self.get_logger().info("basket_above_pose")
            time.sleep(2)
            self.arm.move_to_joints(basket_place_pose)
            self.get_logger().info("basket_place_pose")
            time.sleep(2)
            self.arm.control_gripper(100) 
            self.get_logger().info("control_gripper")
            time.sleep(2)
            self.arm.move_to_joints(basket_above_pose)
            self.get_logger().info("basket_above_pose2")
            time.sleep(2)
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
            self.get_logger().info("LYING_DOWN_POSE")
            time.sleep(2)

            response.success = True
            response.message = "Product placed successfully"
            self._publish_task_status(pub, request, product_id, "completed", "done"
            , 1.0, response.message)
        except Exception as e:
            response.success = False
            response.message = f"Place sequence failed: {e}"
            self.get_logger().error(response.message)
            self._publish_task_status(pub, request, product_id, "failed", "error",
            0.5, response.message)

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