#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus

# Import modularized components
from . import arm_poses
from .arm_control import ArmControl

class PickeeArmNode(Node):
    """
    Coordinates ROS services with arm movements.
    Handles high-level logic and leaves hardware control to ArmControl.
    """

    def __init__(self):
        super().__init__('pickee_arm_node')
        self.arm = None
        
        try:
            self.arm = ArmControl(self.get_logger())
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
        
        self.move_to_pose_srv = self.create_service(
            ArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        self.pick_product_srv = self.create_service(
            ArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)
        self.place_product_srv = self.create_service(
            ArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)
            
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Node has been started.')

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

    def _publish_task_status(self, publisher, request, product_id, status, current_phase, progress, message=""):
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
            self.get_logger().info("Moving to lying down pose and holding position.")
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
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

    def pick_product_callback(self, request, response):
        self.get_logger().info(f'Pick product request for "{request.target_product.product_id}" received.')
        pub = self.pick_status_pub
        product_id = request.target_product.product_id
        self._publish_task_status(pub, request, product_id, "in_progress", "starting", 0.05, "Starting pick sequence")

        try:
            self.arm.control_gripper(100)
            self._publish_task_status(pub, request, product_id, "in_progress", "approaching", 0.2, "Approaching item")
            self.arm.move_to_joints(arm_poses.PRE_PICK_1_POSE)
            self.arm.move_to_joints(arm_poses.PRE_PICK_2_POSE)
            self._publish_task_status(pub, request, product_id, "in_progress", "grasping", 0.6, "Grasping item")
            self.arm.move_to_joints(arm_poses.PICK_ITEM_POSE)
            self.arm.control_gripper(0)
            self._publish_task_status(pub, request, product_id, "in_progress", "lifting", 0.8, "Lifting item")
            self.arm.move_to_joints(arm_poses.PRE_PICK_2_POSE)
            self.arm.move_to_joints(arm_poses.PRE_PICK_1_POSE)
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
            
            response.success = True
            response.message = "Product picked successfully"
            self._publish_task_status(pub, request, product_id, "completed", "done", 1.0, response.message)
        except Exception as e:
            response.success = False
            response.message = f"Pick sequence failed: {e}"
            self.get_logger().error(response.message)
            self._publish_task_status(pub, request, product_id, "failed", "error", 0.5, response.message)

        return response

    def place_product_callback(self, request, response):
        self.get_logger().info(f'Place product request for "{request.product_id}" received.')
        pub = self.place_status_pub
        product_id = request.product_id
        self._publish_task_status(pub, request, product_id, "in_progress", "starting", 0.05, "Starting place sequence")

        try:
            self._publish_task_status(pub, request, product_id, "in_progress", "moving", 0.3, "Moving to basket")
            self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE)
            self.arm.move_to_joints(arm_poses.BASKET_PLACE_POSE)
            self._publish_task_status(pub, request, product_id, "in_progress", "releasing", 0.8, "Releasing item")
            self.arm.control_gripper(100)
            self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE)
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)

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