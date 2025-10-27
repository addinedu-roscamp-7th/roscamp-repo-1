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
        self.arm = None # self.arm을 먼저 None으로 초기화
        
        try:
            self.arm = ArmControl(self.get_logger())
            # Initial startup sequence
            self.arm.control_gripper(100) # Start with gripper open
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
        except Exception as e:
            self.get_logger().fatal(f"ArmControl initialization failed: {e}. Shutting down node.")
            return

        # Mapping from pose name string to pose data
        self.pose_map = {
            "standby": arm_poses.STANDBY_POSE,
            "lying_down": arm_poses.LYING_DOWN_POSE,
            "shelf_view": arm_poses.CHECK_SHELF_POSE,
        }
        
        # Create ROS2 services and publishers
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

    def destroy_node(self):
        """Node shutdown callback. Releases the robot arm safely."""
        # self.arm이 성공적으로 생성되었는지 확인
        if self.arm and self.arm.is_connected():
            self.get_logger().info("Moving to lying down pose before releasing servos.")
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
            self.arm.release_servos()
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
        self.get_logger().info(f'Move to pose request received: {request.pose_type}')
        
        target_pose = self.pose_map.get(request.pose_type)
        
        if target_pose:
            self.arm.move_to_joints(target_pose)
            status_msg = ArmPoseStatus(status="completed", progress=1.0, message=f"Reached {request.pose_type} pose")
            response.success = True
        else:
            self.get_logger().error(f"Unknown pose_type: {request.pose_type}")
            status_msg = ArmPoseStatus(status="failed", progress=0.0, message=f"Unknown pose type: {request.pose_type}")
            response.success = False
            
        self.pose_status_pub.publish(status_msg)
        return response

    def pick_product_callback(self, request, response):
        self.get_logger().info(f'Pick product request for "{request.target_product.product_id}" received.')
        self.get_logger().info("--- Starting Pick Sequence ---")
        
        self.arm.control_gripper(100) # Open
        self.arm.move_to_joints(arm_poses.PRE_PICK_1_POSE)
        self.arm.move_to_joints(arm_poses.PRE_PICK_2_POSE)
        self.arm.move_to_joints(arm_poses.PICK_ITEM_POSE)
        self.arm.control_gripper(0) # Close
        self.arm.move_to_joints(arm_poses.PRE_PICK_2_POSE)
        self.arm.move_to_joints(arm_poses.PRE_PICK_1_POSE)
        self.arm.move_to_joints(arm_poses.STANDBY_POSE)
        
        self.get_logger().info("--- Pick Sequence Finished ---")
        status_msg = ArmTaskStatus(status="completed", current_phase="done", progress=1.0, message="Product picked successfully")
        status_msg.product_id = request.target_product.product_id
        status_msg.arm_side = ""
        self.pick_status_pub.publish(status_msg)
        response.success = True
        return response

    def place_product_callback(self, request, response):
        self.get_logger().info(f'Place product request for "{request.product_id}" received.')
        self.get_logger().info("--- Starting Place Sequence ---")
        
        self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE)
        self.arm.move_to_joints(arm_poses.BASKET_PLACE_POSE)
        self.arm.control_gripper(100) # Open
        self.arm.move_to_joints(arm_poses.BASKET_ABOVE_POSE)
        self.arm.move_to_joints(arm_poses.STANDBY_POSE)
        
        self.get_logger().info("--- Place Sequence Finished ---")
        status_msg = ArmTaskStatus(status="completed", current_phase="done", progress=1.0, message="Product placed successfully")
        status_msg.product_id = request.product_id
        status_msg.arm_side = ""
        self.place_status_pub.publish(status_msg)
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PickeeArmNode()
    # node.arm이 성공적으로 생성되었는지 확인
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
