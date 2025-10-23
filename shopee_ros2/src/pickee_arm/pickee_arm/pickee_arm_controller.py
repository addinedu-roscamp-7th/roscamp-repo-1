#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pickee_arm_controller.py

Simple, hardcoded implementation of the pickee arm controller.
It receives target joint angles from pickee_main and executes the motion.
"""

import rclpy
from rclpy.node import Node
import time

from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus, Pose6D

# ====================================================================
# HARDCODED POSES (as joint angles list [j1, j2, j3, j4, j5, j6])
# TODO: These values must be found and calibrated using a real robot.
# ====================================================================
STANDBY_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SHELF_VIEW_POSE = [0.0, -45.0, 45.0, 0.0, 0.0, 0.0]
CART_VIEW_POSE = [0.0, 45.0, -45.0, 0.0, 0.0, 0.0]

# This is a sample sequence for picking a product from a specific location
PICK_SEQUENCE = {
    "pre_pick": [10.0, -30.0, -50.0, 0.0, 0.0, 0.0], # Move above the target
    "pick":     [15.0, -35.0, -55.0, 0.0, 0.0, 0.0], # Move down to the target
    "post_pick":[10.0, -30.0, -50.0, 0.0, 0.0, 0.0]  # Lift it up
}

class PickeeArmController(Node):
    """A simple, hardcoded controller for the Pickee Arm."""

    def __init__(self):
        super().__init__('pickee_arm_controller')

        # Create Service Servers based on the latest interface spec
        self.move_to_pose_srv = self.create_service(
            ArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        self.pick_product_srv = self.create_service(
            ArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)
        self.place_product_srv = self.create_service(
            ArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)

        # Create Publishers
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Hardcoded Controller has been started.')

    def _move_to_target_joints(self, joint_angles):
        """Simulates sending joint angles to the physical robot driver."""
        self.get_logger().info(f"Executing move to joint angles: {joint_angles}")
        # TODO: Implement the actual communication with the robot hardware driver here.
        time.sleep(1.0) # Simulate movement time
        self.get_logger().info("Move complete.")

    def move_to_pose_callback(self, request, response):
        self.get_logger().info(f'Move to pose request received: {request.pose_type}')
        
        target_pose_joints = STANDBY_POSE
        if request.pose_type == "shelf_view":
            target_pose_joints = SHELF_VIEW_POSE
        elif request.pose_type == "cart_view":
            target_pose_joints = CART_VIEW_POSE
        
        self._move_to_target_joints(target_pose_joints)

        status_msg = ArmPoseStatus(status="completed", progress=1.0, message=f"Reached {request.pose_type} pose")
        self.pose_status_pub.publish(status_msg)
        
        response.success = True
        return response

    def pick_product_callback(self, request, response):
        self.get_logger().info(f'Pick product request received for product_id: {request.target_product.product_id}')

        # Execute a pre-defined sequence of motions for picking
        self.get_logger().info("--- Starting Pick Sequence ---")
        self._move_to_target_joints(PICK_SEQUENCE["pre_pick"])
        self._move_to_target_joints(PICK_SEQUENCE["pick"])
        # TODO: Add gripper control logic here (e.g., close_gripper())
        self.get_logger().info("GRIPPER CLOSED (simulation)")
        self._move_to_target_joints(PICK_SEQUENCE["post_pick"])
        self.get_logger().info("--- Pick Sequence Finished ---")

        status_msg = ArmTaskStatus(status="completed", current_phase="done", progress=1.0, message="Product picked successfully")
        status_msg.product_id = request.target_product.product_id
        status_msg.arm_side = ""
        self.pick_status_pub.publish(status_msg)

        response.success = True
        return response

    def place_product_callback(self, request, response):
        self.get_logger().info(f'Place product request received for product_id: {request.product_id}')
        
        # According to the new spec, the request itself contains the target joint angles.
        target_joints = [
            request.pose.joint_1,
            request.pose.joint_2,
            request.pose.joint_3,
            request.pose.joint_4,
            request.pose.joint_5,
            request.pose.joint_6
        ]
        self.get_logger().info("--- Starting Place Sequence ---")
        self._move_to_target_joints(target_joints)
        # TODO: Add gripper control logic here (e.g., open_gripper())
        self.get_logger().info("GRIPPER OPENED (simulation)")
        self.get_logger().info("--- Place Sequence Finished ---")

        status_msg = ArmTaskStatus(status="completed", current_phase="done", progress=1.0, message="Product placed successfully")
        status_msg.product_id = request.product_id
        status_msg.arm_side = ""
        self.place_status_pub.publish(status_msg)

        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PickeeArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
