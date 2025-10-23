#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from pymycobot.mycobot280 import MyCobot280
import time

from shopee_interfaces.srv import ArmMoveToPose, ArmPickProduct, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus

# HARDCODED POSES from user's "와사비 좌표 기록.txt"
STANDBY_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, -57.0]
LYING_DOWN_POSE = [0.43, 101.25, 31.9, -150.73, -89.03, -29.61]
PRE_PICK_1_POSE = [87.8, 101.25, -83.23, -19.59, 12.56, 62.92]
PRE_PICK_2_POSE = [120.14, -12.56, -67.23, -12.48, -2.37, 67.41]
PICK_WASABI_POSE = [116.27, -24.43, -84.9, 18.63, 0.79, 64.59]
BASKET_ABOVE_POSE = [80.77, 53.34, -32.87, 72.5, -163.47, 25.92]
BASKET_PLACE_POSE = [79.71, 112.32, -85.6, 67.5, -163.12, 27.77]

class PickeeArmController(Node):
    """Controls the Pickee Arm based on hardcoded sequences, with improved safety and synchronization."""

    def __init__(self):
        super().__init__('pickee_arm_controller')
        self.mc = None
        try:
            self.mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
            self.get_logger().info("myCobot arm connected successfully.")
            if self.mc.is_power_on() != 1:
                self.mc.power_on()
                self.get_logger().info("Powering on the robot.")
            self.mc.focus_all_servos()
            self.get_logger().info("All servos focused.")
            self._control_gripper(100)
            self._move_to_target_joints(STANDBY_POSE)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to myCobot arm: {e}")
        
        self.move_to_pose_srv = self.create_service(
            ArmMoveToPose, '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        self.pick_product_srv = self.create_service(
            ArmPickProduct, '/pickee/arm/pick_product', self.pick_product_callback)
        self.place_product_srv = self.create_service(
            ArmPlaceProduct, '/pickee/arm/place_product', self.place_product_callback)
        self.pose_status_pub = self.create_publisher(ArmPoseStatus, '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus, '/pickee/arm/place_status', 10)

        self.get_logger().info('Pickee Arm Hardcoded Controller has been started.')

    def _move_to_target_joints(self, joint_angles, speed=40, timeout=5):
        """Sends joint angles and waits for the movement to complete."""
        if self.mc and self.mc.is_power_on() == 1:
            self.get_logger().info(f"Executing sync move to: {joint_angles}")
            self.mc.sync_send_angles(joint_angles, speed, timeout)
            self.get_logger().info("Move complete.")
        else:
            self.get_logger().error("myCobot arm is not connected or powered on. Cannot send command.")

    def _control_gripper(self, value, speed=40):
        """Controls the gripper."""
        if self.mc and self.mc.is_power_on() == 1:
            action = "OPENING" if value == 100 else "CLOSING"
            self.get_logger().info(f"GRIPPER {action} (value: {value})")
            self.mc.set_gripper_value(value, speed)
            time.sleep(1.0)
            self.get_logger().info("Gripper action complete.")
        else:
            self.get_logger().error("myCobot arm is not connected or powered on. Cannot control gripper.")
            
    def destroy_node(self):
        """Node shutdown callback. Releases the robot arm safely."""
        if self.mc:
            self.get_logger().info("Moving to lying down pose before releasing servos.")
            self._move_to_target_joints(LYING_DOWN_POSE)
            self.get_logger().info("Releasing all servos.")
            self.mc.release_all_servos()
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
        self.get_logger().info(f'Move to pose request received: {request.pose_type}')
        target_pose_joints = STANDBY_POSE
        if request.pose_type == "standby":
            target_pose_joints = STANDBY_POSE
        elif request.pose_type == "lying_down":
            target_pose_joints = LYING_DOWN_POSE
        self._move_to_target_joints(target_pose_joints)
        status_msg = ArmPoseStatus(status="completed", progress=1.0, message=f"Reached {request.pose_type} pose")
        self.pose_status_pub.publish(status_msg)
        response.success = True
        return response

    def pick_product_callback(self, request, response):
        self.get_logger().info(f'Pick product request for "{request.target_product.product_id}" received.')
        self.get_logger().info("--- Starting Pick Sequence ---")
        self._move_to_target_joints(STANDBY_POSE)
        self._control_gripper(100)
        self._move_to_target_joints(PRE_PICK_1_POSE)
        self._move_to_target_joints(PRE_PICK_2_POSE)
        self._move_to_target_joints(PICK_WASABI_POSE)
        self._control_gripper(0)
        self._move_to_target_joints(PRE_PICK_2_POSE)
        self._move_to_target_joints(PRE_PICK_1_POSE)
        self._move_to_target_joints(STANDBY_POSE)
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
        self._move_to_target_joints(STANDBY_POSE)
        self._move_to_target_joints(BASKET_ABOVE_POSE)
        self._move_to_target_joints(BASKET_PLACE_POSE)
        self._control_gripper(100)
        self._move_to_target_joints(BASKET_ABOVE_POSE)
        self._move_to_target_joints(STANDBY_POSE)
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