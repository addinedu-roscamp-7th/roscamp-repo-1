#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import Bool

from shopee_interfaces.srv import ArmMoveToPose, ArmCheckBbox, ArmPlaceProduct
from shopee_interfaces.msg import ArmPoseStatus, ArmTaskStatus, Pose6D
from std_srvs.srv import Trigger

from . import arm_poses
from .arm_control import ArmControl

class PickeeArmNode(Node):
    def __init__(self):
        super().__init__('pickee_arm_node_for_basket')
        self.arm = None
        self.basket_item_count = 0
        self.bbox = 0

        # 실시간 시각 서보 제어(Visual Servoing) 활성화 여부를 나타내는 상태 플래그
        self.visual_servoing_active = True # for test
        self.last_servo_time = self.get_clock().now() ##
        try:
            self.arm = ArmControl(self.get_logger())
            self.arm.control_gripper(100)
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
            time.sleep(2)
            #self.arm.move_to_joints(arm_poses.CHECK_SHELF_POSE) #for_test
        except Exception as e:
            self.get_logger().fatal(f"ArmControl initialization failed: {e}. \
Shutting down node.")
            return

        self.pose_map = {
            "standby": arm_poses.STANDBY_POSE,
            "lying_down": arm_poses.LYING_DOWN_POSE,
            "shelf_view": arm_poses.CHECK_SHELF_POSE,
        }

        # 서비스 서버 생성
        self.move_to_pose_srv = self.create_service(ArmMoveToPose,
        '/pickee/arm/move_to_pose', self.move_to_pose_callback)
        # (main -> arm) Bbox_number전달 
        self.check_product_srv = self.create_service(ArmCheckBbox,
        '/pickee/arm/check_product', self.check_product_callback)
        self.grep_product_srv = self.create_service(Trigger,
        # (vision -> arm) z축내려서 그리퍼닫으라는 명령    
        '/pickee/arm/grep_product', self.grep_product_callback)
        self.place_product_srv = self.create_service(ArmPlaceProduct,
        '/pickee/arm/place_product', self.place_product_callback)
        # (vision -> arm) 픽업 시작을 위한 매대 확인 
        self.move_start_srv = self.create_service(Trigger,
        '/pickee/arm/move_start', self.move_start_callback)

        # pickee_vision으로부터 실시간 제어 명령을 받기 위한 Subscriber
        self.servo_subscriber = self.create_subscription(
            Pose6D,
            '/pickee/arm/move_servo',
            self.servo_callback,
            1)

        # 상태 보고용 퍼블리셔 생성
        self.pose_status_pub = self.create_publisher(ArmPoseStatus,
        '/pickee/arm/pose_status', 10)
        self.pick_status_pub = self.create_publisher(ArmTaskStatus,
        '/pickee/arm/pick_status', 10)
        self.place_status_pub = self.create_publisher(ArmTaskStatus,
        '/pickee/arm/place_status', 10)
        self.real_pose_publisher = self.create_publisher(Pose6D, '/pickee/arm/real_pose', 10)
        self.pose_publish_timer = self.create_timer(0.03, self.publish_real_pose_callback)

        # pickee_arm <-> pickee_vision test 
        
        self.is_moving_pub = self.create_publisher(Bool,'/pickee/arm/is_moving', 10)

        self.bool_msg = Bool()
        self.bool_msg.data = False
        self.is_moving_pub.publish(self.bool_msg)  


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
            self.get_logger().info("distroy: Moving to lying down pose and holding \
position.")
            self.arm.move_to_joints(arm_poses.LYING_DOWN_POSE)
        super().destroy_node()

    def move_to_pose_callback(self, request, response):
        
        self.get_logger().info(f'Move to pose request received: {request.pose_type} \
')
        self._publish_pose_status(request, "in_progress", 0.1, f"Starting to move \
to {request.pose_type}")

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
        current_time = self.get_clock().now()
        if (current_time - self.last_servo_time).nanoseconds / 1e9 < 0.1:
            return
        self.last_servo_time = current_time
        self.arm.move_to_coords(new_angles, speed=60)
        print(self.arm.get_coords())
    
    def move_start_callback(self, request, response):
        self.get_logger().info('Move start request received from vision.')
        try:
            self.arm.move_to_joints(arm_poses.STANDBY_POSE)
            time.sleep(2)
            self.arm.move_to_joints(arm_poses.CHECK_SHELF_POSE)
            time.sleep(1)
            while self.arm.is_moving():
                time.sleep(0.1)
            response.success = True
            response.message = "Successfully moved to check shelf pose."
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to move to check shelf pose: {e}"
            self.get_logger().error(response.message)
        return response
    
    def check_product_callback(self, request, response):
        self.bbox_number = request.bbox_number
        target_pose = None

        self.bool_msg.data = False
        self.is_moving_pub.publish(self.bool_msg)

        if self.bbox_number == 1:
            target_pose = arm_poses.TOP_VIEW_POSE_GRID1
        elif self.bbox_number == 2:
            target_pose = arm_poses.TOP_VIEW_POSE_GRID2
        elif self.bbox_number == 3:
            target_pose = arm_poses.TOP_VIEW_POSE_GRID3        
        else:
            response.success = False
            response.message = f"No pose defined for bbox_number: {self.bbox_number}"
            self.get_logger().error(response.message)
            return response

        try:
            self.arm.move_to_coords(target_pose)
            time.sleep(1)
            while self.arm.is_moving():
                time.sleep(0.1)
            
            self.get_logger().info(f"Movement to grid for bbox {self.bbox_number}complete.")
            
            self.bool_msg.data = True
            self.is_moving_pub.publish(self.bool_msg)
            self.get_logger().info("Published move_lock: True to vision.")

            response.success = True
            response.message = "Successfully moved to grid and notified vision."
        except Exception as e:
            response.success = False
            response.message = f"Fkiled during check_product process: {e}"
            self.get_logger().error(response.message)
        
        return response

    def grep_product_callback(self, request, response):

        self.destroy_subscription(self.servo_subscriber)
        self.servo_subscriber = self.create_subscription(
            Pose6D,
            '/pickee/arm/move_servo',
            self.servo_callback,
            10)


        self.get_logger().info(f'Pick product request received. Activating Visual \
Servoing.')

        self.visual_servoing_active = True

        try:
            temp_list = self.arm.get_coords()
            temp_list[2] = 150
            self.arm.move_to_coords(temp_list)
            self.get_logger().info("Reached pre-pick pose. Waiting for \
/pickee/arm/move_servo commands...")
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

        self.get_logger().info(f'Place product request for "{request.product_id}" \
received.')
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