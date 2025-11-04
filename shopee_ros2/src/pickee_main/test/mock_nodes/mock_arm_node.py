#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import ArmTaskStatus, ArmPoseStatus
from shopee_interfaces.srv import (
    ArmMoveToPose, 
    ArmPickProduct, 
    ArmPlaceProduct
)
import threading
import time

class MockArmNode(Node):
    """Pickee Arm 컴포넌트의 Mock 노드"""
    
    def __init__(self):
        super().__init__('mock_arm_node')
        
        # Service Server 생성
        self.move_service = self.create_service(
            ArmMoveToPose,
            '/pickee/arm/move_to_pose',
            self.move_to_pose_callback
        )
        
        self.pick_service = self.create_service(
            ArmPickProduct,
            '/pickee/arm/pick_product',
            self.pick_product_callback
        )
        
        self.place_service = self.create_service(
            ArmPlaceProduct,
            '/pickee/arm/place_product',
            self.place_product_callback
        )
        
        # Publisher 생성
        self.pose_status_pub = self.create_publisher(
            ArmPoseStatus,
            '/pickee/arm/pose_status',
            10
        )
        
        self.pick_status_pub = self.create_publisher(
            ArmTaskStatus,
            '/pickee/arm/pick_status',
            10
        )
        
        self.place_status_pub = self.create_publisher(
            ArmTaskStatus,
            '/pickee/arm/place_status',
            10
        )
        
        # 현재 상태
        self.current_pose = 'home'  # 'home', 'pick', 'place', 'transport'
        self.is_busy = False
        
        self.get_logger().info('Mock Arm Node started successfully')
    
    def move_to_pose_callback(self, request, response):
        """자세 이동 서비스 요청 처리"""
        self.get_logger().info(f'Received move to pose request: {request.pose_type}')
        
        if self.is_busy:
            response.success = False
            response.message = 'Arm is currently busy'
            return response
        
        self.is_busy = True
        
        # 비동기적으로 자세 변경 시뮬레이션
        threading.Thread(
            target=self.simulate_pose_change,
            args=(request.pose_type,)
        ).start()
        
        response.success = True
        response.message = f'Moving to {request.pose_type} pose'
        return response
    
    def pick_product_callback(self, request, response):
        """제품 픽업 서비스 요청 처리"""
        self.get_logger().info(f'Received pick request: product_id={request.product_id}')
        
        # if self.is_busy:
        #     response.accepted = False
        #     response.message = 'Arm is currently busy'
        #     return response
        
        # if self.current_pose != 'pick':
        #     response.accepted = False
        #     response.message = 'Arm is not in pick pose'
        #     return response
        
        # self.is_busy = True
        
        # # 비동기적으로 픽업 시뮬레이션
        # threading.Thread(
        #     target=self.simulate_pick_operation,
        #     args=(request.product_id)
        # ).start()
        
        response.success = True
        response.message = 'Pick operation started'
        return response
    
    def place_product_callback(self, request, response):
        """제품 놓기 서비스 요청 처리"""
        self.get_logger().info(f'Received place request: product_id={request.product_id}')
        
        if self.is_busy:
            response.accepted = False
            response.message = 'Arm is currently busy'
            return response
        
        self.is_busy = True
        
        # 비동기적으로 놓기 시뮬레이션
        threading.Thread(
            target=self.simulate_place_operation,
            args=(request.product_id,)
        ).start()
        
        response.accepted = True
        response.message = 'Place operation started'
        return response
    
    def simulate_pose_change(self, pose_type):
        """자세 변경 시뮬레이션"""
        # 진행 중 상태 발행
        self.publish_pose_status(pose_type, 'in_progress', 0.0)
        
        # 중간 진행 상황들 발행
        for progress in [0.3, 0.6, 0.9]:
            time.sleep(0.5)
            self.publish_pose_status(pose_type, 'in_progress', progress)
        
        time.sleep(0.5)  # 마지막 0.5초
        
        # 완료 상태 발행
        self.publish_pose_status(pose_type, 'completed', 1.0)
        
        self.current_pose = pose_type
        self.is_busy = False
        self.get_logger().info(f'Moved to {pose_type} pose')
    
    def simulate_pick_operation(self, product_id):
        """픽업 동작 시뮬레이션"""
        # 진행 상황 발행
        self.publish_pick_status(product_id, 'in_progress')
        
        time.sleep(3.0)  # 3초 픽업 시뮬레이션
        
        # 완료 상태 발행
        self.publish_pick_status(product_id, 'completed')
        
        self.current_pose = 'transport'
        self.is_busy = False
        self.get_logger().info(f'Pick operation completed: product_id={product_id}')
    
    def simulate_place_operation(self, product_id):
        """놓기 동작 시뮬레이션"""
        # 진행 상황 발행
        self.publish_place_status(product_id, 'in_progress')
        
        time.sleep(2.0)  # 2초 놓기 시뮬레이션
        
        # 완료 상태 발행
        self.publish_place_status(product_id, 'completed')
        
        self.current_pose = 'home'
        self.is_busy = False
        self.get_logger().info(f'Place operation completed: {product_id}')
    
    def publish_pick_status(self, product_id, status):
        """픽업 상태 발행"""
        msg = ArmTaskStatus()
        msg.robot_id = 1
        msg.order_id = 1  # TODO: 실제 order_id 사용
        msg.product_id = product_id
        msg.status = status
        msg.current_phase = 'pick_operation'
        msg.progress = 1.0 if status == 'completed' else 0.5
        msg.message = f'Pick {status}'
        msg.arm_side = 'left'  

        self.pick_status_pub.publish(msg)
        self.get_logger().info(f'Published pick status: {product_id} - {status}')
    
    def publish_place_status(self, product_id, status):
        """놓기 상태 발행"""
        msg = ArmTaskStatus()
        msg.robot_id = 1
        msg.order_id = 1  # TODO: 실제 order_id 사용
        msg.product_id = product_id
        msg.status = status
        msg.current_phase = 'place_operation'
        msg.progress = 1.0 if status == 'completed' else 0.5
        msg.message = f'Place {status}'
        
        self.place_status_pub.publish(msg)
        self.get_logger().info(f'Published place status: {product_id} - {status}')
    
    def publish_pose_status(self, pose_type, status, progress):
        """자세 변경 상태 발행"""
        msg = ArmPoseStatus()
        msg.robot_id = 1
        msg.order_id = 1  # TODO: 실제 order_id 사용
        msg.pose_type = pose_type
        msg.status = status
        msg.progress = progress
        
        # 상태별 메시지 설정
        if status == 'in_progress':
            msg.message = f"Moving to {pose_type} pose"
        elif status == 'completed':
            msg.message = f"Reached {pose_type} pose"
        elif status == 'failed':
            msg.message = f"Failed to reach {pose_type} pose"
        
        self.pose_status_pub.publish(msg)
        self.get_logger().info(f'Published pose status: {pose_type} - {status} ({progress:.1f})')


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    node = MockArmNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Mock Arm Node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()