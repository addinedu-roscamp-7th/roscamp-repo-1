#!/usr/bin/env python3
"""
Mock Robot Node 실행 스크립트
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    print('Error: rclpy is not available. Make sure ROS2 is sourced.')
    sys.exit(1)

try:
    from shopee_interfaces.msg import (
        PackeeAvailability,
        PackeePackingComplete,
        PackeeRobotStatus,
        PickeeArrival,
        PickeeCartHandover,
        PickeeMoveStatus,
        PickeeProductDetection,
        PickeeProductSelection,
        PickeeDetectedProduct,
        PickeeRobotStatus,
    )
    from shopee_interfaces.srv import (
        PackeePackingCheckAvailability,
        PackeePackingStart,
        PickeeProductDetect,
        PickeeProductProcessSelection,
        PickeeWorkflowEndShopping,
        PickeeWorkflowMoveToSection,
        PickeeWorkflowMoveToPackaging,
        PickeeWorkflowReturnToBase,
        PickeeWorkflowReturnToStaff,
        PickeeWorkflowStartTask,
        PickeeMainVideoStreamStart,
        PickeeMainVideoStreamStop,
        MainGetLocationPose,
        MainGetWarehousePose,
        MainGetSectionPose,
    )
except ImportError as exc:
    print(f'Error: shopee_interfaces not found: {exc}')
    print('Make sure shopee_interfaces is built and sourced.')
    sys.exit(1)

from .constants import RobotStatus

logger = logging.getLogger('mock_robot_node')


class MockRobotNode(Node):
    """Pickee/Packee 시뮬레이션 노드."""

    def __init__(self, *, enable_pickee: bool = True, enable_packee: bool = True) -> None:
        if not enable_pickee and not enable_packee:
            raise ValueError('At least one of Pickee or Packee simulation must be enabled.')

        super().__init__('mock_robot_node')
        self._enable_pickee = enable_pickee
        self._enable_packee = enable_packee

        self.current_order_id: Optional[int] = None
        self.current_robot_id: int = 1
        self.packee_available = True
        self._pickee_state = RobotStatus.IDLE.value
        self._pickee_battery = 100.0
        self._pickee_position = (0.0, 0.0, 0.0)
        self._packee_state = RobotStatus.IDLE.value
        self._packee_order_id: Optional[int] = None
        self._packee_items_in_cart = 0
        self._packee_robot_id = 10
        self._staff_station_location_id = 5000
        self._staff_station_warehouse_id = 1

        # Main Service pose 조회용 클라이언트
        self._main_location_pose_client = None
        self._main_section_pose_client = None
        self._main_warehouse_pose_client = None
        if self._enable_pickee or self._enable_packee:
            self._main_location_pose_client = self.create_client(
                MainGetLocationPose, '/main/get_location_pose'
            )
            self._main_section_pose_client = self.create_client(
                MainGetSectionPose, '/main/get_section_pose'
            )
            self._main_warehouse_pose_client = self.create_client(
                MainGetWarehousePose, '/main/get_warehouse_pose'
            )

        # Pickee 퍼블리셔/서비스 초기화
        self.pickee_move_pub = None
        self.pickee_arrival_pub = None
        self.pickee_detection_pub = None
        self.pickee_selection_pub = None
        self.pickee_handover_pub = None
        self.pickee_status_pub = None

        if self._enable_pickee:
            self.create_service(
                PickeeWorkflowStartTask,
                '/pickee/workflow/start_task',
                self.handle_start_task,
            )
            self.create_service(
                PickeeWorkflowMoveToSection,
                '/pickee/workflow/move_to_section',
                self.handle_move_to_section,
            )
            self.create_service(
                PickeeWorkflowMoveToPackaging,
                '/pickee/workflow/move_to_packaging',
                self.handle_move_to_packaging,
            )
            self.create_service(
                PickeeWorkflowReturnToBase,
                '/pickee/workflow/return_to_base',
                self.handle_return_to_base,
            )
            self.create_service(
                PickeeWorkflowReturnToStaff,
                '/pickee/workflow/return_to_staff',
                self.handle_return_to_staff,
            )
            self.create_service(
                PickeeProductDetect,
                '/pickee/product/detect',
                self.handle_product_detect,
            )
            self.create_service(
                PickeeProductProcessSelection,
                '/pickee/product/process_selection',
                self.handle_process_selection,
            )
            self.create_service(
                PickeeWorkflowEndShopping,
                '/pickee/workflow/end_shopping',
                self.handle_end_shopping,
            )
            self.create_service(
                PickeeMainVideoStreamStart,
                '/pickee/video_stream/start',
                self.handle_video_start,
            )
            self.create_service(
                PickeeMainVideoStreamStop,
                '/pickee/video_stream/stop',
                self.handle_video_stop,
            )

            self.pickee_move_pub = self.create_publisher(
                PickeeMoveStatus, '/pickee/moving_status', 10
            )
            self.pickee_arrival_pub = self.create_publisher(
                PickeeArrival, '/pickee/arrival_notice', 10
            )
            self.pickee_detection_pub = self.create_publisher(
                PickeeProductDetection, '/pickee/product_detected', 10
            )
            self.pickee_selection_pub = self.create_publisher(
                PickeeProductSelection, '/pickee/product/selection_result', 10
            )
            self.pickee_handover_pub = self.create_publisher(
                PickeeCartHandover, '/pickee/cart_handover_complete', 10
            )
            self.pickee_status_pub = self.create_publisher(
                PickeeRobotStatus, '/pickee/robot_status', 10
            )

        # Packee 퍼블리셔/서비스 초기화
        self.packee_complete_pub = None
        self.packee_status_pub = None
        self.packee_availability_pub = None

        if self._enable_packee:
            self.create_service(
                PackeePackingCheckAvailability,
                '/packee/packing/check_availability',
                self.handle_check_availability,
            )
            self.create_service(
                PackeePackingStart,
                '/packee/packing/start',
                self.handle_packing_start,
            )
            self.packee_complete_pub = self.create_publisher(
                PackeePackingComplete, '/packee/packing_complete', 10
            )
            self.packee_status_pub = self.create_publisher(
                PackeeRobotStatus, '/packee/robot_status', 10
            )
            self.packee_availability_pub = self.create_publisher(
                PackeeAvailability, '/packee/availability_result', 10
            )

        # 주기적으로 로봇 상태를 퍼블리시하여 RobotStateStore를 채운다.
        self._status_timer = self.create_timer(
            1.0, self._publish_robot_status, clock=self.get_clock()
        )

        enabled_roles = []
        if self._enable_pickee:
            enabled_roles.append('Pickee Robot (피킹 로봇)')
        if self._enable_packee:
            enabled_roles.append('Packee Robot (포장 로봇)')
        enabled_desc = ', '.join(enabled_roles)
        self.get_logger().info(f'Mock Robot Node initialized (enabled: {enabled_desc})')

    # === Pickee 관련 핸들러 ===
    def handle_start_task(self, request, response):  # noqa: D401
        """피킹 작업 시작"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.current_order_id = request.order_id
        self.current_robot_id = request.robot_id
        self._pickee_state = RobotStatus.WORKING.value
        self._publish_robot_status()

        self.get_logger().info(
            f'[MOCK] Start task: Order={request.order_id}, Robot={request.robot_id}'
        )

        # 시퀀스 다이어그램에 맞춰 위치 좌표를 조회한다.
        if self._enable_pickee:
            for product in request.product_list:
                if hasattr(product, 'location_id'):
                    pose_req = MainGetLocationPose.Request()
                    pose_req.location_id = int(product.location_id)
                    self._call_main_service_async(
                        self._main_location_pose_client,
                        pose_req,
                        'get_location_pose',
                    )
                if hasattr(product, 'section_id'):
                    section_req = MainGetSectionPose.Request()
                    section_req.section_id = int(product.section_id)
                    self._call_main_service_async(
                        self._main_section_pose_client,
                        section_req,
                        'get_section_pose',
                    )

        response.success = True
        response.message = 'Mock task started'
        return response

    def handle_move_to_section(self, request, response):
        """섹션 이동 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.get_logger().info(
            f'[MOCK] Moving to section: Location={request.location_id}, Section={request.section_id}'
        )
        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        if self.pickee_move_pub:
            move_msg = PickeeMoveStatus()
            move_msg.robot_id = request.robot_id
            move_msg.order_id = request.order_id
            move_msg.location_id = request.location_id
            self.pickee_move_pub.publish(move_msg)

        if hasattr(request, 'location_id') and request.location_id:
            pose_req = MainGetLocationPose.Request()
            pose_req.location_id = int(request.location_id)
            self._call_main_service_async(
                self._main_location_pose_client,
                pose_req,
                'get_location_pose',
            )

        if hasattr(request, 'location_id'):
            pose_req = MainGetLocationPose.Request()
            pose_req.location_id = int(request.location_id)
            self._call_main_service_async(
                self._main_location_pose_client,
                pose_req,
                'get_location_pose',
            )
        if hasattr(request, 'section_id'):
            section_req = MainGetSectionPose.Request()
            section_req.section_id = int(request.section_id)
            self._call_main_service_async(
                self._main_section_pose_client,
                section_req,
                'get_section_pose',
            )

        def publish_arrival():
            if not self.pickee_arrival_pub:
                timer.cancel()
                return
            arrival_msg = PickeeArrival()
            arrival_msg.robot_id = request.robot_id
            arrival_msg.order_id = request.order_id
            arrival_msg.location_id = request.location_id
            arrival_msg.section_id = request.section_id
            self.pickee_arrival_pub.publish(arrival_msg)

            self.get_logger().info(f'[MOCK] Arrived at section {request.section_id}')
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            timer.cancel()

        timer = self.create_timer(0.5, publish_arrival, clock=self.get_clock())

        response.success = True
        response.message = 'Moving to section'
        return response

    def handle_move_to_packaging(self, request, response):
        """포장대 이동 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.get_logger().info('[MOCK] Moving to packaging area')
        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        if hasattr(request, 'location_id') and request.location_id:
            self._request_location_pose(int(request.location_id))

        if self.pickee_move_pub:
            move_msg = PickeeMoveStatus()
            move_msg.robot_id = request.robot_id
            move_msg.order_id = request.order_id
            move_msg.location_id = request.location_id
            self.pickee_move_pub.publish(move_msg)

        handover_timer = None
        arrival_timer = None

        def publish_handover():
            nonlocal handover_timer
            if self.pickee_handover_pub:
                handover_msg = PickeeCartHandover()
                handover_msg.robot_id = request.robot_id
                handover_msg.order_id = request.order_id
                self.pickee_handover_pub.publish(handover_msg)
                self.get_logger().info('[MOCK] Cart handover complete')
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            if handover_timer is not None:
                handover_timer.cancel()

        def publish_arrival():
            nonlocal arrival_timer, handover_timer
            if self.pickee_arrival_pub:
                arrival_msg = PickeeArrival()
                arrival_msg.robot_id = request.robot_id
                arrival_msg.order_id = request.order_id
                arrival_msg.location_id = request.location_id
                arrival_msg.section_id = -1
                self.pickee_arrival_pub.publish(arrival_msg)
                self.get_logger().info('[MOCK] Arrived at packaging area')
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            if arrival_timer is not None:
                arrival_timer.cancel()
            handover_timer = self.create_timer(0.5, publish_handover, clock=self.get_clock())

        arrival_timer = self.create_timer(0.5, publish_arrival, clock=self.get_clock())

        response.success = True
        response.message = 'Moving to packaging'
        return response

    def handle_return_to_base(self, request, response):
        """복귀 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        base_location_id = int(getattr(request, 'location_id', 0))
        order_id = self.current_order_id or 0

        self.get_logger().info(f'[MOCK] Returning to base (location={base_location_id})')
        if base_location_id > 0:
            self._request_location_pose(base_location_id)

        if self.pickee_move_pub:
            move_msg = PickeeMoveStatus()
            move_msg.robot_id = request.robot_id
            move_msg.order_id = order_id
            move_msg.location_id = base_location_id
            self.pickee_move_pub.publish(move_msg)

        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        arrival_timer = None
        charging_timer = None

        def finish_charging():
            nonlocal charging_timer
            self._pickee_state = RobotStatus.IDLE.value
            self._publish_robot_status()
            if charging_timer is not None:
                charging_timer.cancel()

        def publish_arrival():
            nonlocal arrival_timer, charging_timer
            if self.pickee_arrival_pub:
                arrival_msg = PickeeArrival()
                arrival_msg.robot_id = request.robot_id
                arrival_msg.order_id = order_id
                arrival_msg.location_id = base_location_id
                arrival_msg.section_id = -1
                self.pickee_arrival_pub.publish(arrival_msg)
                self.get_logger().info('[MOCK] Arrived at base')
            self._pickee_state = RobotStatus.CHARGING.value
            self._publish_robot_status()
            if arrival_timer is not None:
                arrival_timer.cancel()
            charging_timer = self.create_timer(1.0, finish_charging, clock=self.get_clock())

        arrival_timer = self.create_timer(0.5, publish_arrival, clock=self.get_clock())

        self.current_order_id = None

        response.success = True
        response.message = 'Returning to base'
        return response

    def handle_return_to_staff(self, request, response):
        """직원 위치 복귀 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        target_location = self._staff_station_location_id
        warehouse_id = self._staff_station_warehouse_id
        order_id = self.current_order_id or 0

        self.get_logger().info('[MOCK] Returning to staff station')
        self._request_warehouse_pose(warehouse_id)

        if self.pickee_move_pub:
            move_msg = PickeeMoveStatus()
            move_msg.robot_id = request.robot_id
            move_msg.order_id = order_id
            move_msg.location_id = target_location
            self.pickee_move_pub.publish(move_msg)

        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        arrival_timer = None

        def publish_arrival():
            nonlocal arrival_timer
            if self.pickee_arrival_pub:
                arrival_msg = PickeeArrival()
                arrival_msg.robot_id = request.robot_id
                arrival_msg.order_id = order_id
                arrival_msg.location_id = target_location
                arrival_msg.section_id = -1
                self.pickee_arrival_pub.publish(arrival_msg)
                self.get_logger().info('[MOCK] Arrived at staff station')
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            if arrival_timer is not None:
                arrival_timer.cancel()

        arrival_timer = self.create_timer(0.5, publish_arrival, clock=self.get_clock())

        response.success = True
        response.message = 'Returning to staff'
        return response

    def handle_product_detect(self, request, response):
        """상품 인식 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.get_logger().info(f'[MOCK] Detecting products: {list(request.product_ids)}')

        detection_msg = PickeeProductDetection()
        detection_msg.robot_id = request.robot_id
        detection_msg.order_id = request.order_id

        bbox_number = 1
        for product_id in request.product_ids:
            detected = PickeeDetectedProduct()
            detected.product_id = product_id
            detected.bbox_number = bbox_number
            detection_msg.products.append(detected)
            bbox_number += 1

        if self.pickee_detection_pub:
            self.pickee_detection_pub.publish(detection_msg)
        self._publish_robot_status()

        response.success = True
        response.message = 'Detection started'
        return response

    def handle_process_selection(self, request, response):
        """상품 선택 처리 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.get_logger().info(
            f'[MOCK] Processing selection: Product={request.product_id}, BBox={request.bbox_number}'
        )

        if self.pickee_selection_pub:
            selection_msg = PickeeProductSelection()
            selection_msg.robot_id = request.robot_id
            selection_msg.order_id = request.order_id
            selection_msg.product_id = request.product_id
            selection_msg.quantity = 1
            selection_msg.success = True
            selection_msg.message = '상품 담기 성공'
            self.pickee_selection_pub.publish(selection_msg)

        response.success = True
        response.message = 'Selection processing'
        return response

    def handle_end_shopping(self, request, response):
        """쇼핑 종료 시뮬레이션"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response

        self.get_logger().info(f'[MOCK] Ending shopping: Order={request.order_id}')
        self.current_order_id = None
        self._pickee_state = RobotStatus.IDLE.value
        self._publish_robot_status()
        response.success = True
        response.message = 'Shopping ended'
        return response

    def handle_video_start(self, request, response):
        """영상 스트림 시작"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response
        self.get_logger().info(f'[MOCK] Video stream started for robot {request.robot_id}')
        response.success = True
        response.message = 'Video stream started'
        return response

    def handle_video_stop(self, request, response):
        """영상 스트림 중지"""
        if not self._enable_pickee:
            response.success = False
            response.message = 'Pickee simulation disabled'
            return response
        self.get_logger().info(f'[MOCK] Video stream stopped for robot {request.robot_id}')
        response.success = True
        response.message = 'Video stream stopped'
        return response

    # === Packee 관련 핸들러 ===
    def handle_check_availability(self, request, response):
        """Packee 가용성 확인"""
        if not self._enable_packee:
            response.success = False
            response.message = 'Packee simulation disabled'
            return response

        self.get_logger().info(
            f'[MOCK] Checking Packee availability for order {request.order_id} (robot {request.robot_id})'
        )
        response.success = self.packee_available
        response.message = 'Packee available' if self.packee_available else 'All busy'
        if self.packee_availability_pub:
            availability_msg = PackeeAvailability()
            availability_msg.robot_id = request.robot_id
            availability_msg.order_id = request.order_id
            availability_msg.available = bool(self.packee_available)
            availability_msg.cart_detected = bool(self.packee_available)
            availability_msg.message = (
                '장바구니 확인 완료' if self.packee_available else '장바구니 감지 실패'
            )
            self.packee_availability_pub.publish(availability_msg)
        return response

    def handle_packing_start(self, request, response):
        """포장 시작 시뮬레이션"""
        if not self._enable_packee:
            response.success = False
            response.message = 'Packee simulation disabled'
            return response

        self.get_logger().info(
            f'[MOCK] Packing started: Order={request.order_id}, Robot={request.robot_id}'
        )
        self.packee_available = False
        self._packee_state = RobotStatus.WORKING.value
        self._packee_order_id = request.order_id
        self._publish_robot_status()

        def publish_packing_complete():
            if not self.packee_complete_pub:
                timer.cancel()
                return
            complete_msg = PackeePackingComplete()
            complete_msg.robot_id = request.robot_id
            complete_msg.order_id = request.order_id
            complete_msg.success = True
            complete_msg.message = '포장이 완료되었습니다'

            self.packee_complete_pub.publish(complete_msg)
            self.get_logger().info(f'[MOCK] Packing complete for order {request.order_id}')
            self.packee_available = True
            self._packee_state = RobotStatus.IDLE.value
            self._packee_order_id = None
            self._packee_items_in_cart = 0
            self._publish_robot_status()
            timer.cancel()

        timer = self.create_timer(1.0, publish_packing_complete, clock=self.get_clock())

        response.success = True
        response.message = 'Packing started'
        return response

    # === 공통 유틸 ===
    def _request_location_pose(self, location_id: int) -> None:
        """Main Service의 위치 좌표 조회 서비스 호출"""
        if location_id <= 0:
            return
        pose_req = MainGetLocationPose.Request()
        pose_req.location_id = int(location_id)
        self._call_main_service_async(
            self._main_location_pose_client,
            pose_req,
            'get_location_pose',
        )

    def _request_warehouse_pose(self, warehouse_id: int) -> None:
        """Main Service의 창고 좌표 조회 서비스 호출"""
        if warehouse_id <= 0:
            return
        pose_req = MainGetWarehousePose.Request()
        pose_req.warehouse_id = int(warehouse_id)
        self._call_main_service_async(
            self._main_warehouse_pose_client,
            pose_req,
            'get_warehouse_pose',
        )

    def _publish_robot_status(self) -> None:
        """Mock 로봇의 현재 상태를 퍼블리시"""
        if self._enable_pickee and self.pickee_status_pub:
            pickee_msg = PickeeRobotStatus()
            pickee_msg.robot_id = self.current_robot_id
            pickee_msg.state = self._pickee_state
            pickee_msg.battery_level = float(self._pickee_battery)
            pickee_msg.current_order_id = self.current_order_id or 0
            pickee_msg.position_x = float(self._pickee_position[0])
            pickee_msg.position_y = float(self._pickee_position[1])
            pickee_msg.orientation_z = float(self._pickee_position[2])
            self.pickee_status_pub.publish(pickee_msg)

        if self._enable_packee and self.packee_status_pub:
            packee_msg = PackeeRobotStatus()
            packee_msg.robot_id = self._packee_robot_id
            packee_msg.state = self._packee_state
            packee_msg.current_order_id = self._packee_order_id or 0
            packee_msg.items_in_cart = self._packee_items_in_cart
            self.packee_status_pub.publish(packee_msg)

    def _call_main_service_async(self, client, request, description: str) -> None:
        """Main Service 제공 서비스에 비동기 요청을 발행한다."""
        if client is None:
            return
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().debug("Main service %s unavailable", description)
            return
        try:
            client.call_async(request)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug("Failed to call %s: %s", description, exc)


def run_mock_robot(default_mode: str = 'all', argv: Optional[list[str]] = None) -> None:
    """Mock Robot 노드를 실행한다."""
    parser = argparse.ArgumentParser(description='Mock Robot Node')
    parser.add_argument(
        '--mode',
        choices=['all', 'pickee', 'packee'],
        default=default_mode,
        help='시뮬레이션할 로봇 유형 선택',
    )
    args, ros_args = parser.parse_known_args(argv)

    mode = args.mode
    enable_pickee = mode in ('all', 'pickee')
    enable_packee = mode in ('all', 'packee')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    banner = [
        '╔══════════════════════════════════════════════════════════╗',
        '║          Mock Robot Node Starting                        ║',
        '╚══════════════════════════════════════════════════════════╝',
        '',
        'Simulating:',
    ]
    if enable_pickee:
        banner.append('  - Pickee Robot (피킹 로봇)')
    if enable_packee:
        banner.append('  - Packee Robot (포장 로봇)')
    banner.append('')
    banner.append('Services & Topics initialized')
    banner.append('Press Ctrl+C to stop')
    banner_text = '\n'.join(banner)
    print(f'\n{banner_text}\n')

    rclpy.init(args=ros_args)
    node = MockRobotNode(enable_pickee=enable_pickee, enable_packee=enable_packee)

    try:
        logger.info('Mock Robot Node running...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info('Mock Robot Node stopped by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    """콘솔 스크립트 진입점."""
    run_mock_robot()


if __name__ == '__main__':
    main()
