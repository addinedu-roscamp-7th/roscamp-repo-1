#!/usr/bin/env python3
"""
Mock Robot Node 실행 스크립트
"""
import sys
import logging
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    print("Error: rclpy is not available. Make sure ROS2 is sourced.")
    sys.exit(1)

try:
    from shopee_interfaces.msg import (
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
        PickeeWorkflowStartTask,
        PickeeMainVideoStreamStart,
        PickeeMainVideoStreamStop,
    )
except ImportError as e:
    print(f"Error: shopee_interfaces not found: {e}")
    print("Make sure shopee_interfaces is built and sourced.")
    sys.exit(1)

from .constants import RobotStatus
logger = logging.getLogger("mock_robot_node")


class MockRobotNode(Node):
    """Mock 로봇 노드 - Pickee와 Packee 시뮬레이션"""

    def __init__(self):
        super().__init__("mock_robot_node")

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

        # === Pickee Services ===
        self.create_service(
            PickeeWorkflowStartTask,
            "/pickee/workflow/start_task",
            self.handle_start_task
        )
        self.create_service(
            PickeeWorkflowMoveToSection,
            "/pickee/workflow/move_to_section",
            self.handle_move_to_section
        )
        self.create_service(
            PickeeWorkflowMoveToPackaging,
            "/pickee/workflow/move_to_packaging",
            self.handle_move_to_packaging
        )
        self.create_service(
            PickeeWorkflowReturnToBase,
            "/pickee/workflow/return_to_base",
            self.handle_return_to_base
        )
        self.create_service(
            PickeeProductDetect,
            "/pickee/product/detect",
            self.handle_product_detect
        )
        self.create_service(
            PickeeProductProcessSelection,
            "/pickee/product/process_selection",
            self.handle_process_selection
        )
        self.create_service(
            PickeeWorkflowEndShopping,
            "/pickee/workflow/end_shopping",
            self.handle_end_shopping
        )
        self.create_service(
            PickeeMainVideoStreamStart,
            "/pickee/video_stream/start",
            self.handle_video_start
        )
        self.create_service(
            PickeeMainVideoStreamStop,
            "/pickee/video_stream/stop",
            self.handle_video_stop
        )

        # === Packee Services ===
        self.create_service(
            PackeePackingCheckAvailability,
            "/packee/packing/check_availability",
            self.handle_check_availability
        )
        self.create_service(
            PackeePackingStart,
            "/packee/packing/start",
            self.handle_packing_start
        )

        # === Publishers ===
        self.pickee_move_pub = self.create_publisher(
            PickeeMoveStatus, "/pickee/moving_status", 10
        )
        self.pickee_arrival_pub = self.create_publisher(
            PickeeArrival, "/pickee/arrival_notice", 10
        )
        self.pickee_detection_pub = self.create_publisher(
            PickeeProductDetection, "/pickee/product_detected", 10
        )
        self.pickee_selection_pub = self.create_publisher(
            PickeeProductSelection, "/pickee/product/selection_result", 10
        )
        self.pickee_handover_pub = self.create_publisher(
            PickeeCartHandover, "/pickee/cart_handover_complete", 10
        )
        self.packee_complete_pub = self.create_publisher(
            PackeePackingComplete, "/packee/packing_complete", 10
        )
        self.pickee_status_pub = self.create_publisher(
            PickeeRobotStatus, "/pickee/robot_status", 10
        )
        self.packee_status_pub = self.create_publisher(
            PackeeRobotStatus, "/packee/robot_status", 10
        )

        # 주기적으로 로봇 상태를 퍼블리시하여 RobotStateStore를 채웁니다.
        self._status_timer = self.create_timer(1.0, self._publish_robot_status, clock=self.get_clock())

        self.get_logger().info("Mock Robot Node initialized")

    def handle_start_task(self, request, response):
        """피킹 작업 시작"""
        self.current_order_id = request.order_id
        self.current_robot_id = request.robot_id
        self._pickee_state = RobotStatus.WORKING.value
        self._publish_robot_status()

        self.get_logger().info(
            f"[MOCK] Start task: Order={request.order_id}, Robot={request.robot_id}"
        )

        response.success = True
        response.message = "Mock task started"
        return response

    def handle_move_to_section(self, request, response):
        """섹션 이동 시뮬레이션"""
        self.get_logger().info(
            f"[MOCK] Moving to section: Location={request.location_id}, Section={request.section_id}"
        )
        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        # 이동 시작 알림
        move_msg = PickeeMoveStatus()
        move_msg.robot_id = request.robot_id
        move_msg.order_id = request.order_id
        move_msg.location_id = request.location_id
        self.pickee_move_pub.publish(move_msg)

        # 0.5초 후 도착 알림
        def publish_arrival():
            arrival_msg = PickeeArrival()
            arrival_msg.robot_id = request.robot_id
            arrival_msg.order_id = request.order_id
            arrival_msg.location_id = request.location_id
            arrival_msg.section_id = request.section_id
            self.pickee_arrival_pub.publish(arrival_msg)
            self.get_logger().info(f"[MOCK] Arrived at section {request.section_id}")
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            timer.cancel()  # 타이머 취소 (한 번만 실행)

        timer = self.create_timer(0.5, publish_arrival, clock=self.get_clock())

        response.success = True
        response.message = "Moving to section"
        return response

    def handle_product_detect(self, request, response):
        """상품 인식 시뮬레이션"""
        self.get_logger().info(
            f"[MOCK] Detecting products: {request.product_ids}"
        )
        self._pickee_state = RobotStatus.WORKING.value
        self._publish_robot_status()

        # 0.3초 후 상품 인식 완료 알림
        def publish_detection():
            detection_msg = PickeeProductDetection()
            detection_msg.robot_id = request.robot_id
            detection_msg.order_id = request.order_id

            for idx, product_id in enumerate(request.product_ids):
                detected = PickeeDetectedProduct()
                detected.product_id = product_id
                detected.bbox_number = idx + 1
                detection_msg.products.append(detected)

            self.pickee_detection_pub.publish(detection_msg)
            self.get_logger().info(f"[MOCK] Detected {len(request.product_ids)} products")
            timer.cancel()  # 타이머 취소 (한 번만 실행)

        timer = self.create_timer(0.3, publish_detection, clock=self.get_clock())

        response.success = True
        response.message = "Detection started"
        return response

    def handle_process_selection(self, request, response):
        """상품 선택 처리 시뮬레이션"""
        self.get_logger().info(
            f"[MOCK] Processing selection: Product={request.product_id}, BBox={request.bbox_number}"
        )
        self._pickee_state = RobotStatus.WORKING.value
        self._publish_robot_status()

        # 0.3초 후 선택 결과 발행
        def publish_selection():
            selection_msg = PickeeProductSelection()
            selection_msg.robot_id = request.robot_id
            selection_msg.order_id = request.order_id
            selection_msg.product_id = request.product_id
            selection_msg.quantity = 1
            selection_msg.success = True
            selection_msg.message = "상품이 장바구니에 담겼습니다"

            self.pickee_selection_pub.publish(selection_msg)
            self.get_logger().info(f"[MOCK] Product {request.product_id} selected")
            timer.cancel()  # 타이머 취소 (한 번만 실행)

        timer = self.create_timer(0.3, publish_selection, clock=self.get_clock())

        response.success = True
        response.message = "Selection processing"
        return response

    def handle_end_shopping(self, request, response):
        """쇼핑 종료 시뮬레이션"""
        self.get_logger().info(f"[MOCK] Ending shopping: Order={request.order_id}")
        self.current_order_id = None
        self._pickee_state = RobotStatus.IDLE.value
        self._publish_robot_status()
        response.success = True
        response.message = "Shopping ended"
        return response

    def handle_move_to_packaging(self, request, response):
        """포장대 이동 시뮬레이션"""
        self.get_logger().info(f"[MOCK] Moving to packaging area")
        self._pickee_state = RobotStatus.MOVING.value
        self._publish_robot_status()

        # 이동 알림
        move_msg = PickeeMoveStatus()
        move_msg.robot_id = request.robot_id
        move_msg.order_id = request.order_id
        move_msg.location_id = 999  # 포장대 위치 ID (임의 값)
        self.pickee_move_pub.publish(move_msg)

        # 0.5초 후 장바구니 전달 완료
        def publish_handover():
            handover_msg = PickeeCartHandover()
            handover_msg.robot_id = request.robot_id
            handover_msg.order_id = request.order_id
            handover_msg.success = True
            handover_msg.message = "Cart handed over"

            self.pickee_handover_pub.publish(handover_msg)
            self.get_logger().info(f"[MOCK] Cart handover complete")
            self._pickee_state = RobotStatus.WORKING.value
            self._publish_robot_status()
            timer.cancel()  # 타이머 취소 (한 번만 실행)

        timer = self.create_timer(0.5, publish_handover, clock=self.get_clock())

        response.success = True
        response.message = "Moving to packaging"
        return response

    def handle_return_to_base(self, request, response):
        """복귀 시뮬레이션"""
        self.get_logger().info(f"[MOCK] Returning to base")
        self.current_order_id = None
        self._pickee_state = RobotStatus.IDLE.value
        self._publish_robot_status()
        response.success = True
        response.message = "Returning to base"
        return response

    def handle_video_start(self, request, response):
        """영상 스트림 시작"""
        self.get_logger().info(f"[MOCK] Video stream started for robot {request.robot_id}")
        response.success = True
        response.message = "Video stream started"
        return response

    def handle_video_stop(self, request, response):
        """영상 스트림 중지"""
        self.get_logger().info(f"[MOCK] Video stream stopped for robot {request.robot_id}")
        response.success = True
        response.message = "Video stream stopped"
        return response

    def handle_check_availability(self, request, response):
        """Packee 가용성 확인"""
        self.get_logger().info(
            f"[MOCK] Checking Packee availability for order {request.order_id} (robot {request.robot_id})"
        )
        response.success = self.packee_available
        response.message = "Packee available" if self.packee_available else "All busy"
        return response

    def handle_packing_start(self, request, response):
        """포장 시작 시뮬레이션"""
        self.get_logger().info(f"[MOCK] Packing started: Order={request.order_id}, Robot={request.robot_id}")
        self.packee_available = False
        self._packee_state = RobotStatus.WORKING.value
        self._packee_order_id = request.order_id
        self._publish_robot_status()

        # 1초 후 포장 완료
        def publish_packing_complete():
            complete_msg = PackeePackingComplete()
            complete_msg.robot_id = request.robot_id
            complete_msg.order_id = request.order_id
            complete_msg.success = True
            complete_msg.message = "포장이 완료되었습니다"

            self.packee_complete_pub.publish(complete_msg)
            self.get_logger().info(f"[MOCK] Packing complete for order {request.order_id}")
            self.packee_available = True
            self._packee_state = RobotStatus.IDLE.value
            self._packee_order_id = None
            self._packee_items_in_cart = 0
            self._publish_robot_status()
            timer.cancel()  # 타이머 취소 (한 번만 실행)

        timer = self.create_timer(1.0, publish_packing_complete, clock=self.get_clock())

        response.success = True
        response.message = "Packing started"
        return response

    def _publish_robot_status(self):
        """Mock 로봇의 현재 상태를 주기적으로 퍼블리시합니다."""
        pickee_msg = PickeeRobotStatus()
        pickee_msg.robot_id = self.current_robot_id
        pickee_msg.state = self._pickee_state
        pickee_msg.battery_level = float(self._pickee_battery)
        pickee_msg.current_order_id = self.current_order_id or 0
        pickee_msg.position_x = float(self._pickee_position[0])
        pickee_msg.position_y = float(self._pickee_position[1])
        pickee_msg.orientation_z = float(self._pickee_position[2])
        self.pickee_status_pub.publish(pickee_msg)

        packee_msg = PackeeRobotStatus()
        packee_msg.robot_id = self._packee_robot_id
        packee_msg.state = self._packee_state
        packee_msg.current_order_id = self._packee_order_id or 0
        packee_msg.items_in_cart = self._packee_items_in_cart
        self.packee_status_pub.publish(packee_msg)


def main(args=None):
    """Mock Robot Node 메인 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("""
╔══════════════════════════════════════════════════════════╗
║          Mock Robot Node Starting                        ║
╚══════════════════════════════════════════════════════════╝

Simulating:
  - Pickee Robot (피킹 로봇)
  - Packee Robot (포장 로봇)

Services & Topics initialized
Press Ctrl+C to stop
""")

    rclpy.init(args=args)
    node = MockRobotNode()

    try:
        logger.info("Mock Robot Node running...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Mock Robot Node stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
