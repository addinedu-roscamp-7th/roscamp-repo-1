"""
ROS2 로봇 코디네이터

Pickee/Packee 로봇들과의 ROS2 통신을 담당합니다.
- ROS2 서비스 호출 (로봇에게 명령)
- ROS2 토픽 구독 (로봇 상태 수신)
- 로봇 상태 캐싱 및 관리
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node

from .inventory_service import InventoryService
from .product_service import ProductService

from shopee_interfaces.msg import (
    PackeePackingComplete,
    PackeeRobotStatus,
    PickeeArrival,
    PickeeCartHandover,
    PickeeMoveStatus,
    PickeeProductDetection,
    PickeeProductSelection,
    PickeeRobotStatus,
    Pose2D,
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
    MainGetProductLocation,
    MainGetLocationPose, # Add new service import
    PickeeMainVideoStreamStart,
    PickeeMainVideoStreamStop,
)

logger = logging.getLogger(__name__)


class RobotCoordinator(Node):
    """
    ROS2 로봇 코디네이터 노드
    
    Main Service와 로봇 간의 모든 ROS2 통신을 관리합니다.
    - Pickee/Packee에게 명령 전송 (ROS2 Service)
    - 로봇 상태 및 이벤트 수신 (ROS2 Topic)
    - 상태 캐싱 및 콜백 지원
    """

    def __init__(self) -> None:
        """
        ROS2 노드 및 통신 채널 초기화
        
        생성되는 것들:
        - 5개의 토픽 구독 (Subscriber)
        - 4개의 서비스 클라이언트
        - 상태 캐시 및 콜백 핸들러
        """
        super().__init__("robot_coordinator")
        
        # 외부에서 등록할 콜백 함수들 (OrderService 등에서 사용)
        self._pickee_status_cb: Optional[Callable[[PickeeRobotStatus], None]] = None
        self._pickee_selection_cb: Optional[Callable[[PickeeProductSelection], None]] = None
        self._packee_status_cb: Optional[Callable[[PackeeRobotStatus], None]] = None
        self._packee_complete_cb: Optional[Callable[[PackeePackingComplete], None]] = None

        # === Pickee 토픽 구독 ===
        # 로봇 상태 (위치, 배터리, 현재 작업 등)
        self._pickee_status_sub = self.create_subscription(
            PickeeRobotStatus, "/pickee/robot_status", self._on_pickee_status, 10
        )
        # 이동 상태 (이동 시작, 진행 중)
        self._pickee_move_sub = self.create_subscription(
            PickeeMoveStatus, "/pickee/moving_status", self._on_pickee_move, 10
        )
        # 도착 알림
        self._pickee_arrival_sub = self.create_subscription(
            PickeeArrival, "/pickee/arrival_notice", self._on_pickee_arrival, 10
        )
        # 상품 선택 결과 (사용자가 선택한 상품)
        self._pickee_selection_sub = self.create_subscription(
            PickeeProductSelection, "/pickee/product/selection_result", self._on_pickee_selection, 10
        )
        # 장바구니 전달 완료
        self._pickee_handover_sub = self.create_subscription(
            PickeeCartHandover, "/pickee/cart_handover_complete", self._on_pickee_handover, 10
        )
        # 상품 인식 완료
        self._pickee_product_detected_sub = self.create_subscription(
            PickeeProductDetection, "/pickee/product_detected", self._on_product_detected, 10
        )

        # === ROS2 서비스 클라이언트 ===
        # Pickee: 작업 시작 명령
        self._pickee_start_cli = self.create_client(PickeeWorkflowStartTask, "/pickee/workflow/start_task")
        # Pickee: 상품 인식 명령
        # Pickee: 섹션 이동 명령
        self._pickee_move_section_cli = self.create_client(PickeeWorkflowMoveToSection, "/pickee/workflow/move_to_section")
        # Pickee: 포장대로 이동 명령
        self._pickee_move_packaging_cli = self.create_client(PickeeWorkflowMoveToPackaging, "/pickee/workflow/move_to_packaging")
        # Pickee: 복귀 명령
        self._pickee_return_base_cli = self.create_client(PickeeWorkflowReturnToBase, "/pickee/workflow/return_to_base")
        self._pickee_product_detect_cli = self.create_client(PickeeProductDetect, "/pickee/product/detect")
        # Pickee: 상품 선택 처리
        self._pickee_process_cli = self.create_client(PickeeProductProcessSelection, "/pickee/product/process_selection")
        # Pickee: 쇼핑 종료 명령
        self._pickee_end_shopping_cli = self.create_client(PickeeWorkflowEndShopping, "/pickee/workflow/end_shopping")
        # Pickee: 영상 스트림 시작/중지
        self._pickee_video_start_cli = self.create_client(PickeeMainVideoStreamStart, "/pickee/video_stream/start")
        self._pickee_video_stop_cli = self.create_client(PickeeMainVideoStreamStop, "/pickee/video_stream/stop")
        # Packee: 작업 가능 여부 확인
        self._packee_check_cli = self.create_client(PackeePackingCheckAvailability, "/packee/packing/check_availability")
        # Packee: 포장 시작 명령
        self._packee_start_cli = self.create_client(PackeePackingStart, "/packee/packing/start")

        # === ROS2 서비스 서버 ===
        # Main Service가 직접 제공하는 서비스
        self._get_product_location_srv = self.create_service(
            MainGetProductLocation, "/main/get_product_location", self._get_product_location_callback
        )
        self._get_location_pose_srv = self.create_service(
            MainGetLocationPose, "/main/get_location_pose", self._get_location_pose_callback
        )

        # 로봇 상태 캐시 (최근 메시지 저장)
        self._ros_cache: Dict[str, object] = {}

    async def dispatch_pick_task(self, request: PickeeWorkflowStartTask.Request) -> PickeeWorkflowStartTask.Response:
        """
        Pickee에게 피킹 작업 시작 명령
        
        Args:
            request: 작업 요청 (robot_id, order_id, user_id, product_list)
            
        Returns:
            Response: 성공 여부 및 메시지
            
        참고: Main_vs_Pic_Main.md - /pickee/workflow/start_task
        """
        logger.info("Dispatching pick task: %s", request)
        return await self._call_service(self._pickee_start_cli, request)

    async def dispatch_pick_process(
        self, request: PickeeProductProcessSelection.Request
    ) -> PickeeProductProcessSelection.Response:
        """
        Pickee에게 상품 선택 처리 명령
        
        사용자가 선택한 상품을 확정하고 다음 동작을 지시합니다.
        
        Args:
            request: 선택 처리 요청 (robot_id, order_id, action)
            
        Returns:
            Response: 성공 여부
        """
        logger.info("Dispatching pick process: %s", request)
        return await self._call_service(self._pickee_process_cli, request)

    async def dispatch_shopping_end(
        self, request: PickeeWorkflowEndShopping.Request
    ) -> PickeeWorkflowEndShopping.Response:
        """
        Pickee에게 쇼핑 종료 명령
        
        Args:
            request: 쇼핑 종료 요청 (robot_id, order_id)
            
        Returns:
            Response: 성공 여부
        """
        logger.info("Dispatching shopping end: %s", request)
        return await self._call_service(self._pickee_end_shopping_cli, request)

    async def dispatch_product_detect(
        self, request: PickeeProductDetect.Request
    ) -> PickeeProductDetect.Response:
        """
        Pickee에게 상품 인식 시작 명령
        
        Args:
            request: 상품 인식 요청 (robot_id, order_id, product_ids)
            
        Returns:
            Response: 성공 여부
        """
        logger.info("Dispatching product detect: %s", request)
        return await self._call_service(self._pickee_product_detect_cli, request)

    async def dispatch_move_to_section(
        self, request: PickeeWorkflowMoveToSection.Request
    ) -> PickeeWorkflowMoveToSection.Response:
        """Pickee에게 섹션 이동 명령"""
        logger.info("Dispatching move to section: %s", request)
        return await self._call_service(self._pickee_move_section_cli, request)

    async def dispatch_move_to_packaging(
        self, request: PickeeWorkflowMoveToPackaging.Request
    ) -> PickeeWorkflowMoveToPackaging.Response:
        """Pickee에게 포장대로 이동 명령"""
        logger.info("Dispatching move to packaging: %s", request)
        return await self._call_service(self._pickee_move_packaging_cli, request)

    async def dispatch_return_to_base(
        self, request: PickeeWorkflowReturnToBase.Request
    ) -> PickeeWorkflowReturnToBase.Response:
        """Pickee에게 복귀 명령"""
        logger.info("Dispatching return to base: %s", request)
        return await self._call_service(self._pickee_return_base_cli, request)

    async def dispatch_video_stream_start(
        self, request: PickeeMainVideoStreamStart.Request
    ) -> PickeeMainVideoStreamStart.Response:
        """
        Pickee에게 영상 스트림 시작 명령
        """
        logger.info("Dispatching video stream start: %s", request)
        return await self._call_service(self._pickee_video_start_cli, request)

    async def dispatch_video_stream_stop(
        self, request: PickeeMainVideoStreamStop.Request
    ) -> PickeeMainVideoStreamStop.Response:
        """
        Pickee에게 영상 스트림 중지 명령
        """
        logger.info("Dispatching video stream stop: %s", request)
        return await self._call_service(self._pickee_video_stop_cli, request)

    def set_product_service(self, product_service: ProductService) -> None:
        """
        외부에서 ProductService를 주입받기 위한 메서드.
        Main Service 노드에서 주입해줍니다.
        """
        self._product_service = product_service

    def set_inventory_service(self, inventory_service: InventoryService) -> None:
        """
        외부에서 InventoryService를 주입받기 위한 메서드.
        """
        self._inventory_service = inventory_service

    # === 서비스 콜백 함수들 ===

    def _get_product_location_callback(
        self, request: MainGetProductLocation.Request, response: MainGetProductLocation.Response
    ) -> MainGetProductLocation.Response:
        """/main/get_product_location 서비스 요청을 처리합니다."""
        product_id = request.product_id
        logger.info("Received request to get location for product %d", product_id)

        if not hasattr(self, "_product_service"):
            response.success = False
            response.message = "ProductService is not initialized in RobotCoordinator."
            logger.error(response.message)
            return response

        location_info = self._product_service.get_product_location_sync(product_id)

        if location_info:
            response.success = True
            response.warehouse_id = location_info["warehouse_id"]
            response.section_id = location_info["section_id"]
            response.message = "Location found."
        else:
            response.success = False
            response.message = f"Product with ID {product_id} not found."
            logger.warning(response.message)
        
        return response

    async def _get_location_pose_callback(
        self, request: MainGetLocationPose.Request, response: MainGetLocationPose.Response
    ) -> MainGetLocationPose.Response:
        """/main/get_location_pose 서비스 요청을 처리합니다."""
        location_id = request.location_id
        logger.info(f"Received request to get pose for location {location_id}")

        if not hasattr(self, "_inventory_service"):
            response.success = False
            response.message = "InventoryService is not initialized in RobotCoordinator."
            logger.error(response.message)
            return response

        pose_info = await self._inventory_service.get_location_pose(location_id)

        if pose_info:
            response.success = True
            response.pose = Pose2D(x=pose_info['x'], y=pose_info['y'], theta=pose_info['theta'])
            response.message = "Pose found."
        else:
            response.success = False
            response.message = f"Location with ID {location_id} not found."
            logger.warning(response.message)
        
        return response

    async def check_packee_availability(
        self, request: PackeePackingCheckAvailability.Request
    ) -> PackeePackingCheckAvailability.Response:
        """
        Packee 작업 가능 여부 확인
        
        현재 가용한 포장 로봇이 있는지 확인합니다.
        
        Args:
            request: (빈 요청)
            
        Returns:
            Response: available (bool), robot_id (int)
        """
        logger.info("Checking packee availability: %s", request)
        return await self._call_service(self._packee_check_cli, request)

    async def dispatch_pack_task(
        self, request: PackeePackingStart.Request
    ) -> PackeePackingStart.Response:
        """
        Packee에게 포장 작업 시작 명령
        
        Args:
            request: 포장 요청 (robot_id, order_id)
            
        Returns:
            Response: 성공 여부
            
        참고: Main_vs_Pac_Main.md - /packee/packing/start
        """
        logger.info("Dispatching pack task: %s", request)
        return await self._call_service(self._packee_start_cli, request)

    def set_status_callbacks(
        self,
        pickee_status_cb: Optional[Callable[[PickeeRobotStatus], None]] = None,
        pickee_move_cb: Optional[Callable[[PickeeMoveStatus], None]] = None,
        pickee_arrival_cb: Optional[Callable[[PickeeArrival], None]] = None,
        pickee_handover_cb: Optional[Callable[[PickeeCartHandover], None]] = None,
        pickee_product_detected_cb: Optional[Callable[[PickeeProductDetection], None]] = None,
        pickee_selection_cb: Optional[Callable[[PickeeProductSelection], None]] = None,
        packee_status_cb: Optional[Callable[[PackeeRobotStatus], None]] = None,
        packee_complete_cb: Optional[Callable[[PackeePackingComplete], None]] = None,
    ) -> None:
        """
        로봇 이벤트 콜백 등록
        
        OrderService 등에서 로봇 상태 변화를 감지하기 위해 사용합니다.
        """
        self._pickee_status_cb = pickee_status_cb
        self._pickee_move_cb = pickee_move_cb
        self._pickee_arrival_cb = pickee_arrival_cb
        self._pickee_handover_cb = pickee_handover_cb
        self._pickee_product_detected_cb = pickee_product_detected_cb
        self._pickee_selection_cb = pickee_selection_cb
        self._packee_status_cb = packee_status_cb
        self._packee_complete_cb = packee_complete_cb

    async def _call_service(self, client, request):
        """
        ROS2 서비스 호출 헬퍼
        
        서비스가 준비될 때까지 대기하고 비동기로 호출합니다.
        
        Args:
            client: ROS2 서비스 클라이언트
            request: 요청 객체
            
        Returns:
            응답 객체
            
        Raises:
            RuntimeError: 서비스가 1초 내에 준비되지 않으면 발생
        """
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"Service {client.srv_name} unavailable")
        future = client.call_async(request)
        await future
        return future.result()

    # === 토픽 콜백 함수들 ===
    
    def _on_pickee_status(self, msg: PickeeRobotStatus) -> None:
        """Pickee 상태 토픽 콜백"""
        self._ros_cache["pickee_status"] = msg
        if self._pickee_status_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_status_cb):
                asyncio.create_task(self._pickee_status_cb(msg))
            else:
                self._pickee_status_cb(msg)

    def _on_pickee_move(self, msg: PickeeMoveStatus) -> None:
        """Pickee 이동 상태 토픽 콜백"""
        self._ros_cache["pickee_move"] = msg
        if self._pickee_move_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_move_cb):
                asyncio.create_task(self._pickee_move_cb(msg))
            else:
                self._pickee_move_cb(msg)

    def _on_pickee_arrival(self, msg: PickeeArrival) -> None:
        """Pickee 도착 토픽 콜백"""
        self._ros_cache["pickee_arrival"] = msg
        if self._pickee_arrival_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_arrival_cb):
                asyncio.create_task(self._pickee_arrival_cb(msg))
            else:
                self._pickee_arrival_cb(msg)

    def _on_pickee_handover(self, msg: PickeeCartHandover) -> None:
        """Pickee 장바구니 전달 완료 토픽 콜백"""
        self._ros_cache["pickee_handover"] = msg
        if self._pickee_handover_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_handover_cb):
                asyncio.create_task(self._pickee_handover_cb(msg))
            else:
                self._pickee_handover_cb(msg)

    def _on_product_detected(self, msg: PickeeProductDetection) -> None:
        """Pickee 상품 인식 완료 토픽 콜백"""
        self._ros_cache["product_detected"] = msg
        if self._pickee_product_detected_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_product_detected_cb):
                asyncio.create_task(self._pickee_product_detected_cb(msg))
            else:
                self._pickee_product_detected_cb(msg)

    def _on_pickee_selection(self, msg: PickeeProductSelection) -> None:
        """Pickee 상품 선택 토픽 콜백"""
        self._ros_cache["pickee_selection"] = msg
        if self._pickee_selection_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._pickee_selection_cb):
                asyncio.create_task(self._pickee_selection_cb(msg))
            else:
                self._pickee_selection_cb(msg)

    def _on_packee_status(self, msg: PackeeRobotStatus) -> None:
        """Packee 상태 토픽 콜백"""
        self._ros_cache["packee_status"] = msg
        if self._packee_status_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._packee_status_cb):
                asyncio.create_task(self._packee_status_cb(msg))
            else:
                self._packee_status_cb(msg)

    def _on_packee_complete(self, msg: PackeePackingComplete) -> None:
        """Packee 포장 완료 토픽 콜백"""
        self._ros_cache["packee_complete"] = msg
        if self._packee_complete_cb:
            # async 콜백을 asyncio task로 실행
            if asyncio.iscoroutinefunction(self._packee_complete_cb):
                asyncio.create_task(self._packee_complete_cb(msg))
            else:
                self._packee_complete_cb(msg)
