"""
ROS2 로봇 코디네이터

Pickee/Packee 로봇들과의 ROS2 통신을 담당합니다.
- ROS2 서비스 호출 (로봇에게 명령)
- ROS2 토픽 구독 (로봇 상태 수신)
- 로봇 상태 캐싱 및 관리
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import (
    PackeePackingComplete,
    PackeeRobotStatus,
    PickeeMoveStatus,
    PickeeProductSelection,
    PickeeRobotStatus,
)
from shopee_interfaces.srv import (
    PackeePackingCheckAvailability,
    PackeePackingStart,
    PickeeProductProcessSelection,
    PickeeWorkflowStartTask,
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
        # 상품 선택 결과 (사용자가 선택한 상품)
        self._pickee_selection_sub = self.create_subscription(
            PickeeProductSelection, "/pickee/product/selection_result", self._on_pickee_selection, 10
        )
        
        # === Packee 토픽 구독 ===
        # 포장 로봇 상태
        self._packee_status_sub = self.create_subscription(
            PackeeRobotStatus, "/packee/robot_status", self._on_packee_status, 10
        )
        # 포장 완료 알림
        self._packee_complete_sub = self.create_subscription(
            PackeePackingComplete, "/packee/packing_complete", self._on_packee_complete, 10
        )

        # === ROS2 서비스 클라이언트 ===
        # Pickee: 작업 시작 명령
        self._pickee_start_cli = self.create_client(PickeeWorkflowStartTask, "/pickee/workflow/start_task")
        # Pickee: 상품 선택 처리
        self._pickee_process_cli = self.create_client(PickeeProductProcessSelection, "/pickee/product/process_selection")
        # Packee: 작업 가능 여부 확인
        self._packee_check_cli = self.create_client(PackeePackingCheckAvailability, "/packee/packing/check_availability")
        # Packee: 포장 시작 명령
        self._packee_start_cli = self.create_client(PackeePackingStart, "/packee/packing/start")

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
        pickee_selection_cb: Optional[Callable[[PickeeProductSelection], None]] = None,
        packee_status_cb: Optional[Callable[[PackeeRobotStatus], None]] = None,
        packee_complete_cb: Optional[Callable[[PackeePackingComplete], None]] = None,
    ) -> None:
        """
        로봇 이벤트 콜백 등록
        
        OrderService 등에서 로봇 상태 변화를 감지하기 위해 사용합니다.
        
        Args:
            pickee_status_cb: Pickee 상태 변화 시 호출
            pickee_selection_cb: 사용자가 상품 선택 시 호출
            packee_status_cb: Packee 상태 변화 시 호출
            packee_complete_cb: 포장 완료 시 호출
        """
        self._pickee_status_cb = pickee_status_cb
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
            self._pickee_status_cb(msg)

    def _on_pickee_move(self, msg: PickeeMoveStatus) -> None:
        """Pickee 이동 상태 토픽 콜백"""
        self._ros_cache["pickee_move"] = msg

    def _on_pickee_selection(self, msg: PickeeProductSelection) -> None:
        """Pickee 상품 선택 토픽 콜백"""
        self._ros_cache["pickee_selection"] = msg
        if self._pickee_selection_cb:
            self._pickee_selection_cb(msg)

    def _on_packee_status(self, msg: PackeeRobotStatus) -> None:
        """Packee 상태 토픽 콜백"""
        self._ros_cache["packee_status"] = msg
        if self._packee_status_cb:
            self._packee_status_cb(msg)

    def _on_packee_complete(self, msg: PackeePackingComplete) -> None:
        """Packee 포장 완료 토픽 콜백"""
        self._ros_cache["packee_complete"] = msg
        if self._packee_complete_cb:
            self._packee_complete_cb(msg)
