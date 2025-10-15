"""
주문 관리 및 로봇 오케스트레이션 서비스

주문 생성부터 완료까지의 전체 워크플로우를 관리합니다.
- 주문 상태 머신
- Pickee/Packee 로봇 협업
- 이벤트 기반 상태 전환
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
)
from shopee_interfaces.msg import ProductLocation

from .config import settings
from .database_models import Customer, Order, OrderItem, Product, RobotHistory
from .event_bus import EventBus
from .robot_allocator import AllocationContext, RobotAllocator
from .robot_state_store import RobotStateStore
from .constants import RobotType, RobotStatus

if TYPE_CHECKING:
    from shopee_interfaces.msg import (
        PackeeAvailability,
        PackeePackingComplete,
        PickeeArrival,
        PickeeCartHandover,
        PickeeMoveStatus,
        PickeeProductDetection,
        PickeeProductSelection,
    )
    from .database_manager import DatabaseManager
    from .inventory_service import InventoryService
    from .robot_coordinator import RobotCoordinator

logger = logging.getLogger(__name__)


class OrderService:
    """
    주문 서비스
    
    주문의 전체 생명주기를 관리하는 핵심 서비스입니다.
    - 주문 생성 → 피킹(Pickee) → 포장(Packee) → 완료
    - 로봇 이벤트에 따른 상태 전환
    - App으로 진행 상황 알림
    """

    def __init__(
        self,
        db: "DatabaseManager",
        robot_coordinator: "RobotCoordinator",
        event_bus: EventBus,
        allocator: Optional[RobotAllocator] = None,
        state_store: Optional[RobotStateStore] = None,
        inventory_service: Optional["InventoryService"] = None,
    ) -> None:
        self._db = db
        self._robot = robot_coordinator
        self._event_bus = event_bus
        self._allocator = allocator
        self._state_store = state_store
        self._inventory_service = inventory_service
        self._detected_product_bbox: Dict[int, Dict[int, int]] = {}
        self._pickee_assignments: Dict[int, int] = {}
        self._packee_assignments: Dict[int, int] = {}
        self._reservation_monitors: Dict[tuple, asyncio.Task] = {}  # (order_id, robot_id) -> Task
        self._order_user_map: Dict[int, str] = {}

        # 로봇 장애 이벤트 구독 (자동 복구용)
        if settings.ROBOT_AUTO_RECOVERY_ENABLED:
            self._event_bus.subscribe("robot_failure", self.handle_robot_failure)
        self._default_locale_messages = {
            "robot_moving": "상품 위치로 이동 중입니다",
            "robot_arrived": "섹션에 도착했습니다",
            "robot_arrived_generic": "목적지에 도착했습니다",
            "product_detected": "상품을 선택해주세요",
            "cart_add_success": "상품이 장바구니에 담겼습니다",
            "cart_add_fail": "상품 담기에 실패했습니다",
            "packing_success": "포장을 완료했습니다",
            "packing_fail": "포장에 실패했습니다",
        }
        self._status_progress_map = {
            1: 10,
            2: 30,
            3: 60,
            4: 70,
            5: 80,
            6: 90,
            7: 95,
            8: 100,
            9: 100,
        }
        self._status_destination_map = {
            1: "PAYMENT",
            2: "SHELF",
            3: settings.DESTINATION_PACKING_NAME,
            4: settings.DESTINATION_PACKING_NAME,
            5: settings.DESTINATION_PACKING_NAME,
            6: settings.DESTINATION_DELIVERY_NAME,
            7: settings.DESTINATION_DELIVERY_NAME,
            8: settings.DESTINATION_DELIVERY_NAME,
            9: settings.DESTINATION_RETURN_NAME,
        }

    def _register_order_user(self, order_id: int, user_id: str) -> None:
        """order_id와 user_id 연관 정보를 저장합니다."""
        self._order_user_map[order_id] = user_id

    def _clear_order_user(self, order_id: int) -> None:
        """주문 완료/종료 시 매핑을 정리합니다."""
        self._order_user_map.pop(order_id, None)

    def _get_order_user_id(self, order_id: int) -> Optional[str]:
        """order_id에 해당하는 사용자 ID를 조회합니다."""
        user_id = self._order_user_map.get(order_id)
        if user_id:
            return user_id
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if order and order.customer:
                user_id = order.customer.id
                self._order_user_map[order_id] = user_id
                return user_id
        return None

    async def _push_to_user(self, payload: Dict[str, Any], order_id: Optional[int]) -> None:
        """
        사용자별 푸시 메시지를 발송합니다.

        Args:
            payload: 이벤트 페이로드
            order_id: 대상 주문 ID (없으면 브로드캐스트)
        """
        message = dict(payload)
        if order_id is not None:
            user_id = self._get_order_user_id(order_id)
            if user_id:
                message.setdefault("target_user_id", user_id)
        await self._event_bus.publish("app_push", message)

    def _calculate_order_summary(self, session, order_id: int) -> Tuple[int, int]:
        """주문 총 수량과 금액 계산"""
        items = session.query(OrderItem).filter_by(order_id=order_id).all()
        total_items = sum(item.quantity for item in items)

        product_ids = [item.product_id for item in items]
        price_map: Dict[int, int] = {}
        if product_ids:
            products = (
                session.query(Product)
                .filter(Product.product_id.in_(product_ids))
                .all()
            )
            price_map = {prod.product_id: prod.price for prod in products}

        total_price = sum(
            item.quantity * price_map.get(item.product_id, 0) for item in items
        )
        return total_items, total_price

    def _load_products(self, session, product_ids: List[int]) -> Dict[int, Product]:
        """상품 ID 목록을 받아 Product 정보를 매핑으로 반환"""
        if not product_ids:
            return {}
        products = (
            session.query(Product)
            .filter(Product.product_id.in_(product_ids))
            .all()
        )
        return {product.product_id: product for product in products}

    def _build_product_locations(self, session, order_id: int) -> List[ProductLocation]:
        """주문 항목을 기반으로 ProductLocation 리스트 생성"""
        locations: List[ProductLocation] = []
        items = session.query(OrderItem).filter_by(order_id=order_id).all()
        for item in items:
            product = (
                session.query(Product)
                .filter_by(product_id=item.product_id)
                .first()
            )
            if not product or not product.section or not product.section.shelf:
                logger.warning(
                    "Skipping product %s for order %s due to missing section/shelf mapping",
                    item.product_id,
                    order_id,
                )
                continue
            locations.append(
                ProductLocation(
                    product_id=product.product_id,
                    location_id=product.section.shelf.location_id,
                    section_id=product.section_id,
                    quantity=item.quantity,
                )
            )
        return locations

    def _status_progress(self, status: int) -> int:
        return self._status_progress_map.get(status, 0)

    def _default_destination(self, status: int) -> str:
        return self._status_destination_map.get(status, "")

    async def _monitor_reservation_timeout(
        self,
        robot_id: int,
        order_id: int,
        robot_type: RobotType,
        timeout: float = None,
    ) -> None:
        """
        예약 후 타임아웃 모니터링.

        로봇이 예약 후 일정 시간 내에 WORKING 상태로 전환되지 않으면
        타임아웃 처리하여 예약을 해제하고 알림을 발행합니다.
        """
        if timeout is None:
            timeout = settings.ROBOT_RESERVATION_TIMEOUT

        logger.debug(
            "Starting reservation timeout monitor for robot %d, order %d (timeout=%ds)",
            robot_id, order_id, timeout
        )

        await asyncio.sleep(timeout)

        # 타임아웃 후 상태 확인
        if self._state_store:
            state = await self._state_store.get_state(robot_id)
            if state and state.status == RobotStatus.IDLE.value and state.reserved:
                # 아직도 IDLE이면서 예약 상태이면 타임아웃
                logger.warning(
                    "Reservation timeout: robot %d still IDLE after %ds for order %d",
                    robot_id, timeout, order_id
                )

                # 예약 해제
                if self._allocator:
                    await self._allocator.release_robot(robot_id, order_id)

                # 할당 딕셔너리에서 제거
                if robot_type == RobotType.PICKEE:
                    self._pickee_assignments.pop(order_id, None)
                else:
                    self._packee_assignments.pop(order_id, None)

                # 이벤트 발행
                await self._event_bus.publish("reservation_timeout", {
                    "robot_id": robot_id,
                    "order_id": order_id,
                    "robot_type": robot_type.value,
                    "timeout": timeout,
                })

                # 재고 복구 및 주문 실패 처리
                await self._fail_order(order_id, f"Robot {robot_id} timeout after {timeout}s")

                # 사용자 알림 (이미 _fail_order에서 발송하지만 타임아웃 전용 메시지 추가)
                await self._push_to_user(
                    {
                        "type": "robot_timeout_notification",
                        "result": False,
                        "error_code": "ROBOT_003",
                        "data": {
                            "order_id": order_id,
                            "robot_id": robot_id,
                            "robot_type": robot_type.value,
                        },
                        "message": f"로봇 {robot_id}이(가) 응답하지 않습니다. 다시 시도해주세요.",
                    },
                    order_id=order_id,
                )
            else:
                logger.debug(
                    "Reservation timeout monitor: robot %d is working normally for order %d",
                    robot_id, order_id
                )

    def _start_reservation_monitor(
        self,
        robot_id: int,
        order_id: int,
        robot_type: RobotType,
    ) -> None:
        """예약 타임아웃 모니터링 시작"""
        key = (order_id, robot_id)
        # 기존 모니터가 있으면 취소
        if key in self._reservation_monitors:
            self._reservation_monitors[key].cancel()

        # 새 모니터 시작
        monitor_task = asyncio.create_task(
            self._monitor_reservation_timeout(robot_id, order_id, robot_type)
        )
        self._reservation_monitors[key] = monitor_task

    def _cancel_reservation_monitor(self, order_id: int, robot_id: int) -> None:
        """예약 타임아웃 모니터링 취소 (정상 작업 시작 시)"""
        key = (order_id, robot_id)
        if key in self._reservation_monitors:
            self._reservation_monitors[key].cancel()
            del self._reservation_monitors[key]
            logger.debug(
                "Cancelled reservation timeout monitor for robot %d, order %d",
                robot_id, order_id
            )

    async def _emit_work_info_notification(
        self,
        order_id: int,
        robot_id: int,
        destination: str = "",
    ) -> None:
        """관리자용 작업 정보 알림 발행"""
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if not order or not order.customer:
                return

            customer = order.customer
            order_items = list(order.items or [])
            order_item_ids = [item.order_item_id for item in order_items if item.order_item_id]

            active_duration = 0
            if order_item_ids:
                history = (
                    session.query(RobotHistory)
                    .filter(RobotHistory.robot_id == robot_id)
                    .filter(RobotHistory.order_item_id.in_(order_item_ids))
                    .order_by(RobotHistory.created_at.desc())
                    .first()
                )
                if history and history.active_duration is not None:
                    active_duration = history.active_duration

            progress = self._status_progress(order.order_status)
            destination_value = destination or self._default_destination(order.order_status)

            data = {
                "robot_id": robot_id,
                "destination": destination_value,
                "progress": progress,
                "active_duration": active_duration,
                "user_id": customer.id,
                "customer_name": customer.name,
                "customer_allergy_info_id": customer.allergy_info_id,
                "customer_is_vegan": customer.is_vegan,
            }

        await self._push_to_user(
            {
                "type": "work_info_notification",
                "result": True,
                "error_code": None,
                "data": data,
                "message": "작업 정보 업데이트",
            },
            order_id=order_id,
        )

    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """
        주문 생성 및 Pickee 작업 할당
        """
        logger.info("Creating order for user '%s' with %d items", user_id, len(items))
        reserved_robot_id: Optional[int] = None
        order_id: Optional[int] = None
        reserved_stocks: List[Tuple[int, int]] = []  # (product_id, quantity)

        with self._db.session_scope() as session:
            try:
                customer = session.query(Customer).filter_by(id=user_id).first()
                if not customer:
                    raise ValueError(f"Order creation failed: User '{user_id}' not found.")

                # ===== 1. 재고 확인 및 예약 (추가!) =====
                if self._inventory_service:
                    for item in items:
                        product_id = item["product_id"]
                        quantity = item["quantity"]

                        # 재고 확인 및 차감
                        stock_ok = await self._inventory_service.check_and_reserve_stock(
                            product_id, quantity
                        )

                        if not stock_ok:
                            # 재고 부족 시 이미 차감한 재고 복구
                            for reserved_pid, reserved_qty in reserved_stocks:
                                await self._inventory_service.release_stock(
                                    reserved_pid, reserved_qty
                                )

                            raise ValueError(
                                f"Insufficient stock for product {product_id}. "
                                f"Requested: {quantity}"
                            )

                        # 재고 예약 성공 기록
                        reserved_stocks.append((product_id, quantity))

                    logger.info("Stock reservation successful for order")
                else:
                    logger.warning("InventoryService not available, skipping stock check")

                # ===== 2. 주문 생성 (기존 로직) =====
                new_order = Order(
                    customer_id=customer.customer_id,
                    start_time=datetime.now(timezone.utc),
                    order_status=1,  # 1: PAID
                )
                session.add(new_order)
                session.flush()
                order_id = new_order.order_id

                self._register_order_user(order_id, customer.id)

                if self._allocator:
                    context = AllocationContext(
                        order_id=order_id,
                        required_type=RobotType.PICKEE,
                    )
                    reserved_state = await self._allocator.reserve_robot(context)
                    if not reserved_state:
                        raise RuntimeError("No available Pickee robot.")
                    robot_id = reserved_state.robot_id
                    reserved_robot_id = robot_id
                else:
                    robot_id = 1  # 기본값 (이전 동작 유지)

                product_locations = []
                for item in items:
                    product_id = item["product_id"]
                    quantity = item["quantity"]
                    product = session.query(Product).filter_by(product_id=product_id).first()
                    if not product:
                        raise ValueError(f"Product with ID {product_id} not found.")

                    new_item = OrderItem(
                        order_id=new_order.order_id,
                        product_id=product_id,
                        quantity=quantity,
                    )
                    session.add(new_item)
                    product_locations.append(
                        ProductLocation(
                            product_id=product.product_id,
                            location_id=product.section.shelf.location_id,
                            section_id=product.section_id,
                            quantity=quantity,
                        )
                    )

                request = PickeeWorkflowStartTask.Request(
                    robot_id=robot_id,
                    order_id=new_order.order_id,
                    user_id=user_id,
                    product_list=product_locations,
                )
                response = await self._robot.dispatch_pick_task(request)

                if not response.success:
                    raise RuntimeError(f"Failed to dispatch pick task: {response.message}")

                if product_locations:
                    first_location = product_locations[0]
                    try:
                        move_req = PickeeWorkflowMoveToSection.Request(
                            robot_id=robot_id,
                            order_id=new_order.order_id,
                            location_id=first_location.location_id,
                            section_id=first_location.section_id,
                        )
                        await self._robot.dispatch_move_to_section(move_req)
                    except Exception as move_exc:  # noqa: BLE001
                        logger.error(
                            "Failed to dispatch move_to_section for order %d: %s",
                            new_order.order_id,
                            move_exc,
                        )
                        # 예약 해제
                        if self._allocator and reserved_robot_id is not None:
                            await self._allocator.release_robot(reserved_robot_id, new_order.order_id)
                            self._pickee_assignments.pop(new_order.order_id, None)
                        raise  # 주문 생성 실패로 전파

                session.commit()
                logger.info("Order %d created and dispatched to robot %d", new_order.order_id, robot_id)
                if self._allocator and reserved_robot_id is not None:
                    self._pickee_assignments[new_order.order_id] = reserved_robot_id
                    # 타임아웃 모니터링 시작
                    self._start_reservation_monitor(reserved_robot_id, new_order.order_id, RobotType.PICKEE)
                return new_order.order_id, robot_id

            except Exception as e:
                logger.exception("Order creation failed: %s", e)
                session.rollback()

                # ===== 5. 실패 시 재고 복구 (추가!) =====
                if self._inventory_service:
                    for reserved_pid, reserved_qty in reserved_stocks:
                        try:
                            await self._inventory_service.release_stock(
                                reserved_pid, reserved_qty
                            )
                        except Exception as release_exc:
                            logger.error(
                                f"Failed to release stock for product {reserved_pid}: {release_exc}"
                            )

                if self._allocator and reserved_robot_id is not None and order_id is not None:
                    await self._allocator.release_robot(reserved_robot_id, order_id)
                if order_id is not None:
                    self._clear_order_user(order_id)
                return None

    async def select_product(
        self, order_id: int, robot_id: int, bbox_number: int, product_id: int
    ) -> bool:
        """
        사용자의 상품 선택을 처리하고 Pickee에게 피킹을 지시합니다.
        """
        logger.info(
            "Dispatching product selection: Order=%d, Robot=%d, BBox=%d, Product=%d",
            order_id, robot_id, bbox_number, product_id
        )
        try:
            request = PickeeProductProcessSelection.Request(
                robot_id=robot_id,
                order_id=order_id,
                product_id=product_id,
                bbox_number=bbox_number,
            )
            response = await self._robot.dispatch_pick_process(request)
            return response.success
        except Exception as e:
            logger.exception("Failed to dispatch product selection: %s", e)
            return False

    async def end_shopping(self, order_id: int, robot_id: int) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        쇼핑을 종료하고 로봇을 포장대로 이동시킵니다.
        """
        logger.info("Ending shopping for order %d", order_id)
        try:
            request = PickeeWorkflowEndShopping.Request(robot_id=robot_id, order_id=order_id)
            response = await self._robot.dispatch_shopping_end(request)

            if not response.success:
                logger.error("Robot failed to end shopping for order %d: %s", order_id, response.message)
                return False, None

            summary: Optional[Dict[str, int]] = None
            with self._db.session_scope() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()
                if order:
                    order.order_status = 3  # 3: PICKED_UP
                    logger.info("Order %d status updated to PICKED_UP", order_id)
                    total_items, total_price = self._calculate_order_summary(session, order_id)
                    summary = {
                        "total_items": total_items,
                        "total_price": total_price,
                    }
                else:
                    logger.error("Cannot update status for non-existent order %d", order_id)
                    return False, None

            packaging_location_id = settings.PICKEE_PACKING_LOCATION_ID
            if packaging_location_id:
                try:
                    move_pack_req = PickeeWorkflowMoveToPackaging.Request(
                        robot_id=robot_id,
                        order_id=order_id,
                        location_id=packaging_location_id,
                    )
                    await self._robot.dispatch_move_to_packaging(move_pack_req)
                except Exception as pack_exc:  # noqa: BLE001
                    logger.error("Failed to dispatch move_to_packaging for order %d: %s", order_id, pack_exc)
                    # 이 시점에서는 이미 주문 상태가 변경되었으므로
                    # 사용자에게 알림만 보내고 예약은 유지 (수동 복구 가능하도록)
                    await self._push_to_user(
                        {
                            "type": "robot_command_failed",
                            "result": False,
                            "error_code": "ROBOT_002",
                            "data": {
                                "order_id": order_id,
                                "robot_id": robot_id,
                                "command": "move_to_packaging",
                            },
                            "message": "로봇 이동 명령이 실패했습니다. 직원이 확인 중입니다.",
                        },
                        order_id=order_id,
                    )

            return True, summary
        except Exception as e:
            logger.exception("Failed to end shopping for order %d: %s", order_id, e)
            return False, None

    async def handle_moving_status(self, msg: "PickeeMoveStatus") -> None:
        """
        Pickee의 이동 상태 이벤트를 처리합니다.
        """
        logger.info("Handling moving status for order %d", msg.order_id)

        # 타임아웃 모니터 취소 (로봇이 실제로 작업 시작함)
        self._cancel_reservation_monitor(msg.order_id, msg.robot_id)

        # PickeeMoveStatus에는 location_id만 있음
        destination_text = f"LOCATION_{msg.location_id}"
        await self._push_to_user(
            {
                "type": "robot_moving_notification",
                "result": True,
                "error_code": None,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "destination": destination_text,
                    "location_id": msg.location_id,
                },
                "message": self._default_locale_messages["robot_moving"],
            },
            order_id=msg.order_id,
        )
        await self._emit_work_info_notification(
            order_id=msg.order_id,
            robot_id=msg.robot_id,
            destination=destination_text,
        )

    async def handle_arrival_notice(self, msg: "PickeeArrival") -> None:
        """
        Pickee의 도착 이벤트를 처리하고, 상품 인식 단계를 시작합니다.
        """
        is_section = msg.section_id is not None and msg.section_id >= 0
        location_desc = f"section {msg.section_id}" if is_section else "non-section location"
        logger.info("Handling arrival notice for order %d at %s", msg.order_id, location_desc)
        section_payload = msg.section_id if is_section else -1
        message_key = "robot_arrived" if is_section else "robot_arrived_generic"
        await self._push_to_user(
            {
                "type": "robot_arrived_notification",
                "result": True,
                "error_code": None,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "location_id": msg.location_id,
                    "section_id": section_payload,
                },
                "message": self._default_locale_messages[message_key],
            },
            order_id=msg.order_id,
        )
        if not is_section:
            logger.debug(
                "Skipping product detection for order %d because arrival location is not a section",
                msg.order_id,
            )
            return
        try:
            with self._db.session_scope() as session:
                product_ids = [
                    item.product_id for item in 
                    session.query(OrderItem).filter_by(order_id=msg.order_id).all()
                ]
            if not product_ids:
                logger.warning("No products to detect for order %d", msg.order_id)
                return

            req = PickeeProductDetect.Request(
                robot_id=msg.robot_id,
                order_id=msg.order_id,
                product_ids=product_ids
            )
            await self._robot.dispatch_product_detect(req)
        except Exception as e:
            logger.exception("Failed to start product detection for order %d: %s", msg.order_id, e)

    async def handle_product_detected(self, msg: "PickeeProductDetection") -> None:
        """
        Pickee의 상품 인식 완료 이벤트를 처리합니다.
        """
        logger.info("Handling product detection for order %d. Found %d products.", msg.order_id, len(msg.products))
        product_ids = [p.product_id for p in msg.products]

        # DB 세션 안에서 딕셔너리 데이터 생성
        products_data = []
        if product_ids:
            with self._db.session_scope() as session:
                product_map = self._load_products(session, product_ids)
                # 세션 안에서 데이터 추출
                products_data = [
                    {
                        "product_id": detected.product_id,
                        "name": product_map.get(detected.product_id).name if product_map.get(detected.product_id) else "",
                        "bbox_number": detected.bbox_number,
                    }
                    for detected in msg.products
                ]

        self._detected_product_bbox[msg.order_id] = {
            detected.product_id: detected.bbox_number for detected in msg.products
        }

        await self._push_to_user(
            {
                "type": "product_selection_start",
                "result": True,
                "error_code": None,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "products": products_data,
                },
                "message": self._default_locale_messages["product_detected"],
            },
            order_id=msg.order_id,
        )

    async def handle_pickee_selection(self, msg: "PickeeProductSelection") -> None:
        """
        Pickee의 상품 선택 완료 이벤트를 처리합니다.
        """
        logger.info("Handling pickee selection result for order %d", msg.order_id)
        # DB 세션 안에서 모든 데이터 추출
        product_name = ""
        product_price = 0
        total_items = 0
        total_price = 0

        with self._db.session_scope() as session:
            order_item = (
                session.query(OrderItem)
                .filter_by(order_id=msg.order_id, product_id=msg.product_id)
                .first()
            )
            if not order_item:
                logger.error(
                    "Received selection result for non-existent order item. Order: %d, Product: %d",
                    msg.order_id,
                    msg.product_id,
                )
                return

            product = (
                session.query(Product)
                .filter_by(product_id=msg.product_id)
                .first()
            )
            if product:
                product_name = product.name
                product_price = product.price

            total_items, total_price = self._calculate_order_summary(session, msg.order_id)

        message = msg.message or (
            self._default_locale_messages["cart_add_success"]
            if msg.success
            else self._default_locale_messages["cart_add_fail"]
        )
        error_code = None if msg.success else "ROBOT_002"
        await self._push_to_user(
            {
                "type": "cart_update_notification",
                "result": msg.success,
                "error_code": error_code,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "action": "add" if msg.success else "add_fail",
                    "product": {
                        "product_id": msg.product_id,
                        "name": product_name,
                        "quantity": msg.quantity,
                        "price": product_price,
                    },
                    "total_items": total_items,
                    "total_price": total_price,
                },
                "message": message,
            },
            order_id=msg.order_id,
        )
        bbox_map = self._detected_product_bbox.get(msg.order_id)
        if bbox_map and msg.product_id in bbox_map:
            bbox_map.pop(msg.product_id, None)
            if not bbox_map:
                self._detected_product_bbox.pop(msg.order_id, None)

    async def handle_cart_handover(self, msg: "PickeeCartHandover") -> None:
        """
        Pickee의 장바구니 전달 완료 이벤트를 처리하여 Packee 작업을 시작하고 복귀 여부를 판단합니다.
        """
        order_id = msg.order_id
        robot_id = msg.robot_id
        logger.info("Handling cart handover for order %d, starting packing process.", order_id)
        packee_robot_id: Optional[int] = None
        allocator_reserved = False
        state_reserved = False
        state_reserved_robot_id: Optional[int] = None
        try:
            if self._allocator:
                context = AllocationContext(
                    order_id=order_id,
                    required_type=RobotType.PACKEE,
                )
                reserved_state = await self._allocator.reserve_robot(context)
                if reserved_state:
                    packee_robot_id = reserved_state.robot_id
                    allocator_reserved = True
                    logger.info("Reserved Packee robot %d for order %d via allocator.", packee_robot_id, order_id)
                else:
                    logger.info("Allocator could not find available Packee robot for order %d. Falling back to ROS check.", order_id)

            if packee_robot_id is None and self._state_store:
                available_packees = await self._state_store.list_available(RobotType.PACKEE)
                if not available_packees:
                    logger.error("No available Packee robot for order %d (state store).", order_id)
                    return

                candidate_state = available_packees[0]
                reserved = await self._state_store.try_reserve(candidate_state.robot_id, order_id)
                if not reserved:
                    logger.error(
                        "Failed to reserve Packee robot %d for order %d via state store.",
                        candidate_state.robot_id,
                        order_id,
                    )
                    return

                packee_robot_id = candidate_state.robot_id
                state_reserved = True
                state_reserved_robot_id = candidate_state.robot_id
                logger.info(
                    "Reserved Packee robot %d for order %d via state store fallback.",
                    packee_robot_id,
                    order_id,
                )

            availability_response = None
            if packee_robot_id is None:
                check_req = PackeePackingCheckAvailability.Request(
                    robot_id=0,
                    order_id=order_id,
                )
                availability_response = await self._robot.check_packee_availability(check_req)

                available = getattr(availability_response, "available", None)
                if available is None:
                    available = getattr(availability_response, "success", False)

                if not available:
                    logger.error("No available Packee robot for order %d.", order_id)
                    if allocator_reserved and packee_robot_id is not None:
                        await self._allocator.release_robot(packee_robot_id, order_id)
                    if state_reserved and state_reserved_robot_id is not None:
                        await self._state_store.release(state_reserved_robot_id, order_id)
                    return

                provided_robot_id = getattr(availability_response, "robot_id", None)
                if provided_robot_id:
                    packee_robot_id = provided_robot_id

            if packee_robot_id is None:
                logger.error("No Packee robot id available for order %d.", order_id)
                if state_reserved and state_reserved_robot_id is not None:
                    await self._state_store.release(state_reserved_robot_id, order_id)
                return

            start_req = PackeePackingStart.Request(robot_id=packee_robot_id, order_id=order_id)
            start_res = await self._robot.dispatch_pack_task(start_req)

            if not start_res.success:
                logger.error("Failed to start packing for order %d. Reason: %s", order_id, start_res.message)
                if allocator_reserved:
                    await self._allocator.release_robot(packee_robot_id, order_id)
                if state_reserved:
                    await self._state_store.release(packee_robot_id, order_id)
                return

            logger.info("Successfully dispatched packing task for order %d to robot %d.", order_id, packee_robot_id)
            self._packee_assignments[order_id] = packee_robot_id
            # 타임아웃 모니터링 시작
            self._start_reservation_monitor(packee_robot_id, order_id, RobotType.PACKEE)

            packing_payload = {
                "order_status": "PACKING",
                "product_id": 0,
                "product_name": "",
                "product_price": 0,
                "product_quantity": 0,
            }
            with self._db.session_scope() as session:
                order_item = (
                    session.query(OrderItem)
                    .filter_by(order_id=order_id)
                    .first()
                )
                if order_item:
                    product = (
                        session.query(Product)
                        .filter_by(product_id=order_item.product_id)
                        .first()
                    )
                    packing_payload.update(
                        {
                            "product_id": order_item.product_id,
                            "product_name": product.name if product else "",
                            "product_price": product.price if product else 0,
                            "product_quantity": order_item.quantity,
                        }
                    )

            await self._push_to_user(
                {
                    "type": "packing_info_notification",
                    "result": True,
                    "error_code": None,
                    "data": {**packing_payload, "order_id": order_id},
                    "message": "포장 정보 업데이트",
                },
                order_id=order_id,
            )

            home_location_id = settings.PICKEE_HOME_LOCATION_ID
            if home_location_id:
                try:
                    return_req = PickeeWorkflowReturnToBase.Request(
                        robot_id=robot_id,
                        location_id=home_location_id,
                    )
                    await self._robot.dispatch_return_to_base(return_req)
                except Exception as return_exc:  # noqa: BLE001
                    logger.warning("Failed to dispatch return_to_base for order %d: %s", order_id, return_exc)
                    # 복귀 실패는 치명적이지 않음 (이미 작업 완료)
                    # 로깅만 하고 예약은 정상 해제

            await self._release_pickee(order_id)

        except Exception as e:
            logger.exception("Failed to handle cart handover for order %d: %s", order_id, e)
            if allocator_reserved and packee_robot_id is not None:
                try:
                    self._packee_assignments.pop(order_id, None)
                    await self._allocator.release_robot(packee_robot_id, order_id)
                except Exception as release_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to release Packee robot %d for order %d after exception: %s",
                        packee_robot_id,
                        order_id,
                        release_exc,
                    )
            if state_reserved and state_reserved_robot_id is not None:
                try:
                    await self._state_store.release(state_reserved_robot_id, order_id)
                except Exception as release_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to release Packee robot %d via state store for order %d after exception: %s",
                        state_reserved_robot_id,
                        order_id,
                        release_exc,
                    )

    async def handle_packee_availability(self, msg: "PackeeAvailability") -> None:
        """
        Packee 작업 가능 여부 결과를 처리합니다.
        """
        logger.info(
            "Packee availability result for order %d (robot %d): available=%s cart_detected=%s message=%s",
            msg.order_id,
            msg.robot_id,
            msg.available,
            msg.cart_detected,
            msg.message,
        )
        if not msg.available:
            logger.warning(
                "Packee unavailable for order %d. Reason: %s", msg.order_id, msg.message
            )
        await self._emit_work_info_notification(
            order_id=msg.order_id,
            robot_id=msg.robot_id,
        )

    async def handle_packee_complete(self, msg: "PackeePackingComplete") -> None:
        """
        Packee의 포장 완료 이벤트를 처리합니다.
        """
        logger.info("Handling packee complete for order %d. Success: %s", msg.order_id, msg.success)

        # 타임아웃 모니터 취소
        self._cancel_reservation_monitor(msg.order_id, msg.robot_id)

        final_status = 8 if msg.success else 9  # 8: PACKED, 9: FAIL_PACK
        
        product_info = {
            "product_id": 0,
            "product_name": "",
            "product_price": 0,
            "product_quantity": 0,
        }

        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=msg.order_id).first()
            if not order:
                logger.error("Received packing complete for non-existent order %d", msg.order_id)
                return

            order.order_status = final_status
            order.end_time = datetime.now(timezone.utc)
            if not msg.success:
                order.failure_reason = msg.message
            
            logger.info("Order %d status updated to %d", msg.order_id, final_status)
            order_item = (
                session.query(OrderItem)
                .filter_by(order_id=msg.order_id)
                .first()
            )
            if order_item:
                product = (
                    session.query(Product)
                    .filter_by(product_id=order_item.product_id)
                    .first()
                )
                product_info = {
                    "product_id": order_item.product_id,
                    "product_name": product.name if product else "",
                    "product_price": product.price if product else 0,
                    "product_quantity": order_item.quantity,
                }
            
        order_status_text = "PACKED" if msg.success else "FAIL_PACK"
        packing_message = msg.message or (
            self._default_locale_messages["packing_success"]
            if msg.success
            else self._default_locale_messages["packing_fail"]
        )

        await self._push_to_user(
            {
                "type": "packing_info_notification",
                "result": msg.success,
                "error_code": None if msg.success else "ROBOT_002",
                "data": {
                    "order_id": msg.order_id,
                    "order_status": order_status_text,
                    **product_info,
                },
                "message": packing_message,
            },
            order_id=msg.order_id,
        )

        await self._emit_work_info_notification(
            order_id=msg.order_id,
            robot_id=msg.robot_id,
            destination=self._default_destination(final_status),
        )
        self._detected_product_bbox.pop(msg.order_id, None)
        await self._release_packee(msg.order_id)
        await self._release_pickee(msg.order_id)

        # 포장 실패 시 재고 복구
        if not msg.success:
            await self._release_stock_for_order(msg.order_id)

        if msg.success:
            self._clear_order_user(msg.order_id)

    async def _release_pickee(self, order_id: int) -> None:
        """Pickee 예약을 해제합니다."""
        robot_id = self._pickee_assignments.pop(order_id, None)
        if robot_id is None:
            return
        self._cancel_reservation_monitor(order_id, robot_id)
        if self._allocator:
            try:
                await self._allocator.release_robot(robot_id, order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to release Pickee robot %d for order %d: %s",
                    robot_id,
                    order_id,
                    exc,
                )
        elif self._state_store:
            try:
                await self._state_store.release(robot_id, order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to release Pickee robot %d via state store for order %d: %s",
                    robot_id,
                    order_id,
                    exc,
                )

    async def _release_packee(self, order_id: int) -> None:
        """Packee 예약을 해제합니다."""
        robot_id = self._packee_assignments.pop(order_id, None)
        if robot_id is None:
            return
        self._cancel_reservation_monitor(order_id, robot_id)
        if self._allocator:
            try:
                await self._allocator.release_robot(robot_id, order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to release Packee robot %d for order %d: %s",
                    robot_id,
                    order_id,
                    exc,
                )
        elif self._state_store:
            try:
                await self._state_store.release(robot_id, order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to release Packee robot %d via state store for order %d: %s",
                    robot_id,
                    order_id,
                    exc,
                )

    async def _reassign_pickee(self, order_id: int, new_robot_id: int) -> bool:
        """장애 발생 후 Pickee 로봇을 재할당합니다."""
        try:
            with self._db.session_scope() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()
                if not order or not order.customer:
                    raise ValueError(f"Cannot reassign pickee: order {order_id} not found or has no customer")
                user_id = order.customer.id
                product_locations = self._build_product_locations(session, order_id)
                if not product_locations:
                    raise ValueError(f"No product locations available for order {order_id}")

            request = PickeeWorkflowStartTask.Request(
                robot_id=new_robot_id,
                order_id=order_id,
                user_id=user_id,
                product_list=product_locations,
            )
            response = await self._robot.dispatch_pick_task(request)
            if not response.success:
                raise RuntimeError(f"Failed to dispatch pick task to robot {new_robot_id}: {response.message}")

            first_location = product_locations[0]
            try:
                move_req = PickeeWorkflowMoveToSection.Request(
                    robot_id=new_robot_id,
                    order_id=order_id,
                    location_id=first_location.location_id,
                    section_id=first_location.section_id,
                )
                await self._robot.dispatch_move_to_section(move_req)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to send move_to_section during reassign of order %d to robot %d: %s",
                    order_id,
                    new_robot_id,
                    exc,
                )

            self._pickee_assignments[order_id] = new_robot_id
            self._start_reservation_monitor(new_robot_id, order_id, RobotType.PICKEE)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Reassigning pickee robot failed: order=%d, new_robot=%d, reason=%s",
                order_id,
                new_robot_id,
                exc,
            )
            return False

    async def _reassign_packee(self, order_id: int, new_robot_id: int) -> bool:
        """장애 발생 후 Packee 로봇을 재할당합니다."""
        try:
            request = PackeePackingStart.Request(robot_id=new_robot_id, order_id=order_id)
            response = await self._robot.dispatch_pack_task(request)
            if not response.success:
                raise RuntimeError(f"Failed to dispatch pack task to robot {new_robot_id}: {response.message}")

            self._packee_assignments[order_id] = new_robot_id
            self._start_reservation_monitor(new_robot_id, order_id, RobotType.PACKEE)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Reassigning packee robot failed: order=%d, new_robot=%d, reason=%s",
                order_id,
                new_robot_id,
                exc,
            )
            return False

    async def _notify_reassignment_success(
        self,
        order_id: int,
        old_robot_id: int,
        new_robot_id: int,
        robot_type: str,
        status: str,
    ) -> None:
        """로봇 재할당 성공 알림."""
        await self._push_to_user(
            {
                "type": "robot_reassignment_notification",
                "result": True,
                "error_code": None,
                "data": {
                    "order_id": order_id,
                    "old_robot_id": old_robot_id,
                    "new_robot_id": new_robot_id,
                    "robot_type": robot_type,
                    "reason": f"Robot {old_robot_id} {status}",
                },
                "message": (
                    f"로봇 {old_robot_id}에 문제가 발생하여 로봇 {new_robot_id}으로 교체되었습니다."
                ),
            },
            order_id=order_id,
        )

    async def _notify_reassignment_failure(
        self,
        order_id: int,
        robot_id: int,
        robot_type: str,
        status: str,
        reason: str,
    ) -> None:
        """로봇 재할당 실패 알림."""
        await self._push_to_user(
            {
                "type": "robot_failure_notification",
                "result": False,
                "error_code": "ROBOT_001",
                "data": {
                    "order_id": order_id,
                    "robot_id": robot_id,
                    "robot_type": robot_type,
                    "status": status,
                    "reason": reason,
                },
                "message": self._format_reassignment_failure_message(robot_id, reason),
            },
            order_id=order_id,
        )

    def _format_reassignment_failure_message(self, robot_id: int, reason: str) -> str:
        """재할당 실패 메시지를 사유에 따라 생성"""
        if reason == "no_available_robot":
            return f"로봇 {robot_id}에 문제가 발생했으며, 현재 가용한 대체 로봇이 없습니다."
        if reason == "allocator_unavailable":
            return f"로봇 {robot_id}에 문제가 발생했지만, 관제 시스템에서 자동 재할당을 지원하지 않습니다."
        return f"로봇 {robot_id}에 문제가 발생했고, 재할당 중 오류가 발생했습니다."

    def get_detected_bbox(self, order_id: int, product_id: int) -> Optional[int]:
        """
        최근 상품 인식 결과에서 해당 상품의 bbox 번호를 조회합니다.
        """
        return self._detected_product_bbox.get(order_id, {}).get(product_id)

    async def _release_stock_for_order(self, order_id: int) -> None:
        """
        주문의 모든 상품 재고 복구

        Args:
            order_id: 재고를 복구할 주문 ID
        """
        if not self._inventory_service:
            logger.warning(f"Cannot release stock for order {order_id}: InventoryService not available")
            return

        try:
            with self._db.session_scope() as session:
                items = session.query(OrderItem).filter_by(order_id=order_id).all()

                for item in items:
                    await self._inventory_service.release_stock(
                        item.product_id, item.quantity
                    )
                    logger.info(
                        f"Released stock for failed order {order_id}: "
                        f"product={item.product_id}, qty={item.quantity}"
                    )
        except Exception as exc:
            logger.error(
                f"Failed to release stock for order {order_id}: {exc}"
            )

    async def _fail_order(self, order_id: int, reason: str) -> None:
        """
        주문을 실패 상태로 변경

        Args:
            order_id: 실패 처리할 주문 ID
            reason: 실패 사유
        """
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if order:
                order.order_status = 9  # FAIL_PACK
                order.end_time = datetime.now(timezone.utc)
                order.failure_reason = reason
                session.commit()
                logger.info(f"Order {order_id} marked as failed: {reason}")

        # 로봇 예약 해제
        await self._release_pickee(order_id)
        await self._release_packee(order_id)

        # 재고 복구
        await self._release_stock_for_order(order_id)

        # 사용자 알림
        await self._push_to_user(
            {
                "type": "order_failed_notification",
                "result": False,
                "error_code": "ORDER_002",
                "data": {"order_id": order_id, "reason": reason},
                "message": "주문이 처리되지 못했습니다. 재고가 복구되었습니다.",
            },
            order_id=order_id,
        )

        self._clear_order_user(order_id)

    async def handle_robot_failure(self, event_data: Dict[str, Any]) -> None:
        """
        로봇 장애 이벤트를 처리하여 자동으로 다른 로봇을 할당합니다.

        Args:
            event_data: {
                "robot_id": int,
                "robot_type": str,  # "pickee" or "packee"
                "status": str,      # "ERROR" or "OFFLINE"
                "active_order_id": Optional[int]
            }
        """
        robot_id = event_data.get("robot_id")
        robot_type_str = event_data.get("robot_type")
        status = event_data.get("status")
        active_order_id = event_data.get("active_order_id")

        logger.warning(
            "Handling robot failure: robot_id=%d, type=%s, status=%s, order_id=%s",
            robot_id,
            robot_type_str,
            status,
            active_order_id,
        )

        # 활성 주문이 없으면 할당할 필요 없음
        if not active_order_id:
            logger.info("No active order for failed robot %d, skipping reallocation", robot_id)
            return

        # 로봇 타입 확인
        try:
            robot_type = RobotType.PICKEE if robot_type_str == "pickee" else RobotType.PACKEE
        except Exception:  # noqa: BLE001
            logger.error("Invalid robot type: %s", robot_type_str)
            return

        # 기존 예약 해제 및 모니터 취소
        if robot_type == RobotType.PICKEE:
            self._pickee_assignments.pop(active_order_id, None)
        else:
            self._packee_assignments.pop(active_order_id, None)
        self._cancel_reservation_monitor(active_order_id, robot_id)

        if not self._allocator:
            logger.warning("No allocator available for robot reallocation")
            await self._notify_reassignment_failure(
                active_order_id,
                robot_id,
                robot_type_str,
                status,
                reason="allocator_unavailable",
            )
            return

        context = AllocationContext(order_id=active_order_id, required_type=robot_type)
        new_robot_state = await self._allocator.reserve_robot(context)

        if not new_robot_state:
            logger.error(
                "Failed to reallocate order %d: no available %s robot",
                active_order_id,
                robot_type_str,
            )
            # 재할당 실패 → 재고 복구 및 주문 실패 처리
            await self._fail_order(active_order_id, f"No available {robot_type_str} robot after failure")

            await self._notify_reassignment_failure(
                active_order_id,
                robot_id,
                robot_type_str,
                status,
                reason="no_available_robot",
            )
            return

        new_robot_id = new_robot_state.robot_id
        logger.info(
            "Attempting to reassign order %d from robot %d to robot %d",
            active_order_id,
            robot_id,
            new_robot_id,
        )

        try:
            if robot_type == RobotType.PICKEE:
                success = await self._reassign_pickee(active_order_id, new_robot_id)
            else:
                success = await self._reassign_packee(active_order_id, new_robot_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error occurred while reassigning order %d to robot %d: %s",
                active_order_id,
                new_robot_id,
                exc,
            )
            success = False

        if success:
            await self._notify_reassignment_success(
                active_order_id,
                robot_id,
                new_robot_id,
                robot_type_str,
                status or "",
            )
        else:
            await self._allocator.release_robot(new_robot_id, active_order_id)

            # 재할당 실패 → 재고 복구 및 주문 실패 처리
            await self._fail_order(active_order_id, f"Reassignment to robot {new_robot_id} failed")

            await self._notify_reassignment_failure(
                active_order_id,
                robot_id,
                robot_type_str,
                status,
                reason="reassignment_failed",
            )
