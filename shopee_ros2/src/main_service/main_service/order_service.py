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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

from sqlalchemy import func

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
from shopee_interfaces.msg import ProductLocation, ProductInfo

from .config import settings
from .database_models import Box, Customer, Order, OrderItem, Product, RobotHistory
from .event_bus import EventBus
from .robot_selector import AllocationContext, RobotAllocator
from .robot_state_store import RobotStateStore
from .constants import OrderStatus, RobotType, RobotStatus
from .order_states import OrderStateManager
from .order_notifier import OrderNotifier
from .failure_handler import RobotFailureHandler
from .assignment_tracker import RobotAssignmentManager
from .location_builder import ProductLocationBuilder

if TYPE_CHECKING:
    from shopee_interfaces.msg import (
        PackeeAvailability,
        PackeePackingComplete,
        PickeeArrival,
        PickeeCartHandover,
        PickeeMoveStatus,
        PickeeProductDetection,
        PickeeProductLoaded,
        PickeeProductSelection,
    )
    from .database_manager import DatabaseManager
    from .inventory_service import InventoryService
    from .robot_coordinator import RobotCoordinator
    from .robot_state_backend import RobotState

logger = logging.getLogger(__name__)


class OrderService:
    """
    주문 서비스 (오케스트레이터)
    
    주문의 전체 생명주기를 관리하는 핵심 서비스입니다.
    세부 로직을 OrderStateManager, OrderNotifier, RobotFailureHandler에 위임합니다.
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

        # Refactored components
        self._state_manager = OrderStateManager(db)
        self._notifier = OrderNotifier(event_bus, db, self._state_manager)
        self._assignment_manager = RobotAssignmentManager()
        self._product_builder = ProductLocationBuilder(db)
        
        # Local state
        self._detected_product_bbox: Dict[int, Dict[int, int]] = {}
        self._order_section_queue: Dict[int, List[Dict[str, Any]]] = {} # 주문별 방문 구간 목록 {order_id: [{shelf_id, location_id, section_id, is_manual}, ...]}
        self._section_item_count: Dict[int, int] = {} # 주문별 현재 섹션에서 피킹할 상품 종류 수
        self._current_section_info: Dict[int, Dict[str, Any]] = {} # 주문별 현재 작업중인 구간 {order_id: {shelf_id, location_id, section_id, is_manual}}
        self._pending_detection_sections: Dict[int, set[int]] = {} # 주문별 감지 요청 진행 중인 섹션
        self._processed_cart_handover: set[int] = set() # 중복 handover 처리 방지
        self._last_move_location: Dict[int, int] = {}
        self._last_arrival_signature: Dict[int, tuple[int, int]] = {}
        self._last_detection_signature: Dict[int, tuple[int, ...]] = {}
        self._pending_pack_tasks: Dict[int, Dict[str, Any]] = {}
        self._manual_selection_candidates: Dict[int, List[Dict[str, Any]]] = {}
        self._packee_dimension_scale = settings.PACKEE_PRODUCT_DIMENSION_SCALE if settings.PACKEE_PRODUCT_DIMENSION_SCALE > 0 else 1.0

        # Failure handler (순환 참조 제거됨)
        self._failure_handler = RobotFailureHandler(
            db, event_bus, self._state_manager, self._notifier, allocator, 
            state_store, robot_coordinator, inventory_service,
            self._assignment_manager, self._product_builder
        )

    def _to_non_negative_int(self, value: Optional[object], scale: float = 1.0) -> int:
        """Packee 메시지 규격에 맞춰 단위를 보정하고 음수 값을 제거한다."""
        if value is None:
            return 0
        try:
            numeric = float(value) * scale
        except (TypeError, ValueError):
            return 0
        if numeric <= 0:
            return 0
        return int(round(numeric))

    def _create_packee_product_detail(self, product: Product, quantity: int) -> Dict[str, Any]:
        """PackeePackingStart 요청에 사용할 상품 정보를 정수 필드로 변환한다."""
        return {
            "product_id": int(product.product_id),
            "quantity": max(int(quantity), 0),
            # 상품 크기는 설정된 배율을 적용하여 mm 단위로 정규화한다.
            "length": self._to_non_negative_int(product.length, scale=self._packee_dimension_scale),
            "width": self._to_non_negative_int(product.width, scale=self._packee_dimension_scale),
            "height": self._to_non_negative_int(product.height, scale=self._packee_dimension_scale),
            "weight": self._to_non_negative_int(product.weight),
            "fragile": bool(product.fragile),
        }

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

    def _load_products(self, session, product_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """상품 ID 목록을 받아 필요한 메타데이터를 매핑으로 반환"""
        if not product_ids:
            return {}
        products = (
            session.query(
                Product.product_id,
                Product.name,
                Product.auto_select,
            )
            .filter(Product.product_id.in_(product_ids))
            .all()
        )
        product_map: Dict[int, Dict[str, Any]] = {}
        for product in products:
            product_map[product.product_id] = {
                "name": product.name,
                "auto_select": bool(product.auto_select),
            }
        return product_map


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

                if self._inventory_service:
                    for item in items:
                        product_id = item["product_id"]
                        quantity = item["quantity"]
                        stock_ok = await self._inventory_service.check_and_reserve_stock(product_id, quantity)
                        if not stock_ok:
                            for reserved_pid, reserved_qty in reserved_stocks:
                                await self._inventory_service.release_stock(reserved_pid, reserved_qty)
                            raise ValueError(f"Insufficient stock for product {product_id}.")
                        reserved_stocks.append((product_id, quantity))
                    logger.info("Stock reservation successful for order")

                new_order = Order(customer_id=customer.customer_id, start_time=datetime.now(), order_status=1)
                session.add(new_order)
                session.flush()
                order_id = new_order.order_id

                self._notifier.register_order_user(order_id, customer.id)

                if self._allocator:
                    context = AllocationContext(order_id=order_id, required_type=RobotType.PICKEE)
                    reserved_state = await self._allocator.reserve_robot(context)
                    if not reserved_state:
                        raise RuntimeError("No available Pickee robot.")
                    robot_id = reserved_state.robot_id
                    reserved_robot_id = robot_id
                else:
                    robot_id = 1

                for item in items:
                    new_item = OrderItem(order_id=new_order.order_id, product_id=item["product_id"], quantity=item["quantity"])
                    session.add(new_item)

                # OrderItem을 데이터베이스에 기록하여 build_product_locations에서 조회할 수 있도록 함
                session.flush()

                product_locations = self._product_builder.build_product_locations(session, order_id)
                product_ids_for_auto = [pl.product_id for pl in product_locations]
                auto_select_map: Dict[int, bool] = {}
                if product_ids_for_auto:
                    products = (
                        session.query(Product)
                        .filter(Product.product_id.in_(product_ids_for_auto))
                        .all()
                    )
                    auto_select_map = {prod.product_id: bool(prod.auto_select) for prod in products}

                request = PickeeWorkflowStartTask.Request(robot_id=robot_id, order_id=new_order.order_id, user_id=user_id, product_list=product_locations)
                response = await self._robot.dispatch_pick_task(request)
                if not response.success:
                    raise RuntimeError(f"Failed to dispatch pick task: {response.message}")

                section_plan = self._product_builder.build_section_plan(session, order_id)
                if section_plan:
                    section_manual_map: Dict[Tuple[int, int], bool] = {}
                    for location in product_locations:
                        key = (location.location_id, location.section_id)
                        is_manual_item = not auto_select_map.get(location.product_id, True)
                        if is_manual_item:
                            section_manual_map[key] = True
                        elif key not in section_manual_map:
                            section_manual_map[key] = False

                    sorted_plan = sorted(
                        section_plan,
                        key=lambda entry: (entry["shelf_id"], entry["section_id"]),
                    )
                    planned_sections = [
                        {
                            "shelf_id": entry["shelf_id"],
                            "location_id": entry["location_id"],
                            "section_id": entry["section_id"],
                            "is_manual": section_manual_map.get((entry["location_id"], entry["section_id"]), False),
                        }
                        for entry in sorted_plan
                    ]
                    self._order_section_queue[new_order.order_id] = planned_sections
                    logger.info("Order %d section queue (shelf->section): %s", new_order.order_id, planned_sections)

                    first_section = planned_sections[0]
                    self._current_section_info[new_order.order_id] = dict(first_section)
                    try:
                        move_req = PickeeWorkflowMoveToSection.Request(
                            robot_id=robot_id,
                            order_id=new_order.order_id,
                            location_id=first_section["location_id"],
                            section_id=first_section["section_id"],
                        )
                        await self._robot.dispatch_move_to_section(move_req)
                    except Exception as move_exc:
                        logger.error("Failed to dispatch move_to_section for order %d: %s", new_order.order_id, move_exc)
                        if self._allocator and reserved_robot_id is not None:
                            await self._allocator.release_robot(reserved_robot_id, new_order.order_id)
                            self._assignment_manager.release_pickee(new_order.order_id)
                        raise

                session.commit()
                logger.info("Order %d created and dispatched to robot %d", new_order.order_id, robot_id)
                if self._allocator and reserved_robot_id is not None:
                    self._assignment_manager.assign_pickee(new_order.order_id, reserved_robot_id)
                    self._failure_handler.start_reservation_monitor(reserved_robot_id, new_order.order_id, RobotType.PICKEE)
                return new_order.order_id, robot_id

            except Exception as e:
                logger.exception("Order creation failed: %s", e)
                session.rollback()
                if self._inventory_service:
                    for reserved_pid, reserved_qty in reserved_stocks:
                        try:
                            await self._inventory_service.release_stock(reserved_pid, reserved_qty)
                        except Exception as release_exc:
                            logger.error(f"Failed to release stock for product {reserved_pid}: {release_exc}")
                if self._allocator and reserved_robot_id is not None and order_id is not None:
                    await self._allocator.release_robot(reserved_robot_id, order_id)
                    self._assignment_manager.release_pickee(order_id)
                if order_id is not None:
                    self._notifier.clear_order_user(order_id)
                return None

    async def select_product(self, order_id: int, robot_id: int, bbox_number: int, product_id: int) -> bool:
        logger.info("Dispatching product selection: Order=%d, Robot=%d, BBox=%d, Product=%d", order_id, robot_id, bbox_number, product_id)
        try:
            request = PickeeProductProcessSelection.Request(robot_id=robot_id, order_id=order_id, product_id=product_id, bbox_number=bbox_number)
            response = await self._robot.dispatch_pick_process(request)
            logger.info(
                "Product selection response received: order=%d product=%d success=%s message=%s",
                order_id,
                product_id,
                getattr(response, "success", None),
                getattr(response, "message", ""),
            )
            return response.success
        except Exception as e:
            logger.exception("Failed to dispatch product selection: %s", e)
            return False

    async def _resolve_pickee_robot_id(self, order_id: int) -> Optional[int]:
        """주문에 할당된 Pickee 로봇 ID를 조회합니다."""
        robot_id = self._assignment_manager.get_pickee(order_id)
        if robot_id is not None:
            return robot_id

        robot_id = self._assignment_manager.get_last_pickee(order_id)
        if robot_id is not None:
            return robot_id

        if self._state_store:
            try:
                states = await self._state_store.list_states(RobotType.PICKEE)
                for state in states:
                    if state.active_order_id == order_id:
                        return state.robot_id
            except Exception as exc:
                logger.warning("Failed to resolve pickee robot from state store for order %d: %s", order_id, exc)

        return None

    async def end_shopping(self, order_id: int, robot_id: Optional[int] = None) -> Tuple[bool, Optional[Dict[str, int]]]:
        logger.info("Ending shopping for order %d", order_id)
        try:
            resolved_robot_id = robot_id
            if resolved_robot_id is None:
                resolved_robot_id = await self._resolve_pickee_robot_id(order_id)
            if resolved_robot_id is None:
                logger.error("Cannot end shopping for order %d: no Pickee robot assignment found.", order_id)
                return False, None

            robot_id_value = int(resolved_robot_id)
            request = PickeeWorkflowEndShopping.Request(robot_id=robot_id_value, order_id=order_id)
            response = await self._robot.dispatch_shopping_end(request)
            if not response.success:
                logger.error("Robot failed to end shopping for order %d: %s", order_id, response.message)
                return False, None

            summary: Optional[Dict[str, int]] = None
            if self._state_manager.set_status_picked_up(order_id):
                with self._db.session_scope() as session:
                    total_items, total_price = self._calculate_order_summary(session, order_id)
                    summary = {"total_items": total_items, "total_price": total_price}
            else:
                return False, None

            packaging_location_id = settings.PICKEE_PACKING_LOCATION_ID
            if packaging_location_id:
                try:
                    move_pack_req = PickeeWorkflowMoveToPackaging.Request(robot_id=robot_id_value, order_id=order_id, location_id=packaging_location_id)
                    await self._robot.dispatch_move_to_packaging(move_pack_req)
                except Exception as pack_exc:
                    logger.error("Failed to dispatch move_to_packaging for order %d: %s", order_id, pack_exc)
                    await self._notifier._push_to_user({"type": "robot_command_failed", "result": False, "error_code": "ROBOT_002", "data": {"order_id": order_id, "robot_id": robot_id_value, "command": "move_to_packaging"}, "message": "로봇 이동 명령이 실패했습니다. 직원이 확인 중입니다."}, order_id=order_id)

            return True, summary
        except Exception as e:
            logger.exception("Failed to end shopping for order %d: %s", order_id, e)
            return False, None

    async def handle_moving_status(self, msg: "PickeeMoveStatus") -> None:
        """Pickee 이동 상태 처리"""
        last_location = self._last_move_location.get(msg.order_id)
        if last_location == msg.location_id:
            logger.debug("Ignoring duplicate moving status for order %d (location %s)", msg.order_id, msg.location_id)
            return
        self._last_move_location[msg.order_id] = msg.location_id
        logger.info("Handling moving status for order %d (location %s)", msg.order_id, msg.location_id)
        self._failure_handler.cancel_reservation_monitor(msg.order_id, msg.robot_id)
        await self._notifier.notify_robot_moving(msg)

    async def handle_arrival_notice(self, msg: "PickeeArrival") -> None:
        """Pickee 도착 알림 처리"""
        is_section = msg.section_id is not None and msg.section_id >= 0
        signature = (msg.location_id, msg.section_id if is_section else -1)
        if self._last_arrival_signature.get(msg.order_id) == signature:
            logger.debug(
                "Ignoring duplicate arrival notice for order %d at location %s section %s",
                msg.order_id,
                msg.location_id,
                msg.section_id if is_section else 'N/A',
            )
            return
        self._last_arrival_signature[msg.order_id] = signature
        logger.info("Handling arrival notice for order %d at section %s", msg.order_id, msg.section_id if is_section else 'N/A')
        await self._notifier.notify_robot_arrived(msg)
        if not is_section:
            logger.debug("Skipping product detection for non-section arrival.")
            return
        try:
            with self._db.session_scope() as session:
                # 현재 섹션에 있는 상품들만 조회
                products_in_section = (
                    session.query(OrderItem.product_id)
                    .join(Product, OrderItem.product_id == Product.product_id)
                    .filter(OrderItem.order_id == msg.order_id)
                    .filter(Product.section_id == msg.section_id)
                    .all()
                )
                product_ids = [pid for pid, in products_in_section]

            # 현재 섹션에서 피킹할 상품 종류 수 저장
            self._section_item_count[msg.order_id] = len(product_ids)
            logger.info(f"Order {msg.order_id} has {len(product_ids)} item types to pick in section {msg.section_id}")

            if not product_ids:
                logger.warning("No products to detect for order %d in section %d. Moving to next.", msg.order_id, msg.section_id)
                # 이 섹션에 상품이 없으면 바로 다음 단계로 진행
                await self._move_to_next_or_end(msg.order_id, msg.robot_id)
                return

            pending_sections = self._pending_detection_sections.setdefault(msg.order_id, set())
            if msg.section_id in pending_sections:
                logger.debug("Detection already pending for order %d section %d. Skipping duplicate arrival.", msg.order_id, msg.section_id)
                return
            pending_sections.add(msg.section_id)
            logger.info(
                "Order %d pending detection sections: %s",
                msg.order_id,
                pending_sections,
            )

            req = PickeeProductDetect.Request(robot_id=msg.robot_id, order_id=msg.order_id, product_ids=product_ids)
            await self._robot.dispatch_product_detect(req)
        except Exception as e:
            logger.exception("Failed to start product detection for order %d: %s", msg.order_id, e)

    async def handle_product_detected(self, msg: "PickeeProductDetection") -> None:
        """Pickee 상품 인식 완료 처리. 수동/자동 피킹 분기."""
        product_signature = tuple(sorted((p.product_id, p.bbox_number) for p in msg.products))
        if self._last_detection_signature.get(msg.order_id) == product_signature:
            logger.debug("Ignoring duplicate product detection for order %d.", msg.order_id)
            return
        self._last_detection_signature[msg.order_id] = product_signature
        logger.info("Handling product detection for order %d. Found %d products.", msg.order_id, len(msg.products))
        
        current_section = self._current_section_info.get(msg.order_id)
        if not current_section:
            logger.debug("Stale product detection received for order %d. No current section info; ignoring.", msg.order_id)
            return

        # 공통: 인식된 상품 BBox 정보 저장
        self._detected_product_bbox[msg.order_id] = {p.product_id: p.bbox_number for p in msg.products}
        logger.info(
            "Order %d detected product BBox map: %s",
            msg.order_id,
            self._detected_product_bbox[msg.order_id],
        )

        if msg.order_id in self._manual_selection_candidates:
            logger.info("Order %d clearing stale manual candidates before update.", msg.order_id)
        self._manual_selection_candidates.pop(msg.order_id, None)

        product_ids = [p.product_id for p in msg.products]
        product_map: Dict[int, Dict[str, Any]] = {}
        if product_ids:
            with self._db.session_scope() as session:
                logger.info(
                    "Order %d loading product metadata for detected ids: %s",
                    msg.order_id,
                    product_ids,
                )
                product_map = self._load_products(session, product_ids)
                logger.info(
                    "Order %d loaded product metadata keys: %s",
                    msg.order_id,
                    list(product_map.keys()),
                )
        else:
            logger.warning("Order %d product detection returned empty id list.", msg.order_id)

        manual_candidates: List[Dict[str, Any]] = []
        auto_targets: List[Any] = []
        for detected in msg.products:
            product_meta = product_map.get(detected.product_id, {})
            product_name = product_meta.get("name", "")
            auto_select_flag = bool(product_meta.get("auto_select", True))
            logger.info(
                "Order %d detection item evaluated: product_id=%s auto_select=%s manual_section_hint=%s",
                msg.order_id,
                detected.product_id,
                auto_select_flag,
                product_name,
            )
            if auto_select_flag:
                auto_targets.append(detected)
            else:
                manual_candidates.append(
                    {
                        "product_id": detected.product_id,
                        "name": product_name,
                        "bbox_number": detected.bbox_number,
                    }
                )

        logger.info(
            "Order %d detection classification: auto=%d, manual=%d, shelf=%s, section=%s",
            msg.order_id,
            len(auto_targets),
            len(manual_candidates),
            current_section.get("shelf_id"),
            current_section.get("section_id"),
        )

        is_manual_section = current_section.get("is_manual", False)
        if is_manual_section or manual_candidates:
            logger.info(
                "Order %d manual picking required. manual_section=%s candidate_count=%d",
                msg.order_id,
                is_manual_section,
                len(manual_candidates),
            )
            if not manual_candidates:
                manual_candidates = []
                for detected in msg.products:
                    product_meta = product_map.get(detected.product_id, {})
                    manual_candidates.append(
                        {
                            "product_id": detected.product_id,
                            "name": product_meta.get("name", ""),
                            "bbox_number": detected.bbox_number,
                        }
                    )
            self._manual_selection_candidates[msg.order_id] = list(manual_candidates)
            logger.info(
                "Order %d manual candidate list prepared: %s",
                msg.order_id,
                [
                    {
                        "product_id": candidate.get("product_id"),
                        "bbox_number": candidate.get("bbox_number"),
                    }
                    for candidate in manual_candidates
                ],
            )
            await self._notifier.notify_product_selection_start(msg, manual_candidates)
            return

        if auto_targets:
            logger.info(
                "Order %d auto-selecting %d products in shelf %s section %s.",
                msg.order_id,
                len(auto_targets),
                current_section.get("shelf_id"),
                current_section.get("section_id"),
            )
            selection_tasks = [
                self.select_product(
                    order_id=msg.order_id,
                    robot_id=msg.robot_id,
                    bbox_number=detected.bbox_number,
                    product_id=detected.product_id,
                )
                for detected in auto_targets
            ]
            logger.info(
                "Order %d auto selection tasks prepared: %s",
                msg.order_id,
                [
                    {
                        "product_id": detected.product_id,
                        "bbox_number": detected.bbox_number,
                    }
                    for detected in auto_targets
                ],
            )
            try:
                selection_results = await asyncio.gather(*selection_tasks)
            except Exception as exc:
                logger.exception(
                    "Order %d auto selection tasks raised an exception: %s",
                    msg.order_id,
                    exc,
                )
                raise
            success_count = sum(1 for result in selection_results if result)
            failure_count = len(selection_results) - success_count
            logger.info(
                "Order %d auto selection completed. success=%d failure=%d",
                msg.order_id,
                success_count,
                failure_count,
            )
        else:
            logger.warning(
                "Auto section for order %d but no products to select. Moving to next.",
                msg.order_id,
            )
            await self._move_to_next_or_end(msg.order_id, msg.robot_id)

    async def _move_to_next_or_end(self, order_id: int, robot_id: int):
        """다음 섹션으로 이동하거나, 모든 섹션 방문 시 쇼핑을 종료합니다."""
        
        # 현재 섹션 정보 가져오기
        completed_section = self._current_section_info.get(order_id)
        logger.info(
            "Order %d move_to_next_or_end invoked. completed_section=%s",
            order_id,
            completed_section,
        )
        
        # 다음 방문할 섹션 확인 (큐는 전체 계획이므로 이미 방문한 섹션을 제거)
        queue = self._order_section_queue.get(order_id)
        logger.info(
            "Order %d current section queue snapshot (before pop): %s",
            order_id,
            queue,
        )
        if completed_section and queue:
            first_entry = queue[0]
            if (
                first_entry.get('section_id') == completed_section.get('section_id') and
                first_entry.get('location_id') == completed_section.get('location_id') and
                first_entry.get('shelf_id') == completed_section.get('shelf_id')
            ):
                queue.pop(0)
                logger.info(
                    "Order %d removed completed section from queue. remaining=%d",
                    order_id,
                    len(queue),
                )

        if completed_section:
            pending_sections = self._pending_detection_sections.get(order_id)
            if pending_sections is not None:
                pending_sections.discard(completed_section.get('section_id'))
                if not pending_sections:
                    self._pending_detection_sections.pop(order_id, None)
                logger.info(
                    "Order %d pending detection sections updated: %s",
                    order_id,
                    pending_sections,
                )

        next_section_info = queue[0] if queue else None

        if completed_section and completed_section.get('is_manual') and next_section_info and not next_section_info.get('is_manual'):
            logger.info(f"Order {order_id}: Manual picking complete. Notifying user.")
            await self._notifier.notify_manual_picking_complete(order_id)

        if queue:
            next_section = queue.pop(0)
            self._current_section_info[order_id] = dict(next_section) # 현재 섹션 정보 업데이트
            self._last_detection_signature.pop(order_id, None)
            logger.info(
                "Moving to shelf %s section %s for order %d",
                next_section.get('shelf_id'),
                next_section.get('section_id'),
                order_id,
            )
            move_req = PickeeWorkflowMoveToSection.Request(
                robot_id=robot_id,
                order_id=order_id,
                location_id=next_section['location_id'],
                section_id=next_section['section_id']
            )
            logger.info(
                "Order %d dispatching move_to_section -> location=%s section=%s queue_after_dispatch=%d",
                order_id,
                next_section.get('location_id'),
                next_section.get('section_id'),
                len(queue),
            )
            await self._robot.dispatch_move_to_section(move_req)
        else:
            logger.info(f"All sections visited for order {order_id}. Ending shopping.")
            # 큐와 상태 변수 정리
            self._order_section_queue.pop(order_id, None)
            self._section_item_count.pop(order_id, None)
            self._current_section_info.pop(order_id, None)
            self._pending_detection_sections.pop(order_id, None)
            self._last_detection_signature.pop(order_id, None)
            self._last_arrival_signature.pop(order_id, None)
            self._last_move_location.pop(order_id, None)
            self._manual_selection_candidates.pop(order_id, None)
            logger.info(
                "Order %d all section data cleared. Triggering end_shopping.",
                order_id,
            )
            
            # 피킹 완료 알림 전송
            await self._notifier.notify_picking_complete(order_id, robot_id)
            
            await self.end_shopping(order_id, robot_id)

    async def handle_pickee_selection(self, msg: "PickeeProductSelection") -> None:
        """Pickee 상품 선택 결과 처리 및 다음 섹션 이동 처리"""
        logger.info("Handling pickee selection result for order %d", msg.order_id)
        
        # 사용자에게 카트 업데이트 알림
        summary = {}
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=msg.product_id).first()
            total_items, total_price = self._calculate_order_summary(session, msg.order_id)
            summary = {"product": {"product_id": msg.product_id, "name": product.name if product else "", "quantity": msg.quantity, "price": product.price if product else 0}, "total_items": total_items, "total_price": total_price}
        await self._notifier.notify_cart_update(msg, summary)

        # BBox 맵 정리
        bbox_map = self._detected_product_bbox.get(msg.order_id)
        if bbox_map and msg.product_id in bbox_map:
            bbox_map.pop(msg.product_id, None)
            if not bbox_map:
                self._detected_product_bbox.pop(msg.order_id, None)

        candidates = self._manual_selection_candidates.get(msg.order_id)
        if candidates:
            updated_candidates = [
                candidate for candidate in candidates if candidate.get("product_id") != msg.product_id
            ]
            logger.info(
                "Order %d manual candidates updated after selection. before=%d after=%d",
                msg.order_id,
                len(candidates),
                len(updated_candidates),
            )
            if updated_candidates:
                self._manual_selection_candidates[msg.order_id] = updated_candidates
                prompt_msg = SimpleNamespace(order_id=msg.order_id, robot_id=msg.robot_id)
                await self._notifier.notify_product_selection_start(prompt_msg, updated_candidates)
            else:
                self._manual_selection_candidates.pop(msg.order_id, None)
                logger.info(
                    "Order %d manual candidate list cleared after selection.",
                    msg.order_id,
                )
        # 현재 섹션의 모든 상품을 담았는지 확인 후 다음 단계 진행
        if msg.order_id in self._section_item_count:
            self._section_item_count[msg.order_id] -= 1
            logger.info(
                "Order %d section remaining item types: %d",
                msg.order_id,
                self._section_item_count[msg.order_id],
            )
            if self._section_item_count[msg.order_id] <= 0:
                logger.info(f"Section complete for order {msg.order_id}. Moving to next step.")
                self._manual_selection_candidates.pop(msg.order_id, None)
                await self._move_to_next_or_end(msg.order_id, msg.robot_id)

    async def handle_cart_handover(self, msg: "PickeeCartHandover") -> None:
        """Pickee 장바구니 전달 완료 처리"""
        order_id = msg.order_id
        robot_id = msg.robot_id
        if order_id in self._processed_cart_handover:
            logger.debug("Cart handover already processed for order %d. Skipping duplicate message.", order_id)
            return
        self._processed_cart_handover.add(order_id)
        logger.info("Handling cart handover for order %d, starting packing process.", order_id)

        packee_robot_id: Optional[int] = None
        try:
            if self._allocator:
                context = AllocationContext(order_id=order_id, required_type=RobotType.PACKEE)
                max_attempts = 3
                retry_delay = 0.5
                for attempt in range(1, max_attempts + 1):
                    reserved_state = await self._allocator.reserve_robot(context)
                    if reserved_state:
                        packee_robot_id = reserved_state.robot_id
                        break
                    logger.warning(
                        "Packee allocation attempt %d/%d failed for order %d. Retrying in %.1fs.",
                        attempt,
                        max_attempts,
                        order_id,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)

            if packee_robot_id is None:
                logger.error("No available Packee robot for order %d.", order_id)
                await self._failure_handler.fail_order(order_id, "No available Packee robot.")
                return

            product_details_for_packee: List[Dict[str, Any]] = []
            with self._db.session_scope() as session:
                items_with_products = (
                    session.query(OrderItem, Product)
                    .join(Product, OrderItem.product_id == Product.product_id)
                    .filter(OrderItem.order_id == order_id)
                    .all()
                )
                for order_item, product in items_with_products:
                    product_details_for_packee.append(
                        self._create_packee_product_detail(product, order_item.quantity)
                    )

            self._assignment_manager.assign_packee(order_id, packee_robot_id)
            self._pending_pack_tasks[order_id] = {
                "packee_robot_id": packee_robot_id,
                "product_details": product_details_for_packee,
            }

            check_req = PackeePackingCheckAvailability.Request(robot_id=packee_robot_id, order_id=order_id)
            check_res = await self._robot.check_packee_availability(check_req)
            if not check_res.success:
                raise RuntimeError(f"Packee availability check failed: {check_res.message}")

            await self._notifier.notify_packing_info(
                order_id=order_id,
                payload={"robot_id": packee_robot_id, "status": "checking_availability"},
            )

            home_location_id = settings.PICKEE_HOME_LOCATION_ID
            if home_location_id:
                return_req = PickeeWorkflowReturnToBase.Request(robot_id=robot_id, location_id=home_location_id)
                await self._robot.dispatch_return_to_base(return_req)

        except Exception as e:
            logger.exception("Failed to handle cart handover for order %d: %s", order_id, e)
            self._pending_pack_tasks.pop(order_id, None)
            if packee_robot_id is not None:
                self._assignment_manager.release_packee(order_id)
                if self._allocator:
                    await self._allocator.release_robot(packee_robot_id, order_id)
            await self._failure_handler.fail_order(order_id, "Failed during cart handover.")
        finally:
            await self._failure_handler._release_pickee(order_id)
            if packee_robot_id is None and order_id in self._processed_cart_handover:
                self._processed_cart_handover.discard(order_id)

    async def handle_product_loaded(self, msg: "PickeeProductLoaded") -> None:
        """창고 물품 적재 완료 이벤트 처리"""
        order_id = self._assignment_manager.get_pickee_order(msg.robot_id)
        if order_id is None:
            order_id = self._assignment_manager.get_last_order_for_pickee(msg.robot_id)
        if order_id is None:
            logger.warning(
                "Received product loaded event from robot %d but no active order mapping was found.",
                msg.robot_id,
            )
            return

        product_summary: Dict[str, Any] = {
            "product_id": msg.product_id,
            "quantity": msg.quantity,
        }
        total_items = 0
        total_price = 0
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=msg.product_id).first()
            if product:
                product_summary["name"] = product.name
                product_summary["price"] = product.price
            total_items, total_price = self._calculate_order_summary(session, order_id)

        await self._notifier.notify_product_loaded(
            order_id=order_id,
            robot_id=msg.robot_id,
            product=product_summary,
            total_items=total_items,
            total_price=total_price,
            success=msg.success,
            message=msg.message or "",
        )
        await self._notifier.emit_work_info_notification(order_id=order_id, robot_id=msg.robot_id)

    async def handle_packee_availability(self, msg: "PackeeAvailability") -> None:
        """Packee 작업 가능 여부 확인 처리"""
        order_id = msg.order_id
        logger.info(
            "Packee availability for order %d: available=%s, cart_detected=%s",
            order_id,
            msg.available,
            msg.cart_detected,
        )

        context = self._pending_pack_tasks.get(order_id)
        if not context:
            logger.warning("No pending pack task found for order %d. Ignoring availability result.", order_id)
            await self._notifier.emit_work_info_notification(order_id=order_id, robot_id=msg.robot_id)
            return

        expected_robot_id = context.get("packee_robot_id")
        if expected_robot_id != msg.robot_id:
            logger.warning(
                "Availability result robot mismatch for order %d: expected %s, got %s",
                order_id,
                expected_robot_id,
                msg.robot_id,
            )

        if not msg.available or not msg.cart_detected:
            reason = "Packee unavailable"
            if not msg.cart_detected:
                reason = "Packing cart not detected"
            await self._notifier.notify_packing_unavailable(order_id, msg.robot_id, reason, msg.message)
            self._pending_pack_tasks.pop(order_id, None)
            released_robot = self._assignment_manager.release_packee(order_id)
            if released_robot is not None and self._allocator:
                await self._allocator.release_robot(released_robot, order_id)
            self._processed_cart_handover.discard(order_id)
            await self._notifier.emit_work_info_notification(order_id=order_id, robot_id=msg.robot_id)
            return

        try:
            start_req = PackeePackingStart.Request(robot_id=msg.robot_id, order_id=order_id)
            start_req.products = [ProductInfo(**detail) for detail in context.get("product_details", [])]
            start_res = await self._robot.dispatch_pack_task(start_req)
            if not start_res.success:
                raise RuntimeError(f"Failed to start packing: {start_res.message}")

            if hasattr(start_res, "box_id") and start_res.box_id > 0:
                with self._db.session_scope() as session:
                    order = session.query(Order).filter_by(order_id=order_id).first()
                    if order:
                        order.box_id = start_res.box_id

            await self._notifier.notify_packing_info(
                order_id=order_id,
                payload={"robot_id": msg.robot_id, "status": "packing_started"},
            )
            self._failure_handler.start_reservation_monitor(msg.robot_id, order_id, RobotType.PACKEE)
        except Exception as exc:
            logger.exception("Failed to start packing for order %d: %s", order_id, exc)
            released_robot = self._assignment_manager.release_packee(order_id)
            if released_robot is not None and self._allocator:
                await self._allocator.release_robot(released_robot, order_id)
            await self._failure_handler.fail_order(order_id, "Failed to start packing.")
            self._processed_cart_handover.discard(order_id)
        finally:
            self._pending_pack_tasks.pop(order_id, None)
            await self._notifier.emit_work_info_notification(order_id=order_id, robot_id=msg.robot_id)

    async def handle_packee_complete(self, msg: "PackeePackingComplete") -> None:
        """Packee 포장 완료 처리"""
        logger.info("Handling packee complete for order %d. Success: %s", msg.order_id, msg.success)
        self._failure_handler.cancel_reservation_monitor(msg.order_id, msg.robot_id)

        if self._state_manager.set_status_packed(msg.order_id, msg.success, msg.message):
            product_info = {}
            with self._db.session_scope() as session:
                order_item = session.query(OrderItem).filter_by(order_id=msg.order_id).first()
                if order_item:
                    product = session.query(Product).filter_by(product_id=order_item.product_id).first()
                    product_info = {"product_id": order_item.product_id, "product_name": product.name if product else "", "product_price": product.price if product else 0, "product_quantity": order_item.quantity}
            await self._notifier.notify_packing_complete(msg, product_info)
            await self._notifier.emit_work_info_notification(order_id=msg.order_id, robot_id=msg.robot_id)

            self._detected_product_bbox.pop(msg.order_id, None)
            self._processed_cart_handover.discard(msg.order_id)
            await self._failure_handler.release_robot_for_order(msg.order_id)

            if not msg.success:
                await self._failure_handler._release_stock_for_order(msg.order_id)
            else:
                self._notifier.clear_order_user(msg.order_id)

    def get_detected_bbox(self, order_id: int, product_id: int) -> Optional[int]:
        return self._detected_product_bbox.get(order_id, {}).get(product_id)

    def list_detected_products(self, order_id: int) -> Dict[int, int]:
        return dict(self._detected_product_bbox.get(order_id, {}))

    async def get_active_orders_snapshot(self) -> Dict[str, Any]:
        # DB는 timezone-naive 로컬 시간으로 저장되므로, 비교도 로컬 시간으로 수행
        now = datetime.now()
        active_orders: List[Dict[str, Any]] = []
        status_counter: Dict[int, int] = {}
        truly_active_count = 0

        with self._db.session_scope() as session:
            recent_completed_cutoff = now - timedelta(minutes=30)
            orders = session.query(Order).filter((Order.order_status < 8) | ((Order.order_status >= 8) & (Order.end_time >= recent_completed_cutoff))).order_by(Order.start_time.desc()).all()

            for order in orders:
                total_items, total_price = self._calculate_order_summary(session, order.order_id)
                progress_value = self._state_manager.get_progress(order.order_status)

                # DB의 start_time과 현재 시간 모두 timezone-naive이므로 직접 비교
                # 완료된 주문은 end_time을 사용, 진행 중인 주문은 현재 시간을 사용
                elapsed_seconds = None
                if order.start_time:
                    if order.end_time:
                        elapsed_seconds = (order.end_time - order.start_time).total_seconds()
                    else:
                        elapsed_seconds = (now - order.start_time).total_seconds()

                active_orders.append({"order_id": order.order_id, "customer_id": order.customer.id if order.customer else None, "status_code": order.order_status, "status": self._state_manager.get_label(order.order_status), "progress": progress_value, "started_at": order.start_time.isoformat() if order.start_time else None, "elapsed_seconds": elapsed_seconds, "pickee_robot_id": self._assignment_manager.get_pickee(order.order_id), "packee_robot_id": self._assignment_manager.get_packee(order.order_id), "total_items": total_items, "total_price": total_price})
                status_counter[order.order_status] = status_counter.get(order.order_status, 0) + 1
                
                # 세션 안에서 truly_active_count 계산
                if order.order_status < 8:
                    truly_active_count += 1

        status_summary = {self._state_manager.get_label(code): count for code, count in status_counter.items()}

        return {"orders": active_orders, "summary": {"total_active": truly_active_count, "status_counts": status_summary}}

    async def get_recent_failed_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최근 실패한 주문 목록을 반환합니다.

        Args:
            limit: 조회할 최대 개수
        """
        with self._db.session_scope() as session:
            failed_orders = (
                session.query(Order)
                .filter(Order.order_status == 9)  # FAIL_PACK
                .order_by(Order.created_at.desc())
                .limit(limit)
                .all()
            )

            payload: List[Dict[str, Any]] = []
            for order in failed_orders:
                total_items, total_price = self._calculate_order_summary(session, order.order_id)
                payload.append(
                    {
                        'order_id': order.order_id,
                        'failure_reason': order.failure_reason,
                        'ended_at': order.end_time.isoformat() if order.end_time else None,
                        'total_items': total_items,
                        'total_price': total_price,
                    }
                )
            return payload

    async def get_failed_orders_by_reason(self, window_minutes: int = 60) -> Dict[str, int]:
        """
        최근 실패 주문을 사유별로 집계합니다.

        Args:
            window_minutes: 조회 대상 시간(분)
        """
        # DB는 timezone-naive이므로 datetime.now()를 사용
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        summary: Dict[str, int] = {}
        with self._db.session_scope() as session:
            rows = (
                session.query(Order.failure_reason, func.count(Order.order_id))
                .filter(Order.order_status == 9)
                .filter(Order.end_time.isnot(None))
                .filter(Order.end_time >= cutoff)
                .group_by(Order.failure_reason)
                .all()
            )
            for reason, count in rows:
                key = reason or 'UNKNOWN'
                summary[key] = int(count)
        return summary

    async def get_performance_metrics(
        self,
        *,
        window_minutes: int = 60,
        robot_states: Optional[List["RobotState"]] = None,
        orders_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        시스템 성능 메트릭스를 계산합니다.

        Args:
            window_minutes: 통계를 계산할 시간 범위(분)
            robot_states: 사전 조회한 로봇 상태 목록
            orders_snapshot: 사전 조회한 주문 스냅샷
        """
        # DB는 timezone-naive이므로 datetime.now()를 사용
        window_start = datetime.now() - timedelta(minutes=window_minutes)
        processing_times: List[float] = []
        completed_orders_count = 0

        with self._db.session_scope() as session:
            completed_orders = (
                session.query(Order)
                .filter(Order.end_time.isnot(None))
                .filter(Order.end_time >= window_start)
                .all()
            )
            for order in completed_orders:
                if order.start_time and order.end_time:
                    processing_times.append((order.end_time - order.start_time).total_seconds())
            completed_orders_count = len(completed_orders)
            total_orders = (
                session.query(Order)
                .filter(Order.start_time >= window_start)
                .count()
            )

        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        hourly_throughput = completed_orders_count
        success_rate = (completed_orders_count / total_orders * 100.0) if total_orders else 0.0

        snapshot = orders_snapshot or await self.get_active_orders_snapshot()
        active_orders = snapshot.get('summary', {}).get('total_active', 0)
        max_capacity = 200.0  # 설계 문서 기준 최대 세션 수
        system_load = min((active_orders / max_capacity) * 100.0, 100.0) if max_capacity > 0 else 0.0

        if robot_states is None and self._state_store:
            robot_states = await self._state_store.list_states()
        robot_states = robot_states or []
        total_robot_count = len(robot_states)
        working_robot_count = sum(
            1 for state in robot_states if state.status == RobotStatus.WORKING.value
        )
        robot_utilization = (
            working_robot_count / total_robot_count * 100.0 if total_robot_count else 0.0
        )

        return {
            'avg_processing_time': round(avg_processing_time, 1),
            'hourly_throughput': hourly_throughput,
            'success_rate': round(success_rate, 1),
            'robot_utilization': round(robot_utilization, 1),
            'system_load': round(system_load, 1),
            'active_orders': active_orders,
        }
