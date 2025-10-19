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

        # Failure handler (순환 참조 제거됨)
        self._failure_handler = RobotFailureHandler(
            db, event_bus, self._state_manager, self._notifier, allocator, 
            state_store, robot_coordinator, inventory_service,
            self._assignment_manager, self._product_builder
        )

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

                product_locations = self._product_builder.build_product_locations(session, order_id)
                
                for item in items:
                    new_item = OrderItem(order_id=new_order.order_id, product_id=item["product_id"], quantity=item["quantity"])
                    session.add(new_item)

                request = PickeeWorkflowStartTask.Request(robot_id=robot_id, order_id=new_order.order_id, user_id=user_id, product_list=product_locations)
                response = await self._robot.dispatch_pick_task(request)
                if not response.success:
                    raise RuntimeError(f"Failed to dispatch pick task: {response.message}")

                if product_locations:
                    first_location = product_locations[0]
                    try:
                        move_req = PickeeWorkflowMoveToSection.Request(
                            robot_id=robot_id, order_id=new_order.order_id, location_id=first_location.location_id, section_id=first_location.section_id
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
            return response.success
        except Exception as e:
            logger.exception("Failed to dispatch product selection: %s", e)
            return False

    async def end_shopping(self, order_id: int, robot_id: int) -> Tuple[bool, Optional[Dict[str, int]]]:
        logger.info("Ending shopping for order %d", order_id)
        try:
            request = PickeeWorkflowEndShopping.Request(robot_id=robot_id, order_id=order_id)
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
                    move_pack_req = PickeeWorkflowMoveToPackaging.Request(robot_id=robot_id, order_id=order_id, location_id=packaging_location_id)
                    await self._robot.dispatch_move_to_packaging(move_pack_req)
                except Exception as pack_exc:
                    logger.error("Failed to dispatch move_to_packaging for order %d: %s", order_id, pack_exc)
                    await self._notifier._push_to_user({"type": "robot_command_failed", "result": False, "error_code": "ROBOT_002", "data": {"order_id": order_id, "robot_id": robot_id, "command": "move_to_packaging"}, "message": "로봇 이동 명령이 실패했습니다. 직원이 확인 중입니다."}, order_id=order_id)

            return True, summary
        except Exception as e:
            logger.exception("Failed to end shopping for order %d: %s", order_id, e)
            return False, None

    async def handle_moving_status(self, msg: "PickeeMoveStatus") -> None:
        """Pickee 이동 상태 처리"""
        logger.info("Handling moving status for order %d", msg.order_id)
        self._failure_handler.cancel_reservation_monitor(msg.order_id, msg.robot_id)
        await self._notifier.notify_robot_moving(msg)

    async def handle_arrival_notice(self, msg: "PickeeArrival") -> None:
        """Pickee 도착 알림 처리"""
        is_section = msg.section_id is not None and msg.section_id >= 0
        logger.info("Handling arrival notice for order %d at section %s", msg.order_id, msg.section_id if is_section else 'N/A')
        await self._notifier.notify_robot_arrived(msg)
        if not is_section:
            logger.debug("Skipping product detection for non-section arrival.")
            return
        try:
            with self._db.session_scope() as session:
                product_ids = [item.product_id for item in session.query(OrderItem).filter_by(order_id=msg.order_id).all()]
            if not product_ids:
                logger.warning("No products to detect for order %d", msg.order_id)
                return
            req = PickeeProductDetect.Request(robot_id=msg.robot_id, order_id=msg.order_id, product_ids=product_ids)
            await self._robot.dispatch_product_detect(req)
        except Exception as e:
            logger.exception("Failed to start product detection for order %d: %s", msg.order_id, e)

    async def handle_product_detected(self, msg: "PickeeProductDetection") -> None:
        """Pickee 상품 인식 완료 처리"""
        logger.info("Handling product detection for order %d. Found %d products.", msg.order_id, len(msg.products))
        product_ids = [p.product_id for p in msg.products]
        products_data = []
        if product_ids:
            with self._db.session_scope() as session:
                product_map = self._load_products(session, product_ids)
                products_data = [{"product_id": p.product_id, "name": product_map.get(p.product_id).name if product_map.get(p.product_id) else "", "bbox_number": p.bbox_number} for p in msg.products]
        self._detected_product_bbox[msg.order_id] = {p.product_id: p.bbox_number for p in msg.products}
        await self._notifier.notify_product_selection_start(msg, products_data)

    async def handle_pickee_selection(self, msg: "PickeeProductSelection") -> None:
        """Pickee 상품 선택 결과 처리"""
        logger.info("Handling pickee selection result for order %d", msg.order_id)
        summary = {}
        with self._db.session_scope() as session:
            product = session.query(Product).filter_by(product_id=msg.product_id).first()
            total_items, total_price = self._calculate_order_summary(session, msg.order_id)
            summary = {"product": {"product_id": msg.product_id, "name": product.name if product else "", "quantity": msg.quantity, "price": product.price if product else 0}, "total_items": total_items, "total_price": total_price}
        await self._notifier.notify_cart_update(msg, summary)
        bbox_map = self._detected_product_bbox.get(msg.order_id)
        if bbox_map and msg.product_id in bbox_map:
            bbox_map.pop(msg.product_id, None)
            if not bbox_map:
                self._detected_product_bbox.pop(msg.order_id, None)

    async def handle_cart_handover(self, msg: "PickeeCartHandover") -> None:
        """Pickee 장바구니 전달 완료 처리"""
        order_id = msg.order_id
        robot_id = msg.robot_id
        logger.info("Handling cart handover for order %d, starting packing process.", order_id)
        packee_robot_id: Optional[int] = None
        try:
            if self._allocator:
                context = AllocationContext(order_id=order_id, required_type=RobotType.PACKEE)
                reserved_state = await self._allocator.reserve_robot(context)
                if reserved_state:
                    packee_robot_id = reserved_state.robot_id

            if packee_robot_id is None:
                logger.error("No available Packee robot for order %d.", order_id)
                await self._failure_handler.fail_order(order_id, "No available Packee robot.")
                return

            product_details_for_packee = []
            with self._db.session_scope() as session:
                items_with_products = session.query(OrderItem, Product).join(Product, OrderItem.product_id == Product.product_id).filter(OrderItem.order_id == order_id).all()
                for order_item, product in items_with_products:
                    product_details_for_packee.append({"product_id": product.product_id, "quantity": order_item.quantity, "length": product.length or 0, "width": product.width or 0, "height": product.height or 0, "weight": product.weight or 0, "fragile": product.fragile or False})
            
            start_req = PackeePackingStart.Request(robot_id=packee_robot_id, order_id=order_id)
            start_req.products = [ProductInfo(**detail) for detail in product_details_for_packee]
            start_res = await self._robot.dispatch_pack_task(start_req)
            if not start_res.success:
                raise RuntimeError(f"Failed to start packing: {start_res.message}")

            if hasattr(start_res, 'box_id') and start_res.box_id > 0:
                with self._db.session_scope() as session:
                    order = session.query(Order).filter_by(order_id=order_id).first()
                    if order:
                        order.box_id = start_res.box_id

            self._assignment_manager.assign_packee(order_id, packee_robot_id)
            await self._notifier.notify_packing_info(order_id=order_id, payload={'robot_id': packee_robot_id})
            self._failure_handler.start_reservation_monitor(packee_robot_id, order_id, RobotType.PACKEE)

            home_location_id = settings.PICKEE_HOME_LOCATION_ID
            if home_location_id:
                return_req = PickeeWorkflowReturnToBase.Request(robot_id=robot_id, location_id=home_location_id)
                await self._robot.dispatch_return_to_base(return_req)

            await self._failure_handler._release_pickee(order_id)

        except Exception as e:
            logger.exception("Failed to handle cart handover for order %d: %s", order_id, e)
            if packee_robot_id:
                await self._allocator.release_robot(packee_robot_id, order_id)
            await self._failure_handler.fail_order(order_id, "Failed during cart handover.")

    async def handle_packee_availability(self, msg: "PackeeAvailability") -> None:
        """Packee 작업 가능 여부 확인 처리"""
        logger.info("Packee availability for order %d: %s", msg.order_id, msg.available)
        await self._notifier.emit_work_info_notification(order_id=msg.order_id, robot_id=msg.robot_id)

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
        now = datetime.now(timezone.utc)
        active_orders: List[Dict[str, Any]] = []
        status_counter: Dict[int, int] = {}
        truly_active_count = 0
        
        with self._db.session_scope() as session:
            recent_completed_cutoff = now - timedelta(minutes=30)
            orders = session.query(Order).filter((Order.order_status < 8) | ((Order.order_status >= 8) & (Order.end_time >= recent_completed_cutoff))).order_by(Order.start_time.desc()).all()

            for order in orders:
                total_items, total_price = self._calculate_order_summary(session, order.order_id)
                progress_value = self._state_manager.get_progress(order.order_status)
                active_orders.append({"order_id": order.order_id, "customer_id": order.customer.id if order.customer else None, "status_code": order.order_status, "status": self._state_manager.get_label(order.order_status), "progress": progress_value, "started_at": order.start_time.isoformat() if order.start_time else None, "elapsed_seconds": (now - order.start_time).total_seconds() if order.start_time else None, "pickee_robot_id": self._assignment_manager.get_pickee(order.order_id), "packee_robot_id": self._assignment_manager.get_packee(order.order_id), "total_items": total_items, "total_price": total_price})
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
