"""
주문 관리 및 로봇 오케스트레이션 서비스

주문 생성부터 완료까지의 전체 워크플로우를 관리합니다.
- 주문 상태 머신
- Pickee/Packee 로봇 협업
- 이벤트 기반 상태 전환
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from shopee_interfaces.srv import (
    PackeePackingCheckAvailability,
    PackeePackingStart,
    PickeeProductDetect,
    PickeeProductProcessSelection,
    PickeeWorkflowEndShopping,
    PickeeWorkflowStartTask,
)
from shopee_interfaces.msg import ProductLocation

from .database_models import Customer, Order, OrderItem, Product
from .event_bus import EventBus

if TYPE_CHECKING:
    from shopee_interfaces.msg import (
        PackeePackingComplete,
        PickeeArrival,
        PickeeCartHandover,
        PickeeMoveStatus,
        PickeeProductDetection,
        PickeeProductSelection,
    )
    from .database_manager import DatabaseManager
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
    ) -> None:
        self._db = db
        self._robot = robot_coordinator
        self._event_bus = event_bus

    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """
        주문 생성 및 Pickee 작업 할당
        """
        logger.info("Creating order for user '%s' with %d items", user_id, len(items))
        with self._db.session_scope() as session:
            try:
                customer = session.query(Customer).filter_by(id=user_id).first()
                if not customer:
                    raise ValueError(f"Order creation failed: User '{user_id}' not found.")

                new_order = Order(
                    customer_id=customer.customer_id,
                    start_time=datetime.utcnow(),
                    order_status=1,  # 1: PAID
                )
                session.add(new_order)
                session.flush()

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
                
                robot_id = 1  # TODO: Dynamic robot selection
                request = PickeeWorkflowStartTask.Request(
                    robot_id=robot_id,
                    order_id=new_order.order_id,
                    user_id=user_id,
                    product_list=product_locations,
                )
                response = await self._robot.dispatch_pick_task(request)

                if not response.success:
                    raise RuntimeError(f"Failed to dispatch pick task: {response.message}")

                session.commit()
                logger.info("Order %d created and dispatched to robot %d", new_order.order_id, robot_id)
                return new_order.order_id, robot_id

            except Exception as e:
                logger.exception("Order creation failed: %s", e)
                session.rollback()
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

    async def end_shopping(self, order_id: int, robot_id: int) -> bool:
        """
        쇼핑을 종료하고 로봇을 포장대로 이동시킵니다.
        """
        logger.info("Ending shopping for order %d", order_id)
        try:
            request = PickeeWorkflowEndShopping.Request(robot_id=robot_id, order_id=order_id)
            response = await self._robot.dispatch_shopping_end(request)

            if not response.success:
                logger.error("Robot failed to end shopping for order %d: %s", order_id, response.message)
                return False

            with self._db.session_scope() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()
                if order:
                    order.order_status = 3  # 3: PICKED_UP
                    logger.info("Order %d status updated to PICKED_UP", order_id)
                else:
                    logger.error("Cannot update status for non-existent order %d", order_id)
                    return False
            return True
        except Exception as e:
            logger.exception("Failed to end shopping for order %d: %s", order_id, e)
            return False

    async def handle_moving_status(self, msg: "PickeeMoveStatus") -> None:
        """
        Pickee의 이동 상태 이벤트를 처리합니다.
        """
        logger.info("Handling moving status for order %d", msg.order_id)
        await self._event_bus.publish(
            "app_push",
            {
                "type": "robot_moving_notification",
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "destination": msg.destination,
                },
            },
        )

    async def handle_arrival_notice(self, msg: "PickeeArrival") -> None:
        """
        Pickee의 도착 이벤트를 처리하고, 상품 인식 단계를 시작합니다.
        """
        logger.info("Handling arrival notice for order %d at section %d", msg.order_id, msg.section_id)
        await self._event_bus.publish(
            "app_push",
            {
                "type": "robot_arrived_notification",
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "location_id": msg.location_id,
                    "section_id": msg.section_id,
                },
            },
        )
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
        products_data = [
            {"product_id": p.product_id, "name": "", "bbox_number": p.bbox_number}
            for p in msg.products
        ]
        await self._event_bus.publish(
            "app_push",
            {
                "type": "product_selection_start",
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "products": products_data,
                },
            },
        )

    async def handle_pickee_selection(self, msg: "PickeeProductSelection") -> None:
        """
        Pickee의 상품 선택 완료 이벤트를 처리합니다.
        """
        logger.info("Handling pickee selection result for order %d", msg.order_id)
        with self._db.session_scope() as session:
            order_item = (
                session.query(OrderItem)
                .filter_by(order_id=msg.order_id, product_id=msg.product_id)
                .first()
            )
            if not order_item:
                logger.error("Received selection result for non-existent order item. Order: %d, Product: %d", msg.order_id, msg.product_id)
                return
        await self._event_bus.publish(
            "app_push",
            {
                "type": "cart_update_notification",
                "data": {
                    "order_id": msg.order_id,
                    "action": "add" if msg.success else "add_fail",
                    "product": {
                        "product_id": msg.product_id,
                        "quantity": msg.quantity,
                    },
                },
            },
        )

    async def handle_cart_handover(self, msg: "PickeeCartHandover") -> None:
        """
        Pickee의 장바구니 전달 완료 이벤트를 처리하여 Packee 작업을 시작합니다.
        """
        order_id = msg.order_id
        logger.info("Handling cart handover for order %d, starting packing process.", order_id)
        try:
            check_req = PackeePackingCheckAvailability.Request()
            check_res = await self._robot.check_packee_availability(check_req)
            
            if not check_res.available:
                logger.error("No available Packee robot for order %d. Reason: %s", order_id, check_res.message)
                return

            packee_robot_id = check_res.robot_id
            logger.info("Packee robot %d is available for order %d.", packee_robot_id, order_id)

            start_req = PackeePackingStart.Request(robot_id=packee_robot_id, order_id=order_id)
            start_res = await self._robot.dispatch_pack_task(start_req)

            if not start_res.success:
                logger.error("Failed to start packing for order %d. Reason: %s", order_id, start_res.message)
                return
            
            logger.info("Successfully dispatched packing task for order %d to robot %d.", order_id, packee_robot_id)
        except Exception as e:
            logger.exception("Failed to handle cart handover for order %d: %s", order_id, e)

    async def handle_packee_complete(self, msg: "PackeePackingComplete") -> None:
        """
        Packee의 포장 완료 이벤트를 처리합니다.
        """
        logger.info("Handling packee complete for order %d. Success: %s", msg.order_id, msg.success)
        final_status = 8 if msg.success else 9  # 8: PACKED, 9: FAIL_PACK
        
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=msg.order_id).first()
            if not order:
                logger.error("Received packing complete for non-existent order %d", msg.order_id)
                return

            order.order_status = final_status
            order.end_time = datetime.utcnow()
            if not msg.success:
                order.failure_reason = msg.message
            
            logger.info("Order %d status updated to %d", msg.order_id, final_status)
            
        await self._event_bus.publish(
            "app_push",
            {
                "type": "order_complete_notification",
                "data": {"order_id": msg.order_id, "status": final_status},
            },
        )
