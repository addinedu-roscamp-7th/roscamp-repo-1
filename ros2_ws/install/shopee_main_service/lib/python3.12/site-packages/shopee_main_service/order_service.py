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
    PickeeWorkflowMoveToSection,
    PickeeWorkflowMoveToPackaging,
    PickeeWorkflowReturnToBase,
    PickeeWorkflowStartTask,
)
from shopee_interfaces.msg import ProductLocation

from .config import settings
from .database_models import Customer, Order, OrderItem, Product, RobotHistory
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
        self._default_locale_messages = {
            "robot_moving": "상품 위치로 이동 중입니다",
            "robot_arrived": "섹션에 도착했습니다",
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

    def _status_progress(self, status: int) -> int:
        return self._status_progress_map.get(status, 0)

    def _default_destination(self, status: int) -> str:
        return self._status_destination_map.get(status, "")

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

        await self._event_bus.publish(
            "app_push",
            {
                "type": "work_info_notification",
                "result": True,
                "error_code": None,
                "data": data,
                "message": "작업 정보 업데이트",
            },
        )

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
                        logger.warning(
                            "Failed to dispatch move_to_section for order %d: %s",
                            new_order.order_id,
                            move_exc,
                        )

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
                    logger.warning("Failed to dispatch move_to_packaging for order %d: %s", order_id, pack_exc)

            return True, summary
        except Exception as e:
            logger.exception("Failed to end shopping for order %d: %s", order_id, e)
            return False, None

    async def handle_moving_status(self, msg: "PickeeMoveStatus") -> None:
        """
        Pickee의 이동 상태 이벤트를 처리합니다.
        """
        logger.info("Handling moving status for order %d", msg.order_id)
        # PickeeMoveStatus에는 location_id만 있음
        destination_text = f"LOCATION_{msg.location_id}"
        await self._event_bus.publish(
            "app_push",
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
        logger.info("Handling arrival notice for order %d at section %d", msg.order_id, msg.section_id)
        await self._event_bus.publish(
            "app_push",
            {
                "type": "robot_arrived_notification",
                "result": True,
                "error_code": None,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "location_id": msg.location_id,
                    "section_id": msg.section_id,
                },
                "message": self._default_locale_messages["robot_arrived"],
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

        await self._event_bus.publish(
            "app_push",
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
        await self._event_bus.publish(
            "app_push",
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
        )

    async def handle_cart_handover(self, msg: "PickeeCartHandover") -> None:
        """
        Pickee의 장바구니 전달 완료 이벤트를 처리하여 Packee 작업을 시작하고 복귀 여부를 판단합니다.
        """
        order_id = msg.order_id
        robot_id = msg.robot_id
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

            await self._event_bus.publish(
                "app_push",
                {
                    "type": "packing_info_notification",
                    "result": True,
                    "error_code": None,
                    "data": packing_payload,
                    "message": "포장 정보 업데이트",
                },
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

        except Exception as e:
            logger.exception("Failed to handle cart handover for order %d: %s", order_id, e)

    async def handle_packee_complete(self, msg: "PackeePackingComplete") -> None:
        """
        Packee의 포장 완료 이벤트를 처리합니다.
        """
        logger.info("Handling packee complete for order %d. Success: %s", msg.order_id, msg.success)
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
            order.end_time = datetime.utcnow()
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

        await self._event_bus.publish(
            "app_push",
            {
                "type": "packing_info_notification",
                "result": msg.success,
                "error_code": None if msg.success else "ROBOT_002",
                "data": {
                    "order_status": order_status_text,
                    **product_info,
                },
                "message": packing_message,
            },
        )

        await self._event_bus.publish(
            "app_push",
            {
                "type": "order_complete_notification",
                "result": msg.success,
                "error_code": None if msg.success else "ROBOT_002",
                "data": {"order_id": msg.order_id, "status": order_status_text},
                "message": packing_message,
            },
        )
        await self._emit_work_info_notification(
            order_id=msg.order_id,
            robot_id=msg.robot_id,
            destination=self._default_destination(final_status),
        )
