"""
주문 알림 관리자

사용자에게 전송되는 App 푸시 알림 관련 로직을 담당합니다.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .constants import EventTopic
from .database_models import Customer, Order, OrderItem, Product, RobotHistory

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .event_bus import EventBus
    from .order_states import OrderStateManager
    from shopee_interfaces.msg import (
        PickeeMoveStatus,
        PickeeArrival,
        PickeeProductDetection,
        PickeeProductSelection,
        PackeePackingComplete,
    )

logger = logging.getLogger(__name__)


class OrderNotifier:
    """
    주문 관련 사용자 알림 전송을 담당합니다.
    """

    def __init__(
        self,
        event_bus: EventBus,
        db: DatabaseManager,
        state_manager: OrderStateManager,
    ) -> None:
        self._event_bus = event_bus
        self._db = db
        self._state_manager = state_manager
        self._order_user_map: Dict[int, str] = {}
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

    def register_order_user(self, order_id: int, user_id: str) -> None:
        """order_id와 user_id 연관 정보를 저장합니다."""
        self._order_user_map[order_id] = user_id

    def clear_order_user(self, order_id: int) -> None:
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
        사용자별 푸시 메시지를 EventBus를 통해 발송합니다.
        """
        message = dict(payload)
        if order_id is not None:
            user_id = self._get_order_user_id(order_id)
            if user_id:
                message.setdefault("target_user_id", user_id)
        await self._event_bus.publish(EventTopic.APP_PUSH.value, message)

    async def notify_robot_moving(self, msg: "PickeeMoveStatus") -> None:
        """Pickee 이동 상태를 알립니다."""
        destination_text = f"LOCATION_{msg.location_id}"
        await self._push_to_user(
            {
                "type": "robot_moving_notification",
                "result": True,
                "error_code": "",
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
        await self.emit_work_info_notification(
            order_id=msg.order_id,
            robot_id=msg.robot_id,
            destination=destination_text,
        )

    async def notify_robot_arrived(self, msg: "PickeeArrival") -> None:
        """Pickee 도착을 알립니다."""
        is_section = msg.section_id is not None and msg.section_id >= 0
        section_payload = msg.section_id if is_section else -1
        message_key = "robot_arrived" if is_section else "robot_arrived_generic"
        await self._push_to_user(
            {
                "type": "robot_arrived_notification",
                "result": True,
                "error_code": "",
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

    async def notify_product_selection_start(self, msg: "PickeeProductDetection", products_data: list) -> None:
        """상품 선택 시작을 알립니다."""
        await self._push_to_user(
            {
                "type": "product_selection_start",
                "result": True,
                "error_code": "",
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "products": products_data,
                },
                "message": self._default_locale_messages["product_detected"],
            },
            order_id=msg.order_id,
        )

    async def notify_cart_update(self, msg: "PickeeProductSelection", summary: dict) -> None:
        """장바구니 업데이트를 알립니다."""
        message = msg.message or (
            self._default_locale_messages["cart_add_success"]
            if msg.success
            else self._default_locale_messages["cart_add_fail"]
        )
        error_code = "" if msg.success else "ROBOT_002"
        await self._push_to_user(
            {
                "type": "cart_update_notification",
                "result": msg.success,
                "error_code": error_code,
                "data": {
                    "order_id": msg.order_id,
                    "robot_id": msg.robot_id,
                    "action": "add" if msg.success else "add_fail",
                    "product": summary["product"],
                    "total_items": summary["total_items"],
                    "total_price": summary["total_price"],
                },
                "message": message,
            },
            order_id=msg.order_id,
        )

    async def notify_product_loaded(
        self,
        order_id: int,
        robot_id: int,
        product: Dict[str, Any],
        total_items: int,
        total_price: int,
        success: bool,
        message: str,
    ) -> None:
        """수동 적재 완료 이벤트를 알립니다."""
        await self._push_to_user(
            {
                "type": "product_loaded_notification",
                "result": success,
                "error_code": "" if success else "ROBOT_002",
                "data": {
                    "order_id": order_id,
                    "robot_id": robot_id,
                    "product": product,
                    "total_items": total_items,
                    "total_price": total_price,
                },
                "message": message or ("상품 적재가 완료되었습니다." if success else "상품 적재에 실패했습니다."),
            },
            order_id=order_id,
        )

    async def notify_picking_complete(self, order_id: int, robot_id: int) -> None:
        """모든 상품 피킹 완료를 알립니다."""
        await self._push_to_user(
            {
                "type": "picking_complete_notification",
                "result": True,
                "error_code": "",
                "data": {
                    "order_id": order_id,
                    "robot_id": robot_id,
                },
                "message": "모든 상품을 장바구니에 담았습니다. 포장 스테이션으로 이동합니다.",
            },
            order_id=order_id,
        )

    async def notify_manual_picking_complete(self, order_id: int) -> None:
        """수동 피킹 단계 완료를 알립니다."""
        await self._push_to_user(
            {
                "type": "manual_picking_complete",
                "result": True,
                "error_code": "",
                "data": {
                    "order_id": order_id,
                },
                "message": "수동 선택 상품을 모두 담았습니다. 로봇이 다음 구역으로 이동합니다.",
            },
            order_id=order_id,
        )

    async def notify_packing_info(self, order_id: int, payload: dict) -> None:
        """포장 정보를 알립니다."""
        await self._push_to_user(
            {
                "type": "packing_info_notification",
                "result": True,
                "error_code": "",
                "data": {**payload, "order_id": order_id},
                "message": "포장 정보 업데이트",
            },
            order_id=order_id,
        )

    async def notify_packing_complete(self, msg: "PackeePackingComplete", product_info: dict) -> None:
        """포장 완료를 알립니다."""
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
                "error_code": "" if msg.success else "ROBOT_002",
                "data": {
                    "order_id": msg.order_id,
                    "order_status": order_status_text,
                    **product_info,
                },
                "message": packing_message,
            },
            order_id=msg.order_id,
        )

    async def notify_order_failed(self, order_id: int, reason: str) -> None:
        """주문 실패를 알립니다."""
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
        self.clear_order_user(order_id)

    async def notify_robot_timeout(self, order_id: int, robot_id: int, robot_type: str) -> None:
        """로봇 타임아웃을 알립니다."""
        await self._push_to_user(
            {
                "type": "robot_timeout_notification",
                "result": False,
                "error_code": "ROBOT_003",
                "data": {
                    "order_id": order_id,
                    "robot_id": robot_id,
                    "robot_type": robot_type,
                },
                "message": f"로봇 {robot_id}이(가) 응답하지 않습니다. 다시 시도해주세요.",
            },
            order_id=order_id,
        )

    async def notify_packing_unavailable(self, order_id: int, robot_id: int, reason: str, detail_message: str) -> None:
        """포장 로봇 가용성 문제를 알립니다."""
        await self._push_to_user(
            {
                "type": "packing_unavailable_notification",
                "result": False,
                "error_code": "ROBOT_001",
                "data": {
                    "order_id": order_id,
                    "robot_id": robot_id,
                    "reason": reason,
                },
                "message": detail_message or reason,
            },
            order_id=order_id,
        )

    async def emit_work_info_notification(
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

            progress = self._state_manager.get_progress(order.order_status)
            destination_value = destination or self._state_manager.get_destination(order.order_status)

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
                "error_code": "",
                "data": data,
                "message": "작업 정보 업데이트",
            },
            order_id=order_id,
        )
