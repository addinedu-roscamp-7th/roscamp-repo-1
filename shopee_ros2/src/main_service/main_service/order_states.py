"""
주문 상태 관리자

주문 상태 머신(State Machine)과 관련된 로직을 담당합니다.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .config import settings
from .constants import OrderStatus
from .database_models import Order

if TYPE_CHECKING:
    from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class OrderStateManager:
    """
    주문 상태 머신(State Machine) 관리
    - 주문 상태 변경
    - 상태에 따른 정보(진행률, 목적지 등) 제공
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db
        self._status_progress_map = {
            1: 10,  # PAID
            2: 30,  # MOVING
            3: 60,  # PICKED_UP
            4: 70,  # MOVING_TO_PACK
            5: 80,  # PACKING
            6: 90,  # PACKING (세부 단계)
            7: 95,  # PACKING (세부 단계)
            8: 100, # PACKED
            9: 100, # FAIL_PACK
        }
        self._status_label_map = {
            1: OrderStatus.PAID.value,
            2: OrderStatus.MOVING.value,
            3: OrderStatus.PICKED_UP.value,
            4: OrderStatus.MOVING_TO_PACK.value,
            5: OrderStatus.PACKING.value,
            6: OrderStatus.PACKING.value,
            7: OrderStatus.PACKING.value,
            8: OrderStatus.PACKED.value,
            9: OrderStatus.FAIL_PACK.value,
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

    def get_progress(self, status: int) -> int:
        """주문 상태 코드에 해당하는 진행률을 반환합니다."""
        return self._status_progress_map.get(status, 0)

    def get_label(self, status: int) -> str:
        """주문 상태 코드에 해당하는 라벨 문자열을 반환합니다."""
        return self._status_label_map.get(status, OrderStatus.PAID.value)

    def get_destination(self, status: int) -> str:
        """주문 상태 코드에 해당하는 목적지 문자열을 반환합니다."""
        return self._status_destination_map.get(status, "")

    def set_status(self, order_id: int, status_code: int) -> bool:
        """주문의 상태를 주어진 코드로 변경합니다."""
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if order:
                order.order_status = status_code
                logger.info("Order %d status updated to %d", order_id, status_code)
                return True
            logger.error("Cannot update status for non-existent order %d", order_id)
            return False

    def set_status_picked_up(self, order_id: int) -> bool:
        """주문 상태를 'PICKED_UP'(3)으로 변경합니다."""
        return self.set_status(order_id, 3)

    def set_status_packed(self, order_id: int, success: bool, message: Optional[str] = None) -> bool:
        """주문 상태를 'PACKED'(8) 또는 'FAIL_PACK'(9)으로 변경합니다."""
        final_status = 8 if success else 9
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if not order:
                logger.error("Received packing complete for non-existent order %d", order_id)
                return False

            order.order_status = final_status
            order.end_time = datetime.now()
            if not success:
                order.failure_reason = message
            
            logger.info("Order %d status updated to %d", order_id, final_status)
            return True

    def fail_order(self, order_id: int, reason: str) -> bool:
        """주문을 실패 상태(9)로 변경하고 사유를 기록합니다."""
        with self._db.session_scope() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()
            if order:
                order.order_status = 9  # FAIL_PACK
                order.end_time = datetime.now()
                order.failure_reason = reason
                logger.info("Order %d marked as failed: %s", order_id, reason)
                return True
            return False