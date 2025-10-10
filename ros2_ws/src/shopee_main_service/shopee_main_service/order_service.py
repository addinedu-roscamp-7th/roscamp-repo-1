from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from .event_bus import EventBus

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .robot_coordinator import RobotCoordinator

logger = logging.getLogger(__name__)


class OrderService:
    """Order state machine and orchestration logic."""

    def __init__(
        self,
        db: "DatabaseManager",
        robot_coordinator: "RobotCoordinator",
        event_bus: EventBus,
    ) -> None:
        self._db = db
        self._robot = robot_coordinator
        self._event_bus = event_bus

    async def create_order(self, customer_id: str, items: List[Dict[str, object]]) -> int:
        logger.info("Creating order for %s items=%d", customer_id, len(items))
        # TODO: persist order and return identifier
        return 0

    async def process_order(self, order_id: int) -> None:
        logger.info("Processing order %d", order_id)
        # TODO: trigger pick workflow

    async def handle_pickee_event(self, topic: str, payload: Dict[str, object]) -> None:
        logger.debug("Pickee event topic=%s payload=%s", topic, payload)
        # TODO: update order state and emit events

    async def finalize_order(self, order_id: int, status: str) -> None:
        logger.info("Finalizing order %d status=%s", order_id, status)
        # TODO: update DB and notify app

    async def get_order(self, order_id: int) -> Optional[dict]:
        logger.debug("Fetch order %d", order_id)
        return None
