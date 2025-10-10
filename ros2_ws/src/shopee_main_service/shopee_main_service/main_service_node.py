from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict

import rclpy

from .api_controller import APIController
from .database_manager import DatabaseManager
from .event_bus import EventBus
from .llm_client import LLMClient
from .order_service import OrderService
from .product_service import ProductService
from .robot_coordinator import RobotCoordinator
from .user_service import UserService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shopee_main_service")


class MainServiceApp:
    """Wires modules together following the design blueprint."""

    def __init__(self) -> None:
        self._event_bus = EventBus()
        self._db = DatabaseManager()
        self._llm = LLMClient(base_url="http://localhost:8000")
        self._robot = RobotCoordinator()
        self._user_service = UserService(self._db)
        self._product_service = ProductService(self._db, self._llm)
        self._order_service = OrderService(self._db, self._robot, self._event_bus)
        self._handlers: Dict[str, Callable[[dict], Awaitable[dict]]] = {}
        self._api = APIController("0.0.0.0", 5000, self._handlers, self._event_bus)

    async def run(self) -> None:
        self._install_handlers()
        await self._api.start()
        try:
            while rclpy.ok():
                rclpy.spin_once(self._robot, timeout_sec=0.1)
                await asyncio.sleep(0)
        finally:
            await self._api.stop()
            self._robot.destroy_node()

    def _install_handlers(self) -> None:
        async def handle_user_login(data):
            customer_id = data.get("customer_id", "")
            password = data.get("password", "")
            success = await self._user_service.login(customer_id, password)
            return {
                "type": "user_login_response",
                "result": success,
                "data": {"customer_id": customer_id} if success else {},
                "message": "ok" if success else "unauthorized",
            }

        async def handle_product_search(data):
            products = await self._product_service.search_products(data.get("query", ""))
            return {
                "type": "product_search_response",
                "result": True,
                "data": {"products": products},
                "message": "ok",
            }

        self._handlers.update(
            {
                "user_login": handle_user_login,
                "product_search": handle_product_search,
            }
        )


def main() -> None:
    rclpy.init()
    app = MainServiceApp()
    asyncio.run(app.run())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
