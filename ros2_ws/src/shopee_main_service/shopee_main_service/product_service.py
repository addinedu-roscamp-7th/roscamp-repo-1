from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .llm_client import LLMClient


class ProductService:
    """Coordinates product lookup and inventory operations."""

    def __init__(self, db: "DatabaseManager", llm_client: "LLMClient") -> None:
        self._db = db
        self._llm = llm_client

    async def search_products(self, query: str, filters: Optional[dict] = None) -> List[dict]:
        _ = filters
        # TODO: integrate LLM + DB paths
        return []

    async def update_inventory(self, product_id: int, quantity: int) -> None:
        _ = (product_id, quantity)
        # TODO: write inventory adjustments
