from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .database_manager import DatabaseManager


class UserService:
    """Handles user authentication and profile fetching."""

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    async def login(self, customer_id: str, password: str) -> bool:
        # TODO: hash comparison
        return False

    async def get_user_info(self, customer_id: str) -> Optional[dict]:
        return None
