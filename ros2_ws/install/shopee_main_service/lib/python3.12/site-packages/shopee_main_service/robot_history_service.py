"""
로봇 작업 이력 서비스

로봇 작업 히스토리 조회 기능을 제공합니다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from .database_models import RobotHistory


class RobotHistoryService:
    """로봇 히스토리 검색 로직"""

    def __init__(self, db):
        self._db = db

    async def search_histories(self, filters: Dict[str, object]) -> Tuple[List[Dict[str, object]], int]:
        """
        로봇 작업 히스토리 검색

        Args:
            filters: 검색 조건

        Returns:
            (히스토리 목록, 총 개수)
        """
        with self._db.session_scope() as session:
            query = session.query(RobotHistory)

            if (history_id := filters.get("robot_history_id")) is not None:
                query = query.filter(RobotHistory.robot_history_id == history_id)
            if (robot_id := filters.get("robot_id")) is not None:
                query = query.filter(RobotHistory.robot_id == robot_id)
            if (order_item_id := filters.get("order_item_id")) is not None:
                query = query.filter(RobotHistory.order_item_id == order_item_id)
            if (failure_reason := filters.get("failure_reason")):
                query = query.filter(RobotHistory.failure_reason == failure_reason)
            if (is_complete := filters.get("is_complete")) is not None:
                query = query.filter(RobotHistory.is_complete == self._to_bool(is_complete))
            if (active_duration := filters.get("active_duration")) is not None:
                query = query.filter(RobotHistory.active_duration == int(active_duration))

            created_at = filters.get("created_at")
            if created_at:
                dt = self._parse_datetime(created_at)
                if dt:
                    query = query.filter(RobotHistory.created_at >= dt)

            histories = query.all()
            history_dicts = [self._history_to_dict(history) for history in histories]
            return history_dicts, len(history_dicts)

    def _history_to_dict(self, history: RobotHistory) -> Dict[str, object]:
        return {
            "robot_history_id": history.robot_history_id,
            "robot_id": history.robot_id,
            "order_item_id": history.order_item_id,
            "failure_reason": history.failure_reason,
            "is_complete": history.is_complete,
            "active_duration": history.active_duration,
            "created_at": history.created_at.isoformat() if history.created_at else None,
        }

    def _parse_datetime(self, value: object) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                return None
        return None

    def _to_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "y"}
        return bool(value)
