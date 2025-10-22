"""
로봇 할당 관리자

로봇-주문 간 할당 상태를 중앙에서 관리합니다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from .constants import RobotType

logger = logging.getLogger(__name__)


class RobotAssignmentManager:
    """
    로봇 할당 상태를 중앙에서 관리하는 클래스
    
    Pickee/Packee 로봇의 주문 할당 상태와 예약 모니터링 Task를 관리합니다.
    """

    def __init__(self) -> None:
        self._pickee_assignments: Dict[int, int] = {}  # order_id -> robot_id
        self._packee_assignments: Dict[int, int] = {}  # order_id -> robot_id
        self._pickee_order_by_robot: Dict[int, int] = {}  # robot_id -> order_id
        self._packee_order_by_robot: Dict[int, int] = {}  # robot_id -> order_id
        self._pickee_last_order_by_robot: Dict[int, int] = {}  # robot_id -> last order_id
        self._packee_last_order_by_robot: Dict[int, int] = {}  # robot_id -> last order_id
        self._pickee_history: Dict[int, int] = {}  # order_id -> last robot_id
        self._packee_history: Dict[int, int] = {}  # order_id -> last robot_id
        self._reservation_monitors: Dict[Tuple[int, int], asyncio.Task] = {}  # (order_id, robot_id) -> Task

    def assign_pickee(self, order_id: int, robot_id: int) -> None:
        """Pickee 로봇을 주문에 할당합니다."""
        self._pickee_assignments[order_id] = robot_id
        self._pickee_order_by_robot[robot_id] = order_id
        self._pickee_history[order_id] = robot_id
        self._pickee_last_order_by_robot[robot_id] = order_id
        logger.debug('Assigned Pickee robot %d to order %d', robot_id, order_id)

    def assign_packee(self, order_id: int, robot_id: int) -> None:
        """Packee 로봇을 주문에 할당합니다."""
        self._packee_assignments[order_id] = robot_id
        self._packee_order_by_robot[robot_id] = order_id
        self._packee_history[order_id] = robot_id
        self._packee_last_order_by_robot[robot_id] = order_id
        logger.debug('Assigned Packee robot %d to order %d', robot_id, order_id)

    def get_pickee(self, order_id: int) -> Optional[int]:
        """주문에 할당된 Pickee 로봇 ID를 반환합니다."""
        return self._pickee_assignments.get(order_id)

    def get_packee(self, order_id: int) -> Optional[int]:
        """주문에 할당된 Packee 로봇 ID를 반환합니다."""
        return self._packee_assignments.get(order_id)

    def get_pickee_order(self, robot_id: int) -> Optional[int]:
        """Pickee 로봇에 현재 할당된 주문 ID를 반환합니다."""
        return self._pickee_order_by_robot.get(robot_id)

    def get_packee_order(self, robot_id: int) -> Optional[int]:
        """Packee 로봇에 현재 할당된 주문 ID를 반환합니다."""
        return self._packee_order_by_robot.get(robot_id)

    def get_last_order_for_pickee(self, robot_id: int) -> Optional[int]:
        """특정 Pickee 로봇이 마지막으로 담당했던 주문 ID를 반환합니다."""
        return self._pickee_last_order_by_robot.get(robot_id)

    def get_last_order_for_packee(self, robot_id: int) -> Optional[int]:
        """특정 Packee 로봇이 마지막으로 담당했던 주문 ID를 반환합니다."""
        return self._packee_last_order_by_robot.get(robot_id)

    def get_last_pickee(self, order_id: int) -> Optional[int]:
        """최근에 주문에 배정되었던 Pickee 로봇 ID를 반환합니다."""
        return self._pickee_history.get(order_id)

    def get_last_packee(self, order_id: int) -> Optional[int]:
        """최근에 주문에 배정되었던 Packee 로봇 ID를 반환합니다."""
        return self._packee_history.get(order_id)

    def release_pickee(self, order_id: int) -> Optional[int]:
        """Pickee 로봇 할당을 해제하고 로봇 ID를 반환합니다."""
        robot_id = self._pickee_assignments.pop(order_id, None)
        if robot_id is not None:
            self._pickee_order_by_robot.pop(robot_id, None)
            logger.debug('Released Pickee robot %d from order %d', robot_id, order_id)
        return robot_id

    def release_packee(self, order_id: int) -> Optional[int]:
        """Packee 로봇 할당을 해제하고 로봇 ID를 반환합니다."""
        robot_id = self._packee_assignments.pop(order_id, None)
        if robot_id is not None:
            self._packee_order_by_robot.pop(robot_id, None)
            logger.debug('Released Packee robot %d from order %d', robot_id, order_id)
        return robot_id

    def get_active_pickee_orders(self) -> List[int]:
        """Pickee가 할당된 모든 주문 ID 목록을 반환합니다."""
        return list(self._pickee_assignments.keys())

    def get_active_packee_orders(self) -> List[int]:
        """Packee가 할당된 모든 주문 ID 목록을 반환합니다."""
        return list(self._packee_assignments.keys())

    def add_monitor(self, order_id: int, robot_id: int, task: asyncio.Task) -> None:
        """예약 모니터링 Task를 등록합니다."""
        key = (order_id, robot_id)
        if key in self._reservation_monitors:
            self._reservation_monitors[key].cancel()
        self._reservation_monitors[key] = task
        logger.debug('Added reservation monitor for robot %d, order %d', robot_id, order_id)

    def cancel_monitor(self, order_id: int, robot_id: int) -> None:
        """예약 모니터링 Task를 취소합니다."""
        key = (order_id, robot_id)
        if key in self._reservation_monitors:
            self._reservation_monitors[key].cancel()
            del self._reservation_monitors[key]
            logger.debug('Cancelled reservation monitor for robot %d, order %d', robot_id, order_id)

    def get_assignment_summary(self) -> Dict[str, int]:
        """할당 상태 요약 정보를 반환합니다."""
        return {
            'total_pickee_assignments': len(self._pickee_assignments),
            'total_packee_assignments': len(self._packee_assignments),
            'total_monitors': len(self._reservation_monitors),
        }
