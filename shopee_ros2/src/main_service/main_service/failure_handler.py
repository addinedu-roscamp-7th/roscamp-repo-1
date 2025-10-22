"""
로봇 장애 처리기

로봇 장애 감지 및 재할당 등 복구 로직을 담당합니다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .config import settings
from .constants import EventTopic, RobotType, RobotStatus
from .database_models import Order, OrderItem
from .robot_selector import AllocationContext

if TYPE_CHECKING:
    from .database_manager import DatabaseManager
    from .event_bus import EventBus
    from .inventory_service import InventoryService
    from .order_notifier import OrderNotifier
    from .order_states import OrderStateManager
    from .robot_selector import RobotAllocator
    from .robot_coordinator import RobotCoordinator
    from .robot_state_store import RobotStateStore
    from .assignment_tracker import RobotAssignmentManager
    from .location_builder import ProductLocationBuilder

logger = logging.getLogger(__name__)


class RobotFailureHandler:
    """
    로봇 장애 처리 및 복구 로직을 담당합니다.
    """

    def __init__(
        self,
        db: "DatabaseManager",
        event_bus: "EventBus",
        state_manager: "OrderStateManager",
        notifier: "OrderNotifier",
        allocator: "RobotAllocator",
        state_store: "RobotStateStore",
        robot_coordinator: "RobotCoordinator",
        inventory_service: "InventoryService",
        assignment_manager: "RobotAssignmentManager",
        product_builder: "ProductLocationBuilder",
    ) -> None:
        self._db = db
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._notifier = notifier
        self._allocator = allocator
        self._state_store = state_store
        self._robot = robot_coordinator
        self._inventory_service = inventory_service
        self._assignment_manager = assignment_manager
        self._product_builder = product_builder

        if settings.ROBOT_AUTO_RECOVERY_ENABLED:
            self._event_bus.subscribe(EventTopic.ROBOT_FAILURE.value, self.handle_robot_failure)

    async def handle_robot_failure(self, event_data: Dict[str, Any]) -> None:
        """
        로봇 장애 이벤트를 처리하여 자동으로 다른 로봇을 할당합니다.
        """
        robot_id = event_data.get('robot_id')
        robot_type_str = event_data.get('robot_type')
        status = event_data.get('status')
        active_order_id = event_data.get('active_order_id')

        logger.warning(
            'Handling robot failure: robot_id=%d, type=%s, status=%s, order_id=%s',
            robot_id,
            robot_type_str,
            status,
            active_order_id,
        )

        if not active_order_id:
            logger.info('No active order for failed robot %d, skipping reallocation', robot_id)
            return

        try:
            robot_type = RobotType.PICKEE if robot_type_str == 'pickee' else RobotType.PACKEE
        except Exception:
            logger.error('Invalid robot type: %s', robot_type_str)
            return

        if robot_type == RobotType.PICKEE:
            self._assignment_manager.release_pickee(active_order_id)
        else:
            self._assignment_manager.release_packee(active_order_id)
        self.cancel_reservation_monitor(active_order_id, robot_id)

        if not self._allocator:
            logger.warning('No allocator available for robot reallocation')
            await self._notify_reassignment_failure(
                active_order_id, robot_id, robot_type_str, status, reason='allocator_unavailable'
            )
            return

        context = AllocationContext(order_id=active_order_id, required_type=robot_type)
        new_robot_state = await self._allocator.reserve_robot(context)

        if not new_robot_state:
            logger.error(
                'Failed to reallocate order %d: no available %s robot',
                active_order_id, robot_type_str
            )
            await self.fail_order(active_order_id, f'No available {robot_type_str} robot after failure')
            await self._notify_reassignment_failure(
                active_order_id, robot_id, robot_type_str, status, reason='no_available_robot'
            )
            return

        new_robot_id = new_robot_state.robot_id
        logger.info(
            'Attempting to reassign order %d from robot %d to robot %d',
            active_order_id, robot_id, new_robot_id
        )

        try:
            if robot_type == RobotType.PICKEE:
                success = await self._reassign_pickee(active_order_id, new_robot_id)
            else:
                success = await self._reassign_packee(active_order_id, new_robot_id)
        except Exception as exc:
            logger.exception(
                'Unexpected error occurred while reassigning order %d to robot %d: %s',
                active_order_id, new_robot_id, exc
            )
            success = False

        if success:
            await self._notify_reassignment_success(
                active_order_id, robot_id, new_robot_id, robot_type_str, status or ''
            )
        else:
            await self._allocator.release_robot(new_robot_id, active_order_id)
            await self.fail_order(active_order_id, f'Reassignment to robot {new_robot_id} failed')
            await self._notify_reassignment_failure(
                active_order_id, robot_id, robot_type_str, status, reason='reassignment_failed'
            )

    async def _reassign_pickee(self, order_id: int, new_robot_id: int) -> bool:
        """장애 발생 후 Pickee 로봇을 재할당합니다."""
        from shopee_interfaces.srv import PickeeWorkflowStartTask, PickeeWorkflowMoveToSection
        
        try:
            with self._db.session_scope() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()
                if not order or not order.customer:
                    raise ValueError(f'Cannot reassign pickee: order {order_id} not found or has no customer')
                user_id = order.customer.id
                product_locations = self._product_builder.build_product_locations(session, order_id)
                if not product_locations:
                    raise ValueError(f'No product locations available for order {order_id}')

            request = PickeeWorkflowStartTask.Request(
                robot_id=new_robot_id, order_id=order_id, user_id=user_id, product_list=product_locations
            )
            response = await self._robot.dispatch_pick_task(request)
            if not response.success:
                raise RuntimeError(f'Failed to dispatch pick task to robot {new_robot_id}: {response.message}')

            first_location = product_locations[0]
            try:
                move_req = PickeeWorkflowMoveToSection.Request(
                    robot_id=new_robot_id, order_id=order_id, location_id=first_location.location_id, section_id=first_location.section_id
                )
                await self._robot.dispatch_move_to_section(move_req)
            except Exception as exc:
                logger.warning(
                    'Failed to send move_to_section during reassign of order %d to robot %d: %s',
                    order_id, new_robot_id, exc
                )

            self._assignment_manager.assign_pickee(order_id, new_robot_id)
            self.start_reservation_monitor(new_robot_id, order_id, RobotType.PICKEE)
            return True
        except Exception as exc:
            logger.exception(
                'Reassigning pickee robot failed: order=%d, new_robot=%d, reason=%s',
                order_id, new_robot_id, exc
            )
            return False

    async def _reassign_packee(self, order_id: int, new_robot_id: int) -> bool:
        """장애 발생 후 Packee 로봇을 재할당합니다."""
        from shopee_interfaces.srv import PackeePackingStart
        
        try:
            request = PackeePackingStart.Request(robot_id=new_robot_id, order_id=order_id)
            response = await self._robot.dispatch_pack_task(request)
            if not response.success:
                raise RuntimeError(f'Failed to dispatch pack task to robot {new_robot_id}: {response.message}')

            self._assignment_manager.assign_packee(order_id, new_robot_id)
            self.start_reservation_monitor(new_robot_id, order_id, RobotType.PACKEE)
            return True
        except Exception as exc:
            logger.exception(
                'Reassigning packee robot failed: order=%d, new_robot=%d, reason=%s',
                order_id, new_robot_id, exc
            )
            return False

    async def _notify_reassignment_success(self, order_id: int, old_robot_id: int, new_robot_id: int, robot_type: str, status: str) -> None:
        """로봇 재할당 성공 알림."""
        await self._notifier._push_to_user(
            {
                'type': 'robot_reassignment_notification',
                'result': True,
                'error_code': '',
                'data': {
                    'order_id': order_id,
                    'old_robot_id': old_robot_id,
                    'new_robot_id': new_robot_id,
                    'robot_type': robot_type,
                    'reason': f'Robot {old_robot_id} {status}',
                },
                'message': f'로봇 {old_robot_id}에 문제가 발생하여 로봇 {new_robot_id}으로 교체되었습니다.',
            },
            order_id=order_id,
        )

    async def _notify_reassignment_failure(self, order_id: int, robot_id: int, robot_type: str, status: str, reason: str) -> None:
        """로봇 재할당 실패 알림."""
        await self._notifier._push_to_user(
            {
                'type': 'robot_failure_notification',
                'result': False,
                'error_code': 'ROBOT_001',
                'data': {
                    'order_id': order_id,
                    'robot_id': robot_id,
                    'robot_type': robot_type,
                    'status': status,
                    'reason': reason,
                },
                'message': self._format_reassignment_failure_message(robot_id, reason),
            },
            order_id=order_id,
        )

    def _format_reassignment_failure_message(self, robot_id: int, reason: str) -> str:
        """재할당 실패 메시지를 사유에 따라 생성"""
        if reason == 'no_available_robot':
            return f'로봇 {robot_id}에 문제가 발생했으며, 현재 가용한 대체 로봇이 없습니다.'
        if reason == 'allocator_unavailable':
            return f'로봇 {robot_id}에 문제가 발생했지만, 관제 시스템에서 자동 재할당을 지원하지 않습니다.'
        return f'로봇 {robot_id}에 문제가 발생했고, 재할당 중 오류가 발생했습니다.'

    async def fail_order(self, order_id: int, reason: str) -> None:
        """주문을 실패 상태로 변경하고 관련 리소스를 정리합니다."""
        if self._state_manager.fail_order(order_id, reason):
            await self.release_robot_for_order(order_id)
            await self._release_stock_for_order(order_id)
            await self._notifier.notify_order_failed(order_id, reason)

    async def _release_stock_for_order(self, order_id: int) -> None:
        """주문의 모든 상품 재고를 복구합니다."""
        if not self._inventory_service:
            logger.warning(f'Cannot release stock for order {order_id}: InventoryService not available')
            return
        try:
            with self._db.session_scope() as session:
                items = session.query(OrderItem).filter_by(order_id=order_id).all()
                for item in items:
                    await self._inventory_service.release_stock(item.product_id, item.quantity)
                    logger.info(
                        f'Released stock for failed order {order_id}: product={item.product_id}, qty={item.quantity}'
                    )
        except Exception as exc:
            logger.error(f'Failed to release stock for order {order_id}: {exc}')

    async def release_robot_for_order(self, order_id: int) -> None:
        """해당 주문에 할당된 모든 로봇(Pickee, Packee)의 예약을 해제합니다."""
        await self._release_pickee(order_id)
        await self._release_packee(order_id)

    async def _release_pickee(self, order_id: int) -> None:
        robot_id = self._assignment_manager.release_pickee(order_id)
        if robot_id is not None:
            self.cancel_reservation_monitor(order_id, robot_id)
            await self._allocator.release_robot(robot_id, order_id)

    async def _release_packee(self, order_id: int) -> None:
        robot_id = self._assignment_manager.release_packee(order_id)
        if robot_id is not None:
            self.cancel_reservation_monitor(order_id, robot_id)
            await self._allocator.release_robot(robot_id, order_id)

    def start_reservation_monitor(self, robot_id: int, order_id: int, robot_type: RobotType) -> None:
        """예약 타임아웃 모니터링 시작"""
        monitor_task = asyncio.create_task(self._monitor_reservation_timeout(robot_id, order_id, robot_type))
        self._assignment_manager.add_monitor(order_id, robot_id, monitor_task)

    def cancel_reservation_monitor(self, order_id: int, robot_id: int) -> None:
        """예약 타임아웃 모니터링 취소"""
        self._assignment_manager.cancel_monitor(order_id, robot_id)

    async def _monitor_reservation_timeout(self, robot_id: int, order_id: int, robot_type: RobotType, timeout: float = None) -> None:
        """예약 후 타임아웃이 발생하면 주문을 실패 처리합니다."""
        if timeout is None:
            timeout = settings.ROBOT_RESERVATION_TIMEOUT
        logger.debug('Starting reservation timeout monitor for robot %d, order %d (timeout=%ds)', robot_id, order_id, timeout)
        await asyncio.sleep(timeout)

        if self._state_store:
            state = await self._state_store.get_state(robot_id)
            if state and state.status == RobotStatus.IDLE.value and state.reserved:
                logger.warning('Reservation timeout: robot %d still IDLE after %ds for order %d', robot_id, timeout, order_id)
                await self._allocator.release_robot(robot_id, order_id)
                if robot_type == RobotType.PICKEE:
                    self._assignment_manager.release_pickee(order_id)
                else:
                    self._assignment_manager.release_packee(order_id)

                await self._event_bus.publish(EventTopic.RESERVATION_TIMEOUT.value, {
                    'robot_id': robot_id, 'order_id': order_id, 'robot_type': robot_type.value, 'timeout': timeout
                })
                await self.fail_order(order_id, f'Robot {robot_id} timeout after {timeout}s')
                await self._notifier.notify_robot_timeout(order_id, robot_id, robot_type.value)
            else:
                logger.debug('Reservation timeout monitor: robot %d is working normally for order %d', robot_id, order_id)
