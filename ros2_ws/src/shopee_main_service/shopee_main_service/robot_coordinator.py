from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import (
    PackeePackingComplete,
    PackeeRobotStatus,
    PickeeMoveStatus,
    PickeeProductSelection,
    PickeeRobotStatus,
)
from shopee_interfaces.srv import (
    PackeePackingCheckAvailability,
    PackeePackingStart,
    PickeeProductProcessSelection,
    PickeeWorkflowStartTask,
)

logger = logging.getLogger(__name__)


class RobotCoordinator(Node):
    """ROS2 node managing Pickee/Packee interactions."""

    def __init__(self) -> None:
        super().__init__("robot_coordinator")
        self._pickee_status_cb: Optional[Callable[[PickeeRobotStatus], None]] = None
        self._pickee_selection_cb: Optional[Callable[[PickeeProductSelection], None]] = None
        self._packee_status_cb: Optional[Callable[[PackeeRobotStatus], None]] = None
        self._packee_complete_cb: Optional[Callable[[PackeePackingComplete], None]] = None

        self._pickee_status_sub = self.create_subscription(
            PickeeRobotStatus, "/pickee/robot_status", self._on_pickee_status, 10
        )
        self._pickee_move_sub = self.create_subscription(
            PickeeMoveStatus, "/pickee/moving_status", self._on_pickee_move, 10
        )
        self._pickee_selection_sub = self.create_subscription(
            PickeeProductSelection, "/pickee/product/selection_result", self._on_pickee_selection, 10
        )
        self._packee_status_sub = self.create_subscription(
            PackeeRobotStatus, "/packee/robot_status", self._on_packee_status, 10
        )
        self._packee_complete_sub = self.create_subscription(
            PackeePackingComplete, "/packee/packing_complete", self._on_packee_complete, 10
        )

        self._pickee_start_cli = self.create_client(PickeeWorkflowStartTask, "/pickee/workflow/start_task")
        self._pickee_process_cli = self.create_client(PickeeProductProcessSelection, "/pickee/product/process_selection")
        self._packee_check_cli = self.create_client(PackeePackingCheckAvailability, "/packee/packing/check_availability")
        self._packee_start_cli = self.create_client(PackeePackingStart, "/packee/packing/start")

        self._ros_cache: Dict[str, object] = {}

    async def dispatch_pick_task(self, request: PickeeWorkflowStartTask.Request) -> PickeeWorkflowStartTask.Response:
        logger.info("Dispatching pick task: %s", request)
        return await self._call_service(self._pickee_start_cli, request)

    async def dispatch_pick_process(
        self, request: PickeeProductProcessSelection.Request
    ) -> PickeeProductProcessSelection.Response:
        logger.info("Dispatching pick process: %s", request)
        return await self._call_service(self._pickee_process_cli, request)

    async def check_packee_availability(
        self, request: PackeePackingCheckAvailability.Request
    ) -> PackeePackingCheckAvailability.Response:
        logger.info("Checking packee availability: %s", request)
        return await self._call_service(self._packee_check_cli, request)

    async def dispatch_pack_task(
        self, request: PackeePackingStart.Request
    ) -> PackeePackingStart.Response:
        logger.info("Dispatching pack task: %s", request)
        return await self._call_service(self._packee_start_cli, request)

    def set_status_callbacks(
        self,
        pickee_status_cb: Optional[Callable[[PickeeRobotStatus], None]] = None,
        pickee_selection_cb: Optional[Callable[[PickeeProductSelection], None]] = None,
        packee_status_cb: Optional[Callable[[PackeeRobotStatus], None]] = None,
        packee_complete_cb: Optional[Callable[[PackeePackingComplete], None]] = None,
    ) -> None:
        self._pickee_status_cb = pickee_status_cb
        self._pickee_selection_cb = pickee_selection_cb
        self._packee_status_cb = packee_status_cb
        self._packee_complete_cb = packee_complete_cb

    async def _call_service(self, client, request):
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"Service {client.srv_name} unavailable")
        future = client.call_async(request)
        await future
        return future.result()

    def _on_pickee_status(self, msg: PickeeRobotStatus) -> None:
        self._ros_cache["pickee_status"] = msg
        if self._pickee_status_cb:
            self._pickee_status_cb(msg)

    def _on_pickee_move(self, msg: PickeeMoveStatus) -> None:
        self._ros_cache["pickee_move"] = msg

    def _on_pickee_selection(self, msg: PickeeProductSelection) -> None:
        self._ros_cache["pickee_selection"] = msg
        if self._pickee_selection_cb:
            self._pickee_selection_cb(msg)

    def _on_packee_status(self, msg: PackeeRobotStatus) -> None:
        self._ros_cache["packee_status"] = msg
        if self._packee_status_cb:
            self._packee_status_cb(msg)

    def _on_packee_complete(self, msg: PackeePackingComplete) -> None:
        self._ros_cache["packee_complete"] = msg
        if self._packee_complete_cb:
            self._packee_complete_cb(msg)
