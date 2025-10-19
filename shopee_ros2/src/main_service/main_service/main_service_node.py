"""
Shopee Main Service - 메인 진입점

이 모듈은 모든 서브 모듈(API, DB, 로봇, 서비스 등)을 조립하여 실행합니다.
ROS2와 asyncio를 동시에 실행하는 하이브리드 이벤트 루프를 구성합니다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import rclpy

from shopee_interfaces.srv import PickeeMainVideoStreamStart, PickeeMainVideoStreamStop

from .api_controller import APIController
from .config import settings
from .database_manager import DatabaseManager
from .event_bus import EventBus
from .llm_client import LLMClient
from .streaming_service import StreamingService
from .order_service import OrderService
from .product_service import ProductService
from .robot_coordinator import RobotCoordinator
from .robot_state_store import RobotStateStore
from .robot_selector import (
    RobotAllocator,
    RoundRobinStrategy,
    LeastWorkloadStrategy,
    BatteryAwareStrategy,
)
from .user_service import UserService
from .inventory_service import InventoryService
from .robot_history_service import RobotHistoryService
from .constants import RobotStatus
from .dashboard import DashboardController, DashboardDataProvider

logger = logging.getLogger('main_service')


class MainServiceApp:
    """
    Main Service 애플리케이션 컨테이너
    모든 모듈을 생성하고 연결하여 실행합니다.
    - TCP API 서버 (App과 통신)
    - ROS2 노드 (로봇과 통신)
    - 데이터베이스, LLM, EventBus 등 내부 서비스
    """

    def __init__(
        self,
        db=None,
        llm=None,
        robot=None,
        event_bus=None,
        streaming_service=None,
        inventory_service=None,
        robot_history_service=None,
    ) -> None:
        """
        Args:
            config: Main Service 설정
        """
        # 내부 모듈 초기화 (의존성 순서 주의)
        self._event_bus = event_bus or EventBus()  # 비동기 이벤트 버스
        self._db = db or DatabaseManager()  # DB 세션 관리자
        self._llm = llm or LLMClient(
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT
        )  # LLM 서비스 클라이언트
        self._streaming_service = streaming_service or StreamingService(
            host=settings.API_HOST,
            port=6000  # UDP Port from spec
        ) # UDP 스트리밍 중계기
        self._robot_state_store = RobotStateStore()
        self._robot = robot or RobotCoordinator(state_store=self._robot_state_store, event_bus=self._event_bus)  # ROS2 노드 (로봇 통신)

        # 설정에 따라 전략 선택
        strategy_name = settings.ROBOT_ALLOCATION_STRATEGY.lower()
        if strategy_name == "least_workload":
            strategy = LeastWorkloadStrategy()
        elif strategy_name == "battery_aware":
            strategy = BatteryAwareStrategy(min_battery_level=settings.ROBOT_MIN_BATTERY_LEVEL)
        else:  # 기본값: round_robin
            strategy = RoundRobinStrategy()

        self._robot_allocator = RobotAllocator(self._robot_state_store, strategy)

        # 도메인 서비스 초기화
        self._user_service = UserService(self._db)
        self._product_service = ProductService(self._db, self._llm)
        self._inventory_service = inventory_service or InventoryService(self._db)
        self._order_service = OrderService(
            self._db,
            self._robot,
            self._event_bus,
            allocator=self._robot_allocator,
            state_store=self._robot_state_store,
            inventory_service=self._inventory_service,
        )
        self._robot_history_service = robot_history_service or RobotHistoryService(self._db)
        self._dashboard_controller: Optional[DashboardController] = None

        # RobotCoordinator에 InventoryService 주입
        self._robot.set_inventory_service(self._inventory_service)

        # RobotCoordinator에 ProductService 주입 (의존성 해결)
        self._robot.set_product_service(self._product_service)
        
        # API 메시지 핸들러 등록
        
        # 핸들러들은 (data: dict, peer: tuple) 시그니처를 따릅니다.
        self._handlers: Dict[str, Callable[[dict, tuple], Awaitable[dict]]] = {}

        # TCP API 서버
        self._api = APIController(
            settings.API_HOST,
            settings.API_PORT,
            self._handlers,
            self._event_bus
        )

    async def _initialize_robot_states_from_db(self) -> None:
        """
        데이터베이스에서 모든 로봇을 조회하여 RobotStateStore를 초기화합니다.
        모든 로봇의 초기 상태는 OFFLINE으로 설정됩니다.
        """
        logger.info("Initializing robot states from database...")
        from .database_models import Robot
        from .constants import RobotType, RobotStatus
        from .robot_state_store import RobotState

        try:
            with self._db.session_scope() as session:
                db_robots = session.query(Robot).all()

                if not db_robots:
                    logger.warning("No robots found in the database.")
                    return

                for db_robot in db_robots:
                    # DB의 robot_type (TINYINT)를 RobotType Enum으로 변환
                    # NOTE: 1=PICKEE, 2=PACKEE 라고 가정합니다.
                    try:
                        if db_robot.robot_type == 1:
                            robot_type_enum = RobotType.PICKEE
                        elif db_robot.robot_type == 2:
                            robot_type_enum = RobotType.PACKEE
                        else:
                            raise ValueError(f"Unknown robot_type ID: {db_robot.robot_type}")
                    except Exception:
                        logger.warning(f"Unknown robot_type '{db_robot.robot_type}' for robot_id {db_robot.robot_id}. Skipping.")
                        continue

                    initial_state = RobotState(
                        robot_id=db_robot.robot_id,
                        robot_type=robot_type_enum,
                        status=RobotStatus.OFFLINE.value,
                    )
                    await self._robot_state_store.upsert_state(initial_state)
                
                logger.info(f"Initialized {len(db_robots)} robots to OFFLINE state.")

        except Exception as e:
            logger.exception(f"Failed to initialize robot states from database: {e}")

    async def initialize(self) -> None:
        """
        서비스 초기화

        1. API 핸들러 등록
        2. TCP 서버 시작
        3. 로봇 콜백 등록
        4. 대시보드 컨트롤러 시작 (GUI 활성화 시)
        """
        await self._initialize_robot_states_from_db()
        self._install_handlers()

        # ROS 노드가 비동기 동작을 위임할 이벤트 루프를 주입
        self._robot.set_asyncio_loop(asyncio.get_running_loop())

        # 로봇 이벤트 콜백 등록
        self._robot.set_status_callbacks(
            pickee_move_cb=self._order_service.handle_moving_status,
            pickee_arrival_cb=self._order_service.handle_arrival_notice,
            pickee_handover_cb=self._order_service.handle_cart_handover,
            pickee_product_detected_cb=self._order_service.handle_product_detected,
            pickee_selection_cb=self._order_service.handle_pickee_selection,
            packee_availability_cb=self._order_service.handle_packee_availability,
            packee_complete_cb=self._order_service.handle_packee_complete
        )

        if settings.GUI_ENABLED:
            logger.info("GUI is enabled, starting dashboard controller...")
            await self._start_dashboard_controller()
            logger.info("Dashboard controller started successfully")
        else:
            logger.info(f"GUI is disabled (GUI_ENABLED={settings.GUI_ENABLED})")

        await self._api.start()
        await self._streaming_service.start()

    async def cleanup(self) -> None:
        """
        서비스 종료 시 정리 작업
        """
        if self._dashboard_controller:
            await self._dashboard_controller.stop()
            self._dashboard_controller = None

        await self._api.stop()
        self._streaming_service.stop()
        self._robot.destroy_node()

        try:
            await self._robot_state_store.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to close robot state store: %s", exc)

    def _install_handlers(self) -> None:
        """
        API 메시지 핸들러 등록
        
        App에서 오는 각 요청 타입(type)에 대한 처리 함수를 등록합니다.
        새로운 API를 추가하려면 여기에 핸들러를 추가하면 됩니다.
        """
        
        async def handle_user_login(data, peer=None):
            """사용자 로그인 처리"""
            user_id = data.get("user_id", "")
            password = data.get("password", "")
            
            # 로그인 검증
            success = await self._user_service.login(user_id, password)
            
            user_info = None
            if success:
                # 로그인 성공 시 사용자 정보 조회
                user_info = await self._user_service.get_user_info(user_id)
                if user_info:
                    self._api.associate_peer_with_user(peer, user_id)

            return {
                "type": "user_login_response",
                "result": success and user_info is not None,
                "data": user_info or {},
                "message": "Login successful" if success and user_info else "Invalid credentials",
                "error_code": None if success and user_info else "AUTH_001",
            }

        async def handle_product_search(data, peer=None):
            """상품 검색 처리 (LLM 연동)"""
            query = data.get("query", "")
            result = await self._product_service.search_products(query)

            return {
                "type": "product_search_response",
                "result": True,
                "error_code": None,
                "data": result,  # {"products": [...], "total_count": N}
                "message": "ok",
            }

        async def handle_order_create(data, peer=None):
            """주문 생성 처리"""
            user_id = data.get("user_id")
            cart_items = data.get("cart_items")

            if not user_id or not cart_items:
                return {
                    "type": "order_create_response",
                    "result": False,
                    "message": "user_id and cart_items are required.",
                    "error_code": "SYS_001",
                }

            result = await self._order_service.create_order(user_id, cart_items)

            if result:
                order_id, robot_id = result
                return {
                    "type": "order_create_response",
                    "result": True,
                    "error_code": None,
                    "data": {"order_id": order_id, "robot_id": robot_id},
                    "message": "Order successfully created",
                }
            else:
                return {
                    "type": "order_create_response",
                    "result": False,
                    "message": "Failed to create order.",
                    "error_code": "ORDER_001",
                }

        async def handle_product_selection(data, peer=None):
            """상품 선택 처리"""
            order_id = data.get("order_id")
            robot_id = data.get("robot_id")
            bbox_number = data.get("bbox_number")
            product_id = data.get("product_id")

            if (
                order_id is None
                or robot_id is None
                or bbox_number is None
                or product_id is None
            ):
                return {
                    "type": "product_selection_response",
                    "result": False,
                    "message": "order_id, robot_id, bbox_number, and product_id are required.",
                    "error_code": "SYS_001",
                }

            success = await self._order_service.select_product(
                order_id, robot_id, bbox_number, product_id
            )

            return {
                "type": "product_selection_response",
                "result": success,
                "data": {
                    "order_id": order_id,
                    "product_id": product_id,
                    "bbox_number": bbox_number,
                },
                "message": "Product selection processed" if success else "Failed to process selection",
                "error_code": None if success else "ROBOT_002",
            }

        async def handle_product_selection_by_text(data, peer=None):
            """음성 기반 상품 선택 처리"""
            order_id = data.get("order_id")
            robot_id = data.get("robot_id")
            speech = (data or {}).get("speech") or ""

            if not all([order_id, robot_id, speech]):
                return {
                    "type": "product_selection_by_text_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "order_id, robot_id, and speech are required.",
                }

            detected_map = self._order_service.list_detected_products(int(order_id))

            resolved_product: Optional[int] = None
            bbox_number: Optional[int] = None

            detected_bbox_number = await self._llm.extract_bbox_number(speech)
            if detected_bbox_number is not None:
                for product_key, candidate_bbox in detected_map.items():
                    if candidate_bbox == detected_bbox_number:
                        resolved_product = int(product_key)
                        bbox_number = int(candidate_bbox)
                        break

            if resolved_product is None:
                intent_data = await self._llm.detect_intent(speech)
                if intent_data:
                    entities = intent_data.get("entities") or {}
                    product_id_entity = entities.get("product_id")
                    product_name = entities.get("product_name")

                    if product_id_entity is not None:
                        try:
                            resolved_product = int(product_id_entity)
                        except (TypeError, ValueError):
                            resolved_product = None

                    if resolved_product is None and product_name:
                        product_info = await self._product_service.get_product_by_name(product_name)
                        if product_info:
                            resolved_product = product_info["product_id"]

            if resolved_product is not None and bbox_number is None:
                bbox_number = self._order_service.get_detected_bbox(int(order_id), int(resolved_product))

            if resolved_product is None or bbox_number is None:
                return {
                    "type": "product_selection_by_text_response",
                    "result": False,
                    "error_code": "PROD_001",
                    "data": {},
                    "message": "Could not determine product from speech.",
                }

            success = await self._order_service.select_product(
                int(order_id),
                int(robot_id),
                int(bbox_number),
                int(resolved_product),
            )

            return {
                "type": "product_selection_by_text_response",
                "result": success,
                "error_code": None if success else "ROBOT_002",
                "data": {
                    "bbox": int(bbox_number),
                    "product_id": int(resolved_product),
                },
                "message": "Product selection processed" if success else "Failed to process selection",
            }

        async def handle_shopping_end(data, peer=None):
            """쇼핑 종료 처리"""
            user_id = data.get("user_id") # 로깅/인증용
            order_id = data.get("order_id")
            robot_id = data.get("robot_id") # 로봇 ID는 App에서 관리한다고 가정

            if not all([user_id, order_id, robot_id]):
                return {
                    "type": "shopping_end_response",
                    "result": False,
                    "message": "user_id, order_id, and robot_id are required.",
                    "error_code": "SYS_001",
                }

            success, summary = await self._order_service.end_shopping(order_id, robot_id)

            data_payload = {"order_id": order_id}
            if summary:
                data_payload.update(summary)

            return {
                "type": "shopping_end_response",
                "result": success,
                "data": data_payload,
                "message": "쇼핑이 종료되었습니다" if success else "Failed to end shopping",
                "error_code": None if success else "ROBOT_002",
            }

        async def handle_video_stream_start(data, peer):
            """영상 스트림 시작 처리"""
            robot_id = data.get("robot_id")
            user_id = data.get("user_id")
            user_type = data.get("user_type")
            app_ip, _ = peer  # App의 TCP 포트가 아닌, UDP 포트로 보내야 함
            # App의 UDP 수신 포트는 6000으로 가정
            APP_UDP_PORT = 6000

            logger.info(f"Starting video stream: robot={robot_id}, user={user_id}, app={app_ip}")

            # 1. 세션 기반 중계 시작
            self._streaming_service.start_relay(
                robot_id=robot_id,
                user_id=user_id,
                app_ip=app_ip,
                app_port=APP_UDP_PORT
            )

            # 2. 로봇에게 영상 송출 시작 명령
            req = PickeeMainVideoStreamStart.Request(robot_id=robot_id, user_id=user_id, user_type=user_type)
            res = await self._robot.dispatch_video_stream_start(req)

            success = res.success
            if not success:
                # 로봇이 실패한 경우 세션 제거
                self._streaming_service.stop_relay(robot_id, user_id)

            return {
                "type": "video_stream_start_response",
                "result": success,
                "error_code": None if success else "SYS_001",
                "data": {},
                "message": res.message if success else "Failed to start stream",
            }

        async def handle_video_stream_stop(data, peer=None):
            """영상 스트림 중지 처리"""
            robot_id = data.get("robot_id")
            user_id = data.get("user_id")
            user_type = data.get("user_type")

            logger.info(f"Stopping video stream: robot={robot_id}, user={user_id}")

            # 세션 종료
            self._streaming_service.stop_relay(robot_id, user_id)

            # 마지막 세션이면 로봇에게 중지 명령 전송
            sessions_for_robot = self._streaming_service._sessions.get(robot_id, [])
            if not sessions_for_robot:  # 더 이상 시청자가 없으면
                req = PickeeMainVideoStreamStop.Request(robot_id=robot_id, user_id=user_id, user_type=user_type)

                success = False
                message = "Failed to stop stream"
                try:
                    res = await self._robot.dispatch_video_stream_stop(req)
                    success = res.success
                    if res.message:
                        message = res.message
                    elif success:
                        message = "Video stream stopped"
                    else:
                        message = "Failed to stop stream"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Video stream stop request failed: %s", exc)
            else:
                success = True
                message = f"Session stopped (other users still watching robot {robot_id})"

            return {
                "type": "video_stream_stop_response",
                "result": success,
                "error_code": None if success else "SYS_001",
                "data": {},
                "message": message,
            }

        async def handle_inventory_search(data, peer=None):
            """재고 검색 처리"""
            filters = data or {}
            try:
                products, total_count = await self._inventory_service.search_products(filters)
                return {
                    "type": "inventory_search_response",
                    "result": True,
                    "error_code": None,
                    "data": {"products": products, "total_count": total_count},
                    "message": "Search completed",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Inventory search failed: %s", exc)
                return {
                    "type": "inventory_search_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {"products": [], "total_count": 0},
                    "message": "Failed to search inventory",
                }

        async def handle_inventory_create(data, peer=None):
            """재고 추가 처리"""
            payload = data or {}
            try:
                await self._inventory_service.create_product(payload)
                return {
                    "type": "inventory_create_response",
                    "result": True,
                    "error_code": None,
                    "data": {},
                    "message": "재고 정보를 추가하였습니다.",
                }
            except ValueError as exc:
                return {
                    "type": "inventory_create_response",
                    "result": False,
                    "error_code": "PROD_003",
                    "data": {},
                    "message": str(exc),
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Inventory create failed: %s", exc)
                return {
                    "type": "inventory_create_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "Failed to create inventory",
                }

        async def handle_inventory_update(data, peer=None):
            """재고 수정 처리"""
            payload = data or {}
            try:
                updated = await self._inventory_service.update_product(payload)
                if not updated:
                    return {
                        "type": "inventory_update_response",
                        "result": False,
                        "error_code": "PROD_001",
                        "data": {},
                        "message": "Product not found.",
                    }
                return {
                    "type": "inventory_update_response",
                    "result": True,
                    "error_code": None,
                    "data": {},
                    "message": "재고 정보를 수정하였습니다.",
                }
            except ValueError as exc:
                return {
                    "type": "inventory_update_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": str(exc),
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Inventory update failed: %s", exc)
                return {
                    "type": "inventory_update_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "Failed to update inventory",
                }

        async def handle_inventory_delete(data, peer=None):
            """재고 삭제 처리"""
            product_id = data.get("product_id")
            if product_id is None:
                return {
                    "type": "inventory_delete_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "product_id is required.",
                }
            try:
                deleted = await self._inventory_service.delete_product(product_id)
                if not deleted:
                    return {
                        "type": "inventory_delete_response",
                        "result": False,
                        "error_code": "PROD_001",
                        "data": {},
                        "message": "Product not found.",
                    }
                return {
                    "type": "inventory_delete_response",
                    "result": True,
                    "error_code": None,
                    "data": {},
                    "message": "재고 정보를 삭제하였습니다.",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Inventory delete failed: %s", exc)
                return {
                    "type": "inventory_delete_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "Failed to delete inventory",
                }

        async def handle_robot_history_search(data, peer=None):
            """로봇 작업 이력 검색"""
            filters = data or {}
            try:
                histories, total_count = await self._robot_history_service.search_histories(filters)
                return {
                    "type": "robot_history_search_response",
                    "result": True,
                    "error_code": None,
                    "data": {"histories": histories, "total_count": total_count},
                    "message": "Search completed",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Robot history search failed: %s", exc)
                return {
                    "type": "robot_history_search_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {"histories": [], "total_count": 0},
                    "message": "Failed to search robot histories",
                }

        async def handle_robot_status_request(data, peer=None):
            """로봇 상태 조회 (예약 정보 포함)"""
            robot_type_filter = data.get("robot_type")  # "pickee", "packee", or None

            try:
                if robot_type_filter:
                    from .constants import RobotType
                    rt = RobotType.PICKEE if robot_type_filter == "pickee" else RobotType.PACKEE
                    states = await self._robot_state_store.list_states(robot_type=rt)
                else:
                    states = await self._robot_state_store.list_states()

                robots_data = [
                    {
                        "robot_id": s.robot_id,
                        "type": s.robot_type.value,
                        "status": s.status,
                        "detailed_status": s.detailed_status,  # 세부 상태 추가
                        "reserved": s.reserved,
                        "active_order_id": s.active_order_id,
                        "battery_level": s.battery_level,
                        "maintenance_mode": s.maintenance_mode,
                        "last_update": s.last_update.isoformat() if s.last_update else None,
                    }
                    for s in states
                ]

                return {
                    "type": "robot_status_response",
                    "result": True,
                    "error_code": None,
                    "data": {
                        "robots": robots_data,
                        "total_count": len(robots_data),
                    },
                    "message": "Robot status retrieved",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Robot status request failed: %s", exc)
                return {
                    "type": "robot_status_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {"robots": [], "total_count": 0},
                    "message": "Failed to retrieve robot status",
                }

        async def handle_robot_maintenance_mode(data, peer=None):
            """로봇 유지보수 모드 설정/해제"""
            robot_id = data.get("robot_id")
            enabled = data.get("enabled", False)

            if robot_id is None:
                return {
                    "type": "robot_maintenance_mode_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "robot_id is required",
                }

            try:
                success = await self._robot_state_store.set_maintenance_mode(robot_id, enabled)

                if success:
                    return {
                        "type": "robot_maintenance_mode_response",
                        "result": True,
                        "error_code": None,
                        "data": {
                            "robot_id": robot_id,
                            "maintenance_mode": enabled,
                        },
                        "message": f"Maintenance mode {'enabled' if enabled else 'disabled'} for robot {robot_id}",
                    }
                else:
                    return {
                        "type": "robot_maintenance_mode_response",
                        "result": False,
                        "error_code": "ROBOT_001",
                        "data": {},
                        "message": f"Robot {robot_id} not found",
                    }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to set maintenance mode: %s", exc)
                return {
                    "type": "robot_maintenance_mode_response",
                    "result": False,
                    "error_code": "SYS_001",
                    "data": {},
                    "message": "Failed to set maintenance mode",
                }

        # 핸들러 등록: 메시지 타입 → 처리 함수 매핑
        self._handlers.update(
            {
                "user_login": handle_user_login,
                "product_search": handle_product_search,
                "order_create": handle_order_create,
                "product_selection": handle_product_selection,
                "product_selection_by_text": handle_product_selection_by_text,
                "shopping_end": handle_shopping_end,
                "video_stream_start": handle_video_stream_start,
                "video_stream_stop": handle_video_stream_stop,
                "inventory_search": handle_inventory_search,
                "inventory_create": handle_inventory_create,
                "inventory_update": handle_inventory_update,
                "inventory_delete": handle_inventory_delete,
                "robot_history_search": handle_robot_history_search,
                "robot_status_request": handle_robot_status_request,
                "robot_maintenance_mode": handle_robot_maintenance_mode,
            }
        )

    async def _start_dashboard_controller(self) -> None:
        """
        대시보드 컨트롤러를 초기화하고 주기적 데이터 수집을 시작한다.
        """
        if self._dashboard_controller:
            return

        loop = asyncio.get_running_loop()

        # MetricsCollector 생성
        from .metrics_collector import MetricsCollector
        
        metrics_collector = MetricsCollector(
            order_service=self._order_service,
            product_service=self._product_service,
            api_controller=self._api,
            robot_coordinator=self._robot,
            database_manager=self._db,
        )

        # DashboardDataProvider 생성 (MetricsCollector 사용)
        data_provider = DashboardDataProvider(
            order_service=self._order_service,
            robot_state_store=self._robot_state_store,
            metrics_collector=metrics_collector,
        )

        controller = DashboardController(
            loop,
            data_provider,
            self._event_bus,
            interval=settings.GUI_SNAPSHOT_INTERVAL,
        )
        await controller.start()
        self._dashboard_controller = controller


def main() -> None:
    """메인 진입점 - Qt 기반 이벤트 루프"""
    import sys
    import threading
    from PyQt6.QtWidgets import QApplication

    # 로깅 설정
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    if settings.LOG_FILE:
        logging.getLogger().addHandler(logging.FileHandler(settings.LOG_FILE))

    logger.info("Starting Shopee Main Service")
    logger.info("Config: API=%s:%d, LLM=%s, DB=%s",
                settings.API_HOST, settings.API_PORT,
                settings.LLM_BASE_URL,
                settings.DB_URL.split('@')[-1] if '@' in settings.DB_URL else settings.DB_URL)

    # ROS2 초기화
    rclpy.init()

    try:
        # Qt Application 생성 (메인 스레드에서)
        qt_app = QApplication(sys.argv)

        # MainServiceApp 생성
        service_app = MainServiceApp()

        # asyncio 이벤트 루프를 백그라운드 스레드에서 실행
        def run_asyncio_loop():
            """백그라운드 스레드에서 asyncio 이벤트 루프 실행"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # 서비스 초기화
                loop.run_until_complete(service_app.initialize())
                logger.info("Asyncio services initialized")

                # asyncio 이벤트 루프 계속 실행
                loop.run_forever()
            except Exception as e:
                logger.exception("Asyncio loop error: %s", e)
            finally:
                # 정리 작업
                try:
                    loop.run_until_complete(service_app.cleanup())
                except:
                    pass
                loop.close()
                logger.info("Asyncio loop stopped")

        # asyncio 스레드 시작
        asyncio_thread = threading.Thread(target=run_asyncio_loop, daemon=True, name='AsyncioLoop')
        asyncio_thread.start()

        # asyncio 초기화 대기 (최대 5초)
        import time
        for _ in range(50):
            if service_app._dashboard_controller:
                break
            time.sleep(0.1)

        # GUI가 활성화된 경우 GUI 윈도우 생성
        window = None
        if settings.GUI_ENABLED and service_app._dashboard_controller:
            from .dashboard import DashboardWindow
            window = DashboardWindow(
                service_app._dashboard_controller.bridge,
                service_app._robot,
                service_app._db,
                service_app._streaming_service
            )
            window.show()
            logger.info("Dashboard GUI window created")
        else:
            logger.warning("GUI enabled but dashboard controller not ready")

        logger.info("Starting Qt event loop")

        # Qt 이벤트 루프 실행
        exit_code = qt_app.exec()

        logger.info("Qt event loop finished with exit code: %d", exit_code)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise
    finally:
        rclpy.shutdown()
        logger.info("Shopee Main Service stopped")


if __name__ == "__main__":
    main()
