"""
Shopee Main Service - 메인 진입점

이 모듈은 모든 서브 모듈(API, DB, 로봇, 서비스 등)을 조립하여 실행합니다.
ROS2와 asyncio를 동시에 실행하는 하이브리드 이벤트 루프를 구성합니다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict

import rclpy

from .api_controller import APIController
from .config import MainServiceConfig
from .database_manager import DatabaseManager
from .event_bus import EventBus
from .llm_client import LLMClient
from .order_service import OrderService
from .product_service import ProductService
from .robot_coordinator import RobotCoordinator
from .user_service import UserService

logger = logging.getLogger("shopee_main_service")


class MainServiceApp:
    """
    Main Service 애플리케이션 컨테이너
    모든 모듈을 생성하고 연결하여 실행합니다.
    - TCP API 서버 (App과 통신)
    - ROS2 노드 (로봇과 통신)
    - 데이터베이스, LLM, EventBus 등 내부 서비스
    """

    def __init__(self, config: MainServiceConfig) -> None:
        """
        Args:
            config: Main Service 설정
        """
        self._config = config
        
        # 내부 모듈 초기화 (의존성 순서 주의)
        self._event_bus = EventBus()  # 비동기 이벤트 버스
        self._db = DatabaseManager()  # DB 세션 관리자
        self._llm = LLMClient(
            base_url=config.llm_base_url,
            timeout=config.llm_timeout
        )  # LLM 서비스 클라이언트
        self._robot = RobotCoordinator()  # ROS2 노드 (로봇 통신)
        
        # 도메인 서비스 초기화
        self._user_service = UserService(self._db)
        self._product_service = ProductService(self._db, self._llm)
        self._order_service = OrderService(self._db, self._robot, self._event_bus)
        
        # API 핸들러 등록용 딕셔너리
        """문자열 키와 dict를 입력받아 dict를 반환하는 async 함수로 이루어진 딕셔너리"""
        self._handlers: Dict[str, Callable[[dict], Awaitable[dict]]] = {}
        
        # TCP API 서버
        self._api = APIController(
            config.api_host,
            config.api_port,
            self._handlers,
            self._event_bus
        )

    async def run(self) -> None:
        """
        메인 실행 루프
        
        1. API 핸들러 등록
        2. TCP 서버 시작
        3. ROS2 + asyncio 하이브리드 루프 실행
        4. 종료 시 정리
        """
        self._install_handlers()
        await self._api.start()
        
        try:
            # ROS2와 asyncio를 동시에 실행하는 루프
            while rclpy.ok():
                # ROS2 메시지 처리
                rclpy.spin_once(self._robot, timeout_sec=self._config.ros_spin_timeout)
                # asyncio 이벤트 루프에 제어권 양보
                await asyncio.sleep(0)
        finally:
            # 정리 작업
            await self._api.stop()
            self._robot.destroy_node()

    def _install_handlers(self) -> None:
        """
        API 메시지 핸들러 등록
        
        App에서 오는 각 요청 타입(type)에 대한 처리 함수를 등록합니다.
        새로운 API를 추가하려면 여기에 핸들러를 추가하면 됩니다.
        """
        
        async def handle_user_login(data):
            """사용자 로그인 처리"""
            # user_id는 로그인 ID (customer.id 컬럼)
            user_id = data.get("user_id", "")
            password = data.get("password", "")
            
            # 로그인 검증 (내부적으로 user_id -> customer_id 변환)
            success = await self._user_service.login(user_id, password)
            
            return {
                "type": "user_login_response",
                "result": success,
                "data": {"user_id": user_id} if success else {},
                "message": "ok" if success else "unauthorized",
            }

        async def handle_product_search(data):
            """상품 검색 처리 (LLM 연동)"""
            query = data.get("query", "")
            products = await self._product_service.search_products(query)
            
            return {
                "type": "product_search_response",
                "result": True,
                "data": {"products": products},
                "message": "ok",
            }

        # 핸들러 등록: 메시지 타입 → 처리 함수 매핑
        self._handlers.update(
            {
                "user_login": handle_user_login,
                "product_search": handle_product_search,
                # TODO: 추가 핸들러
                # "order_create": handle_order_create,
                # "product_selection": handle_product_selection,
                # "shopping_end": handle_shopping_end,
            }
        )


def main() -> None:
    """메인 진입점"""
    # 설정 로드
    config = MainServiceConfig.from_env()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    if config.log_file:
        logging.getLogger().addHandler(logging.FileHandler(config.log_file))
    
    logger.info("Starting Shopee Main Service")
    logger.info("Config: API=%s:%d, LLM=%s, DB=%s",
                config.api_host, config.api_port,
                config.llm_base_url,
                config.db_url.split('@')[-1] if '@' in config.db_url else config.db_url)
    
    # ROS2 및 애플리케이션 실행
    rclpy.init()
    try:
        app = MainServiceApp(config)
        asyncio.run(app.run())
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
