"""
Main Service 설정 관리

환경별 설정을 관리하고 환경 변수에서 로드합니다.
- 개발/스테이징/운영 환경 분리
- 환경 변수 우선 적용
- 타입 안전한 설정 접근
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MainServiceConfig:
    """
    Main Service 전역 설정
    
    환경 변수로 오버라이드 가능:
        SHOPEE_API_HOST, SHOPEE_API_PORT
        SHOPEE_LLM_URL, SHOPEE_LLM_TIMEOUT
        SHOPEE_DB_URL, SHOPEE_DB_POOL_SIZE
        SHOPEE_LOG_LEVEL
    """
    
    # === API 서버 설정 ===
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_max_connections: int = 100
    
    # === LLM 서비스 설정 ===
    llm_base_url: str = "http://localhost:8000"
    llm_timeout: float = 1.5  # 초
    llm_max_retries: int = 2
    llm_retry_backoff: float = 0.5  # 초 (지수 백오프 기본값)
    
    # === 데이터베이스 설정 ===
    db_url: str = "mysql+pymysql://shopee:shopee@localhost:3306/shopee"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30  # 초
    db_echo: bool = False  # SQL 로그 출력
    
    # === ROS2 설정 ===
    ros_spin_timeout: float = 0.1  # 초
    ros_service_timeout: float = 1.0  # 초
    
    # === EventBus 설정 ===
    event_bus_max_listeners: int = 100
    
    # === 로깅 설정 ===
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # === 재시도 설정 ===
    push_notification_max_retries: int = 3
    push_notification_retry_backoff: float = 1.0  # 초
    
    @classmethod
    def from_env(cls) -> MainServiceConfig:
        """
        환경 변수에서 설정 로드
        
        환경 변수가 없으면 기본값 사용
        
        Returns:
            MainServiceConfig: 설정 인스턴스
        """
        return cls(
            # API
            api_host=os.getenv("SHOPEE_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("SHOPEE_API_PORT", "5000")),
            api_max_connections=int(os.getenv("SHOPEE_API_MAX_CONN", "100")),
            
            # LLM
            llm_base_url=os.getenv("SHOPEE_LLM_URL", "http://localhost:8000"),
            llm_timeout=float(os.getenv("SHOPEE_LLM_TIMEOUT", "1.5")),
            llm_max_retries=int(os.getenv("SHOPEE_LLM_RETRIES", "2")),
            
            # Database
            db_url=os.getenv(
                "SHOPEE_DB_URL",
                "mysql+pymysql://shopee:shopee@localhost:3306/shopee"
            ),
            db_pool_size=int(os.getenv("SHOPEE_DB_POOL_SIZE", "10")),
            db_max_overflow=int(os.getenv("SHOPEE_DB_MAX_OVERFLOW", "20")),
            db_echo=os.getenv("SHOPEE_DB_ECHO", "false").lower() == "true",
            
            # ROS2
            ros_spin_timeout=float(os.getenv("SHOPEE_ROS_SPIN_TIMEOUT", "0.1")),
            ros_service_timeout=float(os.getenv("SHOPEE_ROS_SVC_TIMEOUT", "1.0")),
            
            # Logging
            log_level=os.getenv("SHOPEE_LOG_LEVEL", "INFO").upper(),
            log_file=os.getenv("SHOPEE_LOG_FILE"),
        )
    
    @classmethod
    def for_development(cls) -> MainServiceConfig:
        """개발 환경용 설정"""
        config = cls()
        config.log_level = "DEBUG"
        config.db_echo = True
        return config
    
    @classmethod
    def for_production(cls) -> MainServiceConfig:
        """운영 환경용 설정"""
        config = cls.from_env()
        config.log_level = "WARNING"
        config.db_echo = False
        return config

