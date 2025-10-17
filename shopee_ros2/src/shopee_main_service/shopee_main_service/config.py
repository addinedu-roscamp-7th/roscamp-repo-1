"""
Main Service 설정 관리

pydantic-settings를 사용하여 환경 변수 및 .env 파일에서 설정을 로드합니다.
- 타입 안전성 보장
- .env 파일 자동 감지
- 환경 변수 우선 적용
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


def _resolve_env_file() -> Optional[str]:
    """환경 변수 파일 경로를 탐색한다."""
    candidates: list[Path] = [
        BASE_DIR / '.env',
    ]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / '.env')
    candidates.append(Path.cwd() / '.env')

    checked: set[Path] = set()
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.is_file():
            return str(candidate)
    return None


class MainServiceConfig(BaseSettings):
    """
    Main Service 전역 설정
    
    .env 파일 또는 환경 변수에서 값을 로드합니다.
    환경 변수가 .env 파일보다 우선 순위가 높습니다.
    
    예: SHOPEE_API_PORT=8000
    """
    
    # pydantic-settings 설정
    model_config = SettingsConfigDict(
        env_file=_resolve_env_file(),    # .env 파일 탐색 결과
        env_file_encoding='utf-8',       # 인코딩
        env_prefix='SHOPEE_',            # 환경 변수 접두사
        case_sensitive=False,            # 대소문자 구분 안함
    )
    
    # === API 서버 설정 ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    API_MAX_CONNECTIONS: int = 100
    
    # === LLM 서비스 설정 ===
    LLM_BASE_URL: str = "http://192.168.0.154:5001"
    LLM_TIMEOUT: float = 1.5
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BACKOFF: float = 0.5
    
    # === 데이터베이스 설정 ===
    DB_URL: str = "mysql+pymysql://shopee:shopee@localhost:3306/shopee"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False
    
    # === ROS2 설정 ===
    ROS_SPIN_TIMEOUT: float = 0.1
    ROS_SERVICE_TIMEOUT: float = 5.0
    ROS_SERVICE_RETRY_ATTEMPTS: int = 3
    ROS_SERVICE_RETRY_BASE_DELAY: float = 0.2
    ROS_STATUS_HEALTH_TIMEOUT: float = 10.0
    ROS_EVENT_TOPIC_TIMEOUT: float = 300.0
    
    PICKEE_PACKING_LOCATION_ID: int = 2
    PICKEE_HOME_LOCATION_ID: int = 0
    DESTINATION_PACKING_NAME: str = "PACKING_AREA_A"
    DESTINATION_DELIVERY_NAME: str = "DELIVERY"
    DESTINATION_RETURN_NAME: str = "RETURN"

    # === Robot Fleet Management 설정 ===
    ROBOT_ALLOCATION_STRATEGY: str = "round_robin"  # round_robin | least_workload | battery_aware
    ROBOT_RESERVATION_TIMEOUT: float = 30.0  # 초
    ROBOT_MIN_BATTERY_LEVEL: float = 20.0  # %
    ROBOT_MAX_PICKEE: int = 10
    ROBOT_MAX_PACKEE: int = 5
    ROBOT_AUTO_RECOVERY_ENABLED: bool = True  # 자동 복구 활성화 여부

    # === 대시보드 설정 ===
    GUI_ENABLED: bool = False
    GUI_SNAPSHOT_INTERVAL: float = 1.0

    # === 로깅 설정 ===
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# 설정 인스턴스 생성
# 이 모듈을 임포트하는 모든 곳에서 이 인스턴스를 사용합니다.
settings = MainServiceConfig()
