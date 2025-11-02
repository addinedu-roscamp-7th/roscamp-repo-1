"""ROS 기반 GUI 애플리케이션을 위한 로깅 모듈"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 로그 디렉터리 설정
LOG_DIR = Path(
    os.getenv("SHOPEE_APP_LOG_DIR", "~/.local/share/shopee_app/logs")
).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 로거 설정
logger = logging.getLogger("shopee_app")
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 파일 핸들러 설정
log_file = LOG_DIR / f'shopee_app_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


# 컴포넌트 로거 설정
def get_component_logger(component_name: str) -> logging.Logger:
    """컴포넌트별 로거를 생성해 반환합니다.

    Args:
        component_name: 컴포넌트 이름

    Returns:
        해당 컴포넌트를 위한 로거 인스턴스
    """
    component_logger = logger.getChild(component_name)
    component_logger.setLevel(logging.DEBUG)
    return component_logger


class ComponentLogger:
    """컴포넌트 상태 변화와 이벤트를 기록하는 로거 클래스"""

    def __init__(self, component_name: str) -> None:
        """
        Args:
            component_name: 컴포넌트 이름
        """
        self.logger = get_component_logger(component_name)

    def info(self, message: str, *args, **kwargs) -> None:
        """정보 메시지를 기록합니다."""
        self.logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """디버그 메시지를 기록합니다."""
        self.logger.debug(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """경고 메시지를 기록합니다."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """오류 메시지를 기록합니다."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """치명적 오류 메시지를 기록합니다."""
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """예외 발생 시 스택 트레이스와 함께 메시지를 기록합니다."""
        self.logger.exception(message, *args, **kwargs)

    def state_change(self, old_state: str, new_state: str) -> None:
        """상태 변경을 기록합니다.

        Args:
            old_state: 이전 상태
            new_state: 새로운 상태
        """
        self.info(f"상태 변경: {old_state} -> {new_state}")

    def event(self, event_name: str, **context: dict) -> None:
        """이벤트 발생을 기록합니다.

        Args:
            event_name: 이벤트 이름
            context: 이벤트 관련 컨텍스트 정보
        """
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            self.info(f"이벤트 발생: {event_name} ({context_str})")
        else:
            self.info(f"이벤트 발생: {event_name}")
