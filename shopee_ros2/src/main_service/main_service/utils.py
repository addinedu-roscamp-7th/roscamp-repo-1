"""
유틸리티 함수

공통으로 사용하는 헬퍼 함수들입니다.
- 재시도 로직
- 데이터 변환
- 검증
"""
from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    비동기 함수 재시도
    
    지수 백오프(exponential backoff)를 사용하여 재시도합니다.
    
    Args:
        func: 재시도할 비동기 함수
        max_retries: 최대 재시도 횟수
        backoff: 초기 대기 시간 (초)
        exceptions: 재시도할 예외 타입 튜플
        
    Returns:
        함수 실행 결과
        
    Raises:
        마지막 시도에서 발생한 예외
        
    사용 예:
        result = await retry_async(
            lambda: llm_client.generate_query(text),
            max_retries=3,
            backoff=0.5
        )
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries - 1:
                # 마지막 시도 실패
                logger.error(
                    "Function %s failed after %d attempts: %s",
                    func.__name__ if hasattr(func, '__name__') else 'lambda',
                    max_retries,
                    str(e)
                )
                raise
            
            # 재시도 전 대기 (지수 백오프)
            wait_time = backoff * (2 ** attempt)
            logger.warning(
                "Function %s failed (attempt %d/%d), retrying in %.2fs: %s",
                func.__name__ if hasattr(func, '__name__') else 'lambda',
                attempt + 1,
                max_retries,
                wait_time,
                str(e)
            )
            await asyncio.sleep(wait_time)
    
    # 이론상 여기 도달하지 않음
    raise last_exception if last_exception else RuntimeError("Retry failed")


def ensure_dict(data: Any) -> Dict[str, Any]:
    """
    데이터가 딕셔너리인지 확인하고 변환
    
    Args:
        data: 입력 데이터
        
    Returns:
        dict: 딕셔너리로 변환된 데이터
        
    Raises:
        TypeError: 딕셔너리로 변환할 수 없는 경우
    """
    if isinstance(data, dict):
        return data
    
    if hasattr(data, '__dict__'):
        return dict(data.__dict__)

    if hasattr(data, 'to_dict'):
        result = data.to_dict()
        return dict(result) if result else {}
    
    raise TypeError(f"Cannot convert {type(data)} to dict")


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    안전하게 딕셔너리에서 값 가져오기
    
    중첩된 키도 지원합니다 (예: "user.name")
    
    Args:
        data: 딕셔너리
        key: 키 (점 표기법 지원)
        default: 기본값
        
    Returns:
        값 또는 기본값
        
    사용 예:
        name = safe_get(user_data, "profile.name", "Unknown")
    """
    if '.' not in key:
        return data.get(key, default)
    
    # 중첩된 키 처리
    keys = key.split('.')
    current: Any = data

    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default

    return current


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """
    필수 필드 검증
    
    Args:
        data: 검증할 딕셔너리
        required_fields: 필수 필드 목록
        
    Raises:
        ValueError: 필수 필드가 누락된 경우
        
    사용 예:
        validate_required_fields(
            request_data,
            ["user_id", "password"]
        )
    """
    missing = [field for field in required_fields if field not in data]
    
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    문자열 자르기
    
    Args:
        text: 원본 문자열
        max_length: 최대 길이
        suffix: 접미사 (기본: "...")
        
    Returns:
        잘린 문자열
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_error_response(
    message_type: str,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    에러 응답 포맷 생성
    
    Args:
        message_type: 메시지 타입
        error_code: 에러 코드
        message: 에러 메시지
        details: 추가 정보
        
    Returns:
        dict: 표준 에러 응답
    """
    response: Dict[str, Any] = {
        "type": message_type,
        "result": False,
        "error_code": error_code,
        "message": message,
        "data": {}
    }

    if details:
        data_dict = response["data"]
        if isinstance(data_dict, dict):
            data_dict["details"] = details

    return response


def format_success_response(
    message_type: str,
    data: Dict[str, Any],
    message: str = "ok"
) -> Dict[str, Any]:
    """
    성공 응답 포맷 생성
    
    Args:
        message_type: 메시지 타입
        data: 응답 데이터
        message: 메시지
        
    Returns:
        dict: 표준 성공 응답
    """
    return {
        "type": message_type,
        "result": True,
        "message": message,
        "data": data
    }


class Timer:
    """
    실행 시간 측정 컨텍스트 매니저
    
    사용 예:
        with Timer("DB Query"):
            result = session.query(...)
        # 출력: "DB Query took 0.12s"
    """
    
    def __init__(self, name: str = "Operation", logger_instance: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            self.logger.debug("%s took %.3fs", self.name, elapsed)

