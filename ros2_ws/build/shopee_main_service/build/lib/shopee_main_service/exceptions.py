"""
커스텀 예외 클래스

Main Service에서 발생하는 모든 예외를 정의합니다.
- 명확한 예외 타입으로 에러 핸들링 개선
- 에러 코드 자동 매핑
- 로깅 및 디버깅 용이
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .constants import ErrorCode


class ShopeeException(Exception):
    """
    Shopee 기본 예외 클래스
    
    모든 커스텀 예외의 부모 클래스입니다.
    에러 코드와 메시지를 포함합니다.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            message: 에러 메시지
            error_code: 에러 코드 (ErrorCode Enum)
            details: 추가 정보 (선택)
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환 (API 응답용)
        
        Returns:
            dict: 에러 정보
        """
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }


# === 인증 예외 ===

class AuthenticationError(ShopeeException):
    """인증 실패"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_CREDENTIALS
        )


class UserNotFoundError(ShopeeException):
    """사용자를 찾을 수 없음"""
    
    def __init__(self, user_id: str):
        super().__init__(
            message=f"User '{user_id}' not found",
            error_code=ErrorCode.USER_NOT_FOUND,
            details={"user_id": user_id}
        )


class UnauthorizedError(ShopeeException):
    """권한 없음"""
    
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED
        )


# === 주문 예외 ===

class OrderNotFoundError(ShopeeException):
    """주문을 찾을 수 없음"""
    
    def __init__(self, order_id: int):
        super().__init__(
            message=f"Order {order_id} not found",
            error_code=ErrorCode.ORDER_NOT_FOUND,
            details={"order_id": order_id}
        )


class InvalidOrderStateError(ShopeeException):
    """잘못된 주문 상태"""
    
    def __init__(self, order_id: int, current_state: str, expected_state: str):
        super().__init__(
            message=f"Order {order_id} is in '{current_state}', expected '{expected_state}'",
            error_code=ErrorCode.INVALID_ORDER_STATE,
            details={
                "order_id": order_id,
                "current_state": current_state,
                "expected_state": expected_state
            }
        )


class OrderCreateError(ShopeeException):
    """주문 생성 실패"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Failed to create order: {reason}",
            error_code=ErrorCode.ORDER_CREATE_FAILED,
            details={"reason": reason}
        )


# === 로봇 예외 ===

class RobotUnavailableError(ShopeeException):
    """로봇 사용 불가"""
    
    def __init__(self, robot_type: str):
        super().__init__(
            message=f"{robot_type} robot is not available",
            error_code=ErrorCode.ROBOT_NOT_AVAILABLE,
            details={"robot_type": robot_type}
        )


class RobotTaskError(ShopeeException):
    """로봇 작업 실패"""
    
    def __init__(self, robot_id: int, task: str, reason: str):
        super().__init__(
            message=f"Robot {robot_id} failed task '{task}': {reason}",
            error_code=ErrorCode.ROBOT_TASK_FAILED,
            details={
                "robot_id": robot_id,
                "task": task,
                "reason": reason
            }
        )


class RobotTimeoutError(ShopeeException):
    """로봇 응답 타임아웃"""
    
    def __init__(self, robot_id: int, service_name: str):
        super().__init__(
            message=f"Robot {robot_id} service '{service_name}' timeout",
            error_code=ErrorCode.ROBOT_TIMEOUT,
            details={
                "robot_id": robot_id,
                "service_name": service_name
            }
        )


# === 상품 예외 ===

class ProductNotFoundError(ShopeeException):
    """상품을 찾을 수 없음"""
    
    def __init__(self, product_id: int):
        super().__init__(
            message=f"Product {product_id} not found",
            error_code=ErrorCode.PRODUCT_NOT_FOUND,
            details={"product_id": product_id}
        )


class InsufficientStockError(ShopeeException):
    """재고 부족"""
    
    def __init__(self, product_id: int, requested: int, available: int):
        super().__init__(
            message=f"Insufficient stock for product {product_id}: requested={requested}, available={available}",
            error_code=ErrorCode.INSUFFICIENT_STOCK,
            details={
                "product_id": product_id,
                "requested": requested,
                "available": available
            }
        )


class ProductSearchError(ShopeeException):
    """상품 검색 실패"""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Product search failed for '{query}': {reason}",
            error_code=ErrorCode.PRODUCT_SEARCH_FAILED,
            details={
                "query": query,
                "reason": reason
            }
        )


# === 시스템 예외 ===

class SystemError(ShopeeException):
    """시스템 일반 오류"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.SYSTEM_ERROR,
            details=details
        )

