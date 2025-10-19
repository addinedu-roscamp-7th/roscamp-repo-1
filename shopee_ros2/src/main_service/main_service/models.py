"""
데이터 전송 객체 (DTO)

API 요청/응답 및 내부 데이터 전달에 사용하는 타입 안전한 클래스들입니다.
- 타입 체크 지원
- IDE 자동완성
- 데이터 검증 용이
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import OrderStatus, ErrorCode


# === API 공통 모델 ===

@dataclass
class ApiResponse:
    """
    API 응답 공통 포맷
    
    모든 API 응답이 따르는 기본 구조입니다.
    """
    type: str                              # 메시지 타입
    result: bool                           # 성공 여부
    message: str                           # 메시지
    data: Dict[str, Any] = field(default_factory=dict)  # 응답 데이터
    error_code: Optional[str] = None       # 에러 코드 (실패 시)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        response = {
            "type": self.type,
            "result": self.result,
            "message": self.message,
            "data": self.data
        }
        if self.error_code:
            response["error_code"] = self.error_code
        return response


@dataclass
class ApiRequest:
    """
    API 요청 공통 포맷
    """
    type: str                              # 메시지 타입
    data: Dict[str, Any] = field(default_factory=dict)  # 요청 데이터


# === 사용자 관련 ===

@dataclass
class LoginRequest:
    """로그인 요청"""
    user_id: str                           # 로그인 ID
    password: str                          # 비밀번호


@dataclass
class UserInfo:
    """사용자 정보"""
    user_id: str                           # 로그인 ID
    customer_id: int                       # 고객 ID (PK)
    name: str                              # 이름
    gender: bool                           # 성별 (1=남, 0=여)
    age: int                               # 나이
    address: str                           # 주소
    allergy_info: Dict[str, bool]          # 알레르기 정보
    is_vegan: bool                         # 비건 여부


# === 상품 관련 ===

@dataclass
class ProductSearchRequest:
    """상품 검색 요청"""
    query: str                             # 검색어
    allergy_filter: Optional[List[str]] = None  # 알레르기 필터
    is_vegan: Optional[bool] = None        # 비건 필터


@dataclass
class ProductInfo:
    """상품 정보"""
    product_id: int                        # 상품 ID
    barcode: str                           # 바코드
    name: str                              # 상품명
    quantity: int                          # 재고
    price: int                             # 가격
    discount_rate: int                     # 할인율 (%)
    category: str                          # 카테고리
    is_vegan_friendly: bool                # 비건 친화
    allergy_info: Dict[str, bool]          # 알레르기 정보
    section_id: int                        # 구역 ID
    warehouse_id: int                      # 창고 ID


@dataclass
class ProductLocation:
    """상품 위치 정보 (ROS 메시지용)"""
    product_id: int                        # 상품 ID
    section_id: int                        # 구역 ID
    warehouse_id: int                      # 창고 ID
    x: float                               # X 좌표
    y: float                               # Y 좌표


# === 주문 관련 ===

@dataclass
class OrderItem:
    """주문 상품 항목"""
    product_id: int                        # 상품 ID
    quantity: int                          # 수량
    price: Optional[int] = None            # 가격 (조회 시 포함)
    name: Optional[str] = None             # 상품명 (조회 시 포함)


@dataclass
class CreateOrderRequest:
    """주문 생성 요청"""
    user_id: str                           # 사용자 ID (로그인 ID)
    items: List[OrderItem]                 # 주문 상품 목록
    payment_method: str                    # 결제 수단


@dataclass
class OrderInfo:
    """주문 정보"""
    order_id: int                          # 주문 ID
    customer_id: int                       # 고객 ID
    order_date: str                        # 주문 일시
    total_price: int                       # 총 금액
    status: OrderStatus                    # 주문 상태
    robot_id: Optional[int] = None         # 로봇 ID
    failure_reason: Optional[str] = None   # 실패 사유
    items: List[OrderItem] = field(default_factory=list)  # 주문 항목


# === 로봇 관련 ===

@dataclass
class RobotInfo:
    """로봇 정보"""
    robot_id: int                          # 로봇 ID
    robot_type: str                        # 로봇 타입 (pickee/packee)
    status: str                            # 상태 (IDLE/WORKING/...)
    battery_level: Optional[float] = None  # 배터리 (%)
    current_task: Optional[str] = None     # 현재 작업
    location_x: Optional[float] = None     # 현재 위치 X
    location_y: Optional[float] = None     # 현재 위치 Y


@dataclass
class PickeeTaskRequest:
    """Pickee 작업 요청"""
    robot_id: int                          # 로봇 ID
    order_id: int                          # 주문 ID
    user_id: str                           # 사용자 ID (로그인 ID)
    product_list: List[ProductLocation]    # 상품 위치 목록


@dataclass
class PackeeTaskRequest:
    """Packee 작업 요청"""
    robot_id: int                          # 로봇 ID
    order_id: int                          # 주문 ID


# === 알림 관련 ===

@dataclass
class RobotMovingNotification:
    """로봇 이동 알림"""
    order_id: int                          # 주문 ID
    robot_id: int                          # 로봇 ID
    section_id: int                        # 목표 구역 ID


@dataclass
class RobotArrivalNotification:
    """로봇 도착 알림"""
    order_id: int                          # 주문 ID
    robot_id: int                          # 로봇 ID
    section_id: int                        # 도착 구역 ID


@dataclass
class CartUpdateNotification:
    """장바구니 업데이트 알림"""
    order_id: int                          # 주문 ID
    product_id: int                        # 상품 ID
    product_name: str                      # 상품명
    quantity: int                          # 수량
    action: str                            # 동작 (add/remove)


@dataclass
class PackingNotification:
    """포장 알림"""
    order_id: int                          # 주문 ID
    robot_id: int                          # 로봇 ID
    status: str                            # 상태 (start/complete)
    failure_reason: Optional[str] = None   # 실패 사유 (실패 시)


# === 재고 관련 (관리자) ===

@dataclass
class InventoryUpdateRequest:
    """재고 업데이트 요청"""
    product_id: int                        # 상품 ID
    quantity_change: int                   # 수량 변경 (양수=입고, 음수=출고)
    reason: str                            # 사유


@dataclass
class InventoryInfo:
    """재고 정보"""
    product_id: int                        # 상품 ID
    product_name: str                      # 상품명
    current_quantity: int                  # 현재 재고
    warehouse_id: int                      # 창고 ID
    section_id: int                        # 구역 ID
    last_updated: str                      # 마지막 업데이트 일시


# === 작업 이력 (관리자) ===

@dataclass
class RobotHistoryInfo:
    """로봇 작업 이력"""
    history_id: int                        # 이력 ID
    robot_id: int                          # 로봇 ID
    order_id: int                          # 주문 ID
    work_type: str                         # 작업 타입 (PICK/PACK)
    start_time: str                        # 시작 시간
    status: str                            # 상태 (SUCCESS/FAIL)
    end_time: Optional[str] = None         # 종료 시간
    failure_reason: Optional[str] = None   # 실패 사유

