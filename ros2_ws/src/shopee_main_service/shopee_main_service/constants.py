"""
상수 및 Enum 정의

시스템 전체에서 사용하는 상수, 상태, 에러 코드 등을 정의합니다.
- 타입 안전성 보장
- IDE 자동완성 지원
- 오타 방지
"""
from __future__ import annotations

from enum import Enum


class OrderStatus(Enum):
    """
    주문 상태
    
    주문의 생명주기를 나타냅니다.
    정상 흐름: PAID → PICKED_UP → MOVING → PICKING → MOVING_TO_PACK → PACKING → PACKED
    실패 흐름: FAIL_PICKUP, FAIL_PACK
    """
    PAID = "PAID"                          # 결제 완료 (생성 직후)
    PICKED_UP = "PICKED_UP"                # 로봇 작업 시작
    MOVING = "MOVING"                      # 상품 위치로 이동 중
    PICKING = "PICKING"                    # 상품 피킹 중
    MOVING_TO_PACK = "MOVING_TO_PACK"      # 포장대로 이동 중
    PACKING = "PACKING"                    # 포장 중
    PACKED = "PACKED"                      # 포장 완료
    FAIL_PICKUP = "FAIL_PICKUP"            # 피킹 실패
    FAIL_PACK = "FAIL_PACK"                # 포장 실패


class RobotStatus(Enum):
    """
    로봇 상태
    
    Pickee/Packee 공통 상태 정의
    """
    IDLE = "IDLE"                          # 대기 중
    WORKING = "WORKING"                    # 작업 중
    MOVING = "MOVING"                      # 이동 중
    CHARGING = "CHARGING"                  # 충전 중
    ERROR = "ERROR"                        # 오류 발생
    OFFLINE = "OFFLINE"                    # 오프라인


class RobotType(Enum):
    """로봇 타입"""
    PICKEE = "pickee"                      # 피킹 로봇
    PACKEE = "packee"                      # 포장 로봇


class ErrorCode(Enum):
    """
    에러 코드
    
    인터페이스 문서(App_vs_Main.md)에 정의된 에러 코드
    """
    # === 시스템 에러 (SYS) ===
    SYSTEM_ERROR = "SYS_001"               # 시스템 일반 오류
    
    # === 인증 에러 (AUTH) ===
    INVALID_CREDENTIALS = "AUTH_001"       # 잘못된 인증 정보
    USER_NOT_FOUND = "AUTH_002"            # 사용자 없음
    UNAUTHORIZED = "AUTH_003"              # 권한 없음
    
    # === 주문 에러 (ORDER) ===
    ORDER_NOT_FOUND = "ORDER_001"          # 주문 없음
    INVALID_ORDER_STATE = "ORDER_002"      # 잘못된 주문 상태
    ORDER_CREATE_FAILED = "ORDER_003"      # 주문 생성 실패
    
    # === 로봇 에러 (ROBOT) ===
    ROBOT_NOT_AVAILABLE = "ROBOT_001"      # 로봇 사용 불가
    ROBOT_TASK_FAILED = "ROBOT_002"        # 로봇 작업 실패
    ROBOT_TIMEOUT = "ROBOT_003"            # 로봇 응답 타임아웃
    
    # === 상품 에러 (PROD) ===
    PRODUCT_NOT_FOUND = "PROD_001"         # 상품 없음
    INSUFFICIENT_STOCK = "PROD_002"        # 재고 부족
    PRODUCT_SEARCH_FAILED = "PROD_003"     # 검색 실패


class MessageType(Enum):
    """
    API 메시지 타입
    
    App ↔ Main 간 TCP 통신 메시지 타입
    참고: docs/InterfaceSpecification/App_vs_Main.md
    """
    # === 사용자 관리 ===
    USER_LOGIN = "user_login"
    USER_LOGIN_RESPONSE = "user_login_response"
    USER_LOGOUT = "user_logout"
    USER_LOGOUT_RESPONSE = "user_logout_response"
    
    # === 상품 검색 ===
    PRODUCT_SEARCH = "product_search"
    PRODUCT_SEARCH_RESPONSE = "product_search_response"
    
    # === 주문 관리 ===
    ORDER_CREATE = "order_create"
    ORDER_CREATE_RESPONSE = "order_create_response"
    ORDER_CANCEL = "order_cancel"
    ORDER_CANCEL_RESPONSE = "order_cancel_response"
    
    # === 상품 선택 (피킹 중) ===
    PRODUCT_SELECTION = "product_selection"
    PRODUCT_SELECTION_RESPONSE = "product_selection_response"
    
    # === 쇼핑 종료 ===
    SHOPPING_END = "shopping_end"
    SHOPPING_END_RESPONSE = "shopping_end_response"
    
    # === 영상 스트림 ===
    VIDEO_STREAM_START = "video_stream_start"
    VIDEO_STREAM_START_RESPONSE = "video_stream_start_response"
    VIDEO_STREAM_STOP = "video_stream_stop"
    VIDEO_STREAM_STOP_RESPONSE = "video_stream_stop_response"
    
    # === 알림 (Push) ===
    ROBOT_MOVING_NOTIFICATION = "robot_moving_notification"
    ROBOT_ARRIVAL_NOTIFICATION = "robot_arrival_notification"
    CART_UPDATE_NOTIFICATION = "cart_update_notification"
    PACKING_START_NOTIFICATION = "packing_start_notification"
    PACKING_COMPLETE_NOTIFICATION = "packing_complete_notification"
    
    # === 관리자 API ===
    ROBOT_STATUS_REQUEST = "robot_status_request"
    ROBOT_STATUS_RESPONSE = "robot_status_response"
    ROBOT_HISTORY_SEARCH = "robot_history_search"
    ROBOT_HISTORY_RESPONSE = "robot_history_response"
    
    # === 재고 관리 (관리자) ===
    INVENTORY_SEARCH = "inventory_search"
    INVENTORY_SEARCH_RESPONSE = "inventory_search_response"
    INVENTORY_CREATE = "inventory_create"
    INVENTORY_CREATE_RESPONSE = "inventory_create_response"
    INVENTORY_UPDATE = "inventory_update"
    INVENTORY_UPDATE_RESPONSE = "inventory_update_response"
    INVENTORY_DELETE = "inventory_delete"
    INVENTORY_DELETE_RESPONSE = "inventory_delete_response"


class EventTopic(Enum):
    """
    EventBus 토픽
    
    내부 모듈 간 이벤트 발행/구독용 토픽
    """
    # === 앱으로 푸시할 이벤트 ===
    APP_PUSH = "app_push"                  # 앱으로 푸시 알림
    
    # === 주문 이벤트 ===
    ORDER_CREATED = "order_created"        # 주문 생성됨
    ORDER_STATE_CHANGED = "order_state_changed"  # 주문 상태 변경
    ORDER_COMPLETED = "order_completed"    # 주문 완료
    ORDER_FAILED = "order_failed"          # 주문 실패
    
    # === 로봇 이벤트 ===
    ROBOT_MOVING = "robot_moving"          # 로봇 이동 시작
    ROBOT_ARRIVED = "robot_arrived"        # 로봇 도착
    ROBOT_TASK_COMPLETED = "robot_task_completed"  # 로봇 작업 완료
    ROBOT_ERROR = "robot_error"            # 로봇 오류


# === 기타 상수 ===

# 기본 타임아웃 (초)
DEFAULT_TIMEOUT = 30.0
API_TIMEOUT = 5.0
ROS_SERVICE_TIMEOUT = 1.0
LLM_TIMEOUT = 1.5

# 재시도 횟수
MAX_RETRIES = 3
LLM_MAX_RETRIES = 2

# DB 페이징
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000

# 로봇 관련
MAX_PICKEE_ROBOTS = 10
MAX_PACKEE_ROBOTS = 5

