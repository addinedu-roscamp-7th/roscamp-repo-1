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
    로봇 상위 레벨 상태
    
    대시보드 요약 및 일반적인 상태 분류용
    Pickee/Packee 공통 상태 정의
    """
    IDLE = "IDLE"                          # 대기 중
    WORKING = "WORKING"                    # 작업 중
    MOVING = "MOVING"                      # 이동 중
    CHARGING = "CHARGING"                  # 충전 중
    ERROR = "ERROR"                        # 오류 발생
    OFFLINE = "OFFLINE"                    # 오프라인


class PickeeDetailedStatus(Enum):
    """
    Pickee 세부 상태
    
    로봇 제어 및 상세 모니터링용
    참고: docs/StateDiagram/StateDiagram_Pickee.md
    """
    # 초기화 및 충전
    INITIALIZING = "INITIALIZING"                      # 초기화 중
    CHARGING_UNAVAILABLE = "CHARGING_UNAVAILABLE"      # 충전 중 (작업 불가)
    CHARGING_AVAILABLE = "CHARGING_AVAILABLE"          # 충전 중 (작업 가능)
    
    # 쇼핑 시나리오 (SC_02)
    MOVING_TO_SHELF = "MOVING_TO_SHELF"                # 상품 위치로 이동 중
    DETECTING_PRODUCT = "DETECTING_PRODUCT"            # 상품 인식 중
    WAITING_SELECTION = "WAITING_SELECTION"            # 사용자 선택 대기 중
    PICKING_PRODUCT = "PICKING_PRODUCT"                # 상품 피킹 중
    MOVING_TO_PACKING = "MOVING_TO_PACKING"            # 포장대로 이동 중
    WAITING_HANDOVER = "WAITING_HANDOVER"              # 장바구니 전달 대기 중
    MOVING_TO_STANDBY = "MOVING_TO_STANDBY"            # 대기 장소로 이동 중
    
    # 재고 보충 시나리오 (SC_06)
    REGISTERING_STAFF = "REGISTERING_STAFF"            # 직원 등록 중
    FOLLOWING_STAFF = "FOLLOWING_STAFF"                # 직원 추종 중
    MOVING_TO_WAREHOUSE = "MOVING_TO_WAREHOUSE"        # 창고로 이동 중
    WAITING_LOADING = "WAITING_LOADING"                # 적재 대기 중
    WAITING_UNLOADING = "WAITING_UNLOADING"            # 하차 대기 중


class PackeeDetailedStatus(Enum):
    """
    Packee 세부 상태
    
    로봇 제어 및 상세 모니터링용
    참고: docs/StateDiagram/StateDiagram_Packee.md
    """
    INITIALIZING = "INITIALIZING"                      # 초기화 중
    STANDBY = "STANDBY"                                # 작업 대기 중
    CHECKING_CART = "CHECKING_CART"                    # 장바구니 확인 중
    DETECTING_PRODUCTS = "DETECTING_PRODUCTS"          # 상품 인식 중
    PLANNING_TASK = "PLANNING_TASK"                    # 작업 계획 중
    PACKING_PRODUCTS = "PACKING_PRODUCTS"              # 상품 포장 중


# 세부 상태를 상위 레벨 상태로 매핑하는 딕셔너리
DETAILED_TO_GENERAL_STATUS = {
    # Pickee 매핑
    "INITIALIZING": RobotStatus.IDLE,
    "CHARGING_UNAVAILABLE": RobotStatus.CHARGING,
    "CHARGING_AVAILABLE": RobotStatus.CHARGING,
    "MOVING_TO_SHELF": RobotStatus.MOVING,
    "DETECTING_PRODUCT": RobotStatus.WORKING,
    "WAITING_SELECTION": RobotStatus.WORKING,
    "PICKING_PRODUCT": RobotStatus.WORKING,
    "MOVING_TO_PACKING": RobotStatus.MOVING,
    "WAITING_HANDOVER": RobotStatus.WORKING,
    "MOVING_TO_STANDBY": RobotStatus.MOVING,
    "REGISTERING_STAFF": RobotStatus.WORKING,
    "FOLLOWING_STAFF": RobotStatus.WORKING,
    "MOVING_TO_WAREHOUSE": RobotStatus.MOVING,
    "WAITING_LOADING": RobotStatus.WORKING,
    "WAITING_UNLOADING": RobotStatus.WORKING,
    
    # Packee 매핑
    "STANDBY": RobotStatus.IDLE,
    "CHECKING_CART": RobotStatus.WORKING,
    "DETECTING_PRODUCTS": RobotStatus.WORKING,
    "PLANNING_TASK": RobotStatus.WORKING,
    "PACKING_PRODUCTS": RobotStatus.WORKING,
}


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
    ROS_TOPIC_RECEIVED = "ros_topic_received"  # ROS 토픽 수신 (대시보드용)
    ROS_SERVICE_CALLED = "ros_service_called"    # ROS 서비스 호출 (대시보드용)
    ROS_SERVICE_RESPONDED = "ros_service_responded" # ROS 서비스 응답 (대시보드용)


# === 기타 상수 ===

# 기본 타임아웃 (초)
DEFAULT_TIMEOUT = 30.0
API_TIMEOUT = 5.0
ROS_SERVICE_TIMEOUT = 1.0
ROS_SERVICE_FALLBACK_TIMEOUT = 5.0  # ROS 서비스 타임아웃 폴백 값
LLM_TIMEOUT = 1.5
TCP_READ_TIMEOUT = 5.0  # TCP 연결 읽기 타임아웃

# 재시도 횟수
MAX_RETRIES = 3
LLM_MAX_RETRIES = 2

# DB 페이징
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000

# 로봇 관련
MAX_PICKEE_ROBOTS = 10
MAX_PACKEE_ROBOTS = 5

# 스트리밍 관련
STREAMING_SESSION_TIMEOUT = 30.0  # 스트리밍 세션 만료 시간 (초)
STREAMING_CLEANUP_INTERVAL = 10.0  # 세션 정리 주기 (초)
STREAMING_FRAME_BUFFER_SIZE = 10  # 프레임 버퍼 최대 크기

# 대시보드 관련
DASHBOARD_UPDATE_INTERVAL = 1.0  # 대시보드 갱신 주기 (초)
MAX_TOPIC_LOG_ENTRIES = 100  # 토픽 모니터 최대 로그 수
GUI_QUEUE_TIMEOUT = 0.0  # GUI 큐 타임아웃 (0.0 = non-blocking)
GUI_SHUTDOWN_TIMEOUT = 1.0  # GUI 종료 대기 시간 (초)

# 이벤트 루프 관련
ROS_SPIN_INTERVAL = 0.1  # ROS 스핀 주기 (초)

