# Shopee Main Service

ROS2 패키지로 구현된 Shopee 중앙 백엔드 서비스입니다.  
`shopee_interfaces` 메시지를 사용하여 Pickee/Packee 로봇과 통신하고, TCP API를 통해 App과 연결됩니다.

## 📁 모듈 구조

```
shopee_main_service/
├── main_service_node.py       # 메인 진입점 (ROS2 + asyncio 하이브리드 루프)
├── config.py                   # 설정 관리 (환경 변수 지원)
│
├── api_controller.py           # TCP API 서버 (App ↔ Main)
├── robot_coordinator.py        # ROS2 노드 (Main ↔ Pickee/Packee)
│
├── user_service.py             # 사용자 인증 및 정보 관리
├── product_service.py          # 상품 검색 (LLM 연동) 및 재고 관리
├── order_service.py            # 주문 생명주기 및 로봇 오케스트레이션
│
├── database_manager.py         # DB 세션 관리 (SQLAlchemy)
├── event_bus.py                # 내부 이벤트 버스 (Pub/Sub)
├── llm_client.py               # LLM 서비스 HTTP 클라이언트
│
├── constants.py                # 상수 및 Enum 정의
├── exceptions.py               # 커스텀 예외 클래스
├── models.py                   # 데이터 전송 객체 (DTO)
└── utils.py                    # 유틸리티 함수
```

## 🏗️ 아키텍처 다이어그램

### 1. 컴포넌트 다이어그램 (모듈 간 의존성)

Main Service 내부 모듈들의 의존성과 외부 시스템과의 연결을 보여줍니다.

```plantuml
@startuml
!theme plain
skinparam componentStyle rectangle
skinparam backgroundColor #FFFFFF
skinparam component {
    BackgroundColor<<external>> #E8F4F8
    BackgroundColor<<interface>> #FFF4E6
    BackgroundColor<<service>> #F0F8E8
    BackgroundColor<<infra>> #F5F5F5
}

' 외부 시스템
component "Mobile App" as App <<external>> #E8F4F8
component "ROS2\n(Pickee/Packee)" as ROS2 <<external>> #E8F4F8
component "LLM Service" as LLM <<external>> #E8F4F8
component "MySQL\nDatabase" as DB <<external>> #E8F4F8

' 인터페이스 레이어
component "APIController\n(TCP Server)" as API <<interface>> #FFF4E6
component "RobotCoordinator\n(ROS2 Node)" as Robot <<interface>> #FFF4E6

' 비즈니스 로직 레이어
component "UserService" as User <<service>> #F0F8E8
component "ProductService" as Product <<service>> #F0F8E8
component "OrderService" as Order <<service>> #F0F8E8

' 인프라 레이어
component "DatabaseManager" as DBMgr <<infra>> #F5F5F5
component "LLMClient" as LLMClient <<infra>> #F5F5F5
component "EventBus" as EventBus <<infra>> #F5F5F5

' 메인 진입점
component "MainServiceApp\n(main_service_node.py)" as Main #FFE6E6

' 연결: 외부 → 인터페이스
App -down-> API : "TCP/IP\nJSON"
ROS2 -down-> Robot : "ROS2\nTopics/Services"

' 연결: 메인 → 모든 모듈
Main ..> API : "초기화 및\n핸들러 등록"
Main ..> Robot : "초기화"
Main ..> User : "초기화"
Main ..> Product : "초기화"
Main ..> Order : "초기화"
Main ..> DBMgr : "초기화"
Main ..> LLMClient : "초기화"
Main ..> EventBus : "초기화"

' 연결: 인터페이스 → 서비스
API --> User : "로그인/로그아웃"
API --> Product : "상품 검색"
API --> Order : "주문 관리"

' 연결: 서비스 → 인프라
User --> DBMgr : "사용자 조회"
Product --> DBMgr : "상품 조회"
Product --> LLMClient : "자연어 검색"
Order --> DBMgr : "주문 CRUD"
Order --> Robot : "로봇 작업 요청"
Order --> EventBus : "이벤트 발행"

' 연결: 인프라 → 외부
DBMgr --> DB : "SQL\nQueries"
LLMClient --> LLM : "HTTP\nREST API"
Robot --> ROS2 : "서비스 호출"

' 연결: EventBus → API (알림)
EventBus --> API : "푸시 알림\n(Pub/Sub)"

note right of Main
  **진입점**
  - ROS2 + asyncio 통합
  - 모든 모듈 의존성 주입
  - 설정 관리
end note

note bottom of EventBus
  **내부 이벤트 버스**
  - order_created
  - robot_moving
  - robot_arrived
  → APIController가 구독하여
     App으로 푸시 알림
end note

@enduml
```

### 2. 아키텍처 레이어 다이어그램 (계층 구조)

Clean Architecture 원칙에 따른 계층 구조를 보여줍니다.

```plantuml
@startuml
!theme plain
skinparam rectangle {
    BackgroundColor<<layer1>> #E8F4F8
    BackgroundColor<<layer2>> #FFF4E6
    BackgroundColor<<layer3>> #F0F8E8
    BackgroundColor<<layer4>> #F5F5F5
    BackgroundColor<<layer5>> #FFE6E6
}

rectangle "**진입점 & 설정**" <<layer5>> #FFE6E6 {
    rectangle "main_service_node.py\n(MainServiceApp)" as Entry
    rectangle "config.py\n(환경 변수, 설정)" as Config
}

rectangle "**Presentation Layer**\n(외부 통신)" <<layer1>> #E8F4F8 {
    rectangle "api_controller.py\n(TCP API 서버)" as API
    rectangle "robot_coordinator.py\n(ROS2 노드)" as Robot
}

rectangle "**Business Logic Layer**\n(도메인 로직)" <<layer2>> #FFF4E6 {
    rectangle "user_service.py\n(인증/사용자 관리)" as User
    rectangle "product_service.py\n(상품 검색/재고)" as Product
    rectangle "order_service.py\n(주문 생명주기)" as Order
}

rectangle "**Infrastructure Layer**\n(기술적 구현)" <<layer3>> #F0F8E8 {
    rectangle "database_manager.py\n(SQLAlchemy)" as DB
    rectangle "llm_client.py\n(HTTP 클라이언트)" as LLM
    rectangle "event_bus.py\n(Pub/Sub)" as Event
}

rectangle "**Common Layer**\n(공통 요소)" <<layer4>> #F5F5F5 {
    rectangle "constants.py\n(Enum, 상수)" as Constants
    rectangle "exceptions.py\n(커스텀 예외)" as Exceptions
    rectangle "models.py\n(DTO)" as Models
    rectangle "utils.py\n(유틸리티)" as Utils
}

' 계층 간 의존성 (위 → 아래만 가능)
Entry -down-> API
Entry -down-> Robot
Entry -down-> User
Entry -down-> Product
Entry -down-> Order
Entry -down-> Config

API -down-> User
API -down-> Product
API -down-> Order

User -down-> DB
Product -down-> DB
Product -down-> LLM
Order -down-> DB
Order -down-> Robot
Order -down-> Event

Event -up-> API : "역방향\n(Pub/Sub)"

API -down-> Models
User -down-> Models
Product -down-> Models
Order -down-> Models

API -down-> Exceptions
User -down-> Exceptions
Product -down-> Exceptions
Order -down-> Exceptions

API -down-> Constants
Order -down-> Constants
Product -down-> Utils

note right of Entry
  **의존성 주입**
  모든 모듈을 생성하고
  의존성을 주입
end note

note bottom of Event
  **느슨한 결합**
  EventBus를 통해
  모듈 간 결합도 감소
end note

@enduml
```

### 3. 데이터 흐름 다이어그램 (주문 생성 플로우)

주문 생성 요청이 어떻게 처리되는지 보여줍니다.

```plantuml
@startuml
!theme plain
skinparam sequence {
    ArrowColor #2C3E50
    LifeLineBorderColor #2C3E50
    ParticipantBackgroundColor #ECF0F1
    ParticipantBorderColor #34495E
}

actor "Mobile App" as App
participant "APIController" as API
participant "OrderService" as Order
participant "ProductService" as Product
participant "DatabaseManager" as DB
participant "RobotCoordinator" as Robot
participant "EventBus" as Event
participant "LLMClient" as LLM

== 1. 주문 생성 요청 ==
App -> API : TCP: {"type":"order_create",\n"data":{"user_id":"user1","items":[...]}}
activate API

API -> Order : create_order(user_id, items)
activate Order

Order -> DB : validate_user(user_id)
activate DB
DB --> Order : user exists ✓
deactivate DB

Order -> Product : check_stock(product_ids)
activate Product
Product -> DB : SELECT quantity FROM product
activate DB
DB --> Product : stock info
deactivate DB
Product --> Order : stock available ✓
deactivate Product

Order -> DB : INSERT INTO `order`
activate DB
DB --> Order : order_id = 123
deactivate DB

Order -> Event : publish("order_created", {order_id: 123})
activate Event
Event --> API : notify subscribers
deactivate Event

Order --> API : {"order_id": 123, "status": "PAID"}
deactivate Order

API --> App : {"result": true, "data": {"order_id": 123}}
deactivate API

== 2. 로봇 작업 할당 ==
Order -> Robot : assign_pickee_task(order_id, products)
activate Robot

Robot -> Robot : find_available_pickee()
Robot -> Robot : call ROS2 service\n(/pickee/workflow/start_task)

Robot --> Order : task_assigned ✓
deactivate Robot

Order -> DB : UPDATE `order` SET status='PICKED_UP'
activate DB
DB --> Order : success
deactivate DB

Order -> Event : publish("robot_moving", {order_id: 123, robot_id: 1})
activate Event
Event -> API : notify
API -> App : Push: "로봇이 이동 중입니다"
deactivate Event

== 3. 상품 검색 (LLM 연동) ==
App -> API : TCP: {"type":"product_search",\n"data":{"query":"비건 사과"}}
activate API

API -> Product : search_products("비건 사과")
activate Product

Product -> LLM : POST /detect_intent\n{"text":"비건 사과"}
activate LLM
LLM --> Product : {"intent":"search","product":"사과"}
deactivate LLM

Product -> LLM : POST /generate_search_query\n{"text":"사과"}
activate LLM
LLM --> Product : {"sql_condition":"name LIKE '%사과%'"}
deactivate LLM

Product -> DB : SELECT * FROM product\nWHERE name LIKE '%사과%'\nAND is_vegan_friendly = true
activate DB
DB --> Product : [product1, product2, ...]
deactivate DB

Product --> API : search results
deactivate Product

API --> App : {"result": true, "data": {"products": [...]}}
deactivate API

note over App, LLM
  **핵심 패턴**
  1. **계층 분리**: API → Service → Infrastructure
  2. **이벤트 기반**: EventBus를 통한 비동기 알림
  3. **LLM 통합**: 자연어 검색을 SQL로 변환
  4. **트랜잭션**: DatabaseManager가 세션 관리
end note

@enduml
```

### 주요 모듈 설명

#### 🎯 **main_service_node.py**
- 모든 모듈을 초기화하고 실행
- ROS2와 asyncio를 동시에 실행하는 하이브리드 이벤트 루프
- API 핸들러 등록 (새 API 추가 시 여기서 등록)

#### 🌐 **api_controller.py**
- 포트 5000에서 TCP 서버 실행
- JSON 형식의 요청/응답 처리
- 메시지 타입별 핸들러 라우팅
- EventBus를 통한 알림 푸시

#### 🤖 **robot_coordinator.py**
- ROS2 노드로 로봇과 통신
- **구독 토픽**: `/pickee/robot_status`, `/pickee/moving_status`, `/packee/packing_complete` 등
- **서비스 클라이언트**: `/pickee/workflow/start_task`, `/packee/packing/start` 등
- 로봇 상태 캐싱 및 콜백 지원

#### 👤 **user_service.py**
- 로그인 인증 (`user_id` + 비밀번호)
- 사용자 정보 조회

#### 📦 **product_service.py**
- LLM 기반 자연어 상품 검색
- 재고 조회 및 업데이트
- 알레르기/비건 필터링

#### 🛒 **order_service.py**
- 주문 생성 → 피킹(Pickee) → 포장(Packee) → 완료
- 로봇 이벤트에 따른 상태 전환
- App으로 진행 상황 알림

#### ⚙️ **config.py**
- 환경별 설정 관리 (개발/스테이징/운영)
- 환경 변수 지원 (`SHOPEE_*`)
- 타입 안전한 설정 접근

#### 📋 **constants.py**
- `OrderStatus`: 주문 상태 (PAID, PICKING, PACKED 등)
- `ErrorCode`: 에러 코드 (SYS_001, AUTH_001 등)
- `MessageType`: API 메시지 타입
- `EventTopic`: 내부 이벤트 토픽

#### ⚠️ **exceptions.py**
- `ShopeeException`: 기본 예외 클래스
- `AuthenticationError`, `OrderNotFoundError` 등
- 에러 코드 자동 매핑

#### 📦 **models.py**
- `ApiRequest`, `ApiResponse`: API 공통 포맷
- `OrderInfo`, `ProductInfo`: 데이터 객체
- `LoginRequest`, `CreateOrderRequest` 등
- 타입 안전한 데이터 전송

#### 🔧 **utils.py**
- `retry_async()`: 비동기 재시도
- `format_error_response()`: 에러 응답 생성
- `Timer`: 실행 시간 측정

## 🚀 빌드 및 실행

### 1. 빌드
```bash
cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
colcon build --packages-select shopee_interfaces shopee_main_service
source install/setup.bash
```

### 2. 실행
```bash
# 기본 실행 (기본 설정 사용)
ros2 run shopee_main_service main_service_node

# 환경 변수로 설정 변경
SHOPEE_API_PORT=8080 \
SHOPEE_LLM_URL=http://llm-server:8000 \
SHOPEE_DB_URL=mysql+pymysql://user:pass@dbhost:3306/shopee \
SHOPEE_LOG_LEVEL=DEBUG \
ros2 run shopee_main_service main_service_node
```

**환경 변수:**
- `SHOPEE_API_HOST`: API 서버 호스트 (기본: `0.0.0.0`)
- `SHOPEE_API_PORT`: API 서버 포트 (기본: `5000`)
- `SHOPEE_LLM_URL`: LLM 서버 URL (기본: `http://localhost:8000`)
- `SHOPEE_DB_URL`: 데이터베이스 URL (기본: `mysql+pymysql://shopee:shopee@localhost:3306/shopee`)
- `SHOPEE_LOG_LEVEL`: 로그 레벨 (기본: `INFO`)
- `SHOPEE_LOG_FILE`: 로그 파일 경로 (선택)

### 3. 테스트 (TCP 클라이언트)
```bash
# 로그인 테스트
echo '{"type":"user_login","data":{"user_id":"testuser","password":"1234"}}' | nc localhost 5000

# 상품 검색 테스트
echo '{"type":"product_search","data":{"query":"비건 사과"}}' | nc localhost 5000
```

## 📝 구현 상태

### ✅ 완료 (스켈레톤)
- [x] ROS2 패키지 구조
- [x] 모든 모듈 스켈레톤
- [x] ROS2 + asyncio 통합
- [x] TCP API 서버
- [x] 로봇 통신 인터페이스
- [x] 한국어 주석 (코드 이해용)
- [x] **설정 관리** (config.py)
- [x] **상수/Enum 정의** (constants.py)
- [x] **커스텀 예외** (exceptions.py)
- [x] **DTO 모델** (models.py)
- [x] **유틸리티 함수** (utils.py)

### 🚧 구현 예정 (TODO)
- [ ] **DatabaseManager**: 실제 SQLAlchemy ORM 연동
- [ ] **UserService**: DB 조회 및 비밀번호 해시 검증 (bcrypt)
- [ ] **ProductService**: LLM 연동 및 DB 검색
- [ ] **OrderService**: 주문 생성 및 로봇 워크플로우
- [ ] **APIController**: 클라이언트 레지스트리 및 푸시 알림
- [ ] **LLMClient**: httpx를 이용한 실제 HTTP 요청

## 🔗 참고 문서

- **설계 문서**: `docs/DevelopmentPlan/MainService/MainServiceDesign.md`
- **개발 계획**: `docs/DevelopmentPlan/MainService/MainServicePlan.md`
- **인터페이스**:
  - `docs/InterfaceSpecification/App_vs_Main.md` (TCP API)
  - `docs/InterfaceSpecification/Main_vs_Pic_Main.md` (ROS2)
  - `docs/InterfaceSpecification/Main_vs_Pac_Main.md` (ROS2)
  - `docs/InterfaceSpecification/Main_vs_LLM.md` (HTTP)
- **ERD**: `docs/ERDiagram/ERDiagram.md`

## 💡 다음 단계

### 1단계: DB 연동
```bash
# 1. DB 생성
mysql -u root -p < docs/ERDiagram/CreateTableStatements.sql

# 2. SQLAlchemy 모델 작성 (예정)
# shopee_main_service/db_models.py 생성
```

### 2단계: 핵심 기능 구현
1. **로그인**: `UserService.login()` - DB 조회 및 bcrypt 검증
2. **상품 검색**: `ProductService.search_products()` - LLM 연동
3. **주문 생성**: `OrderService.create_order()` - 주문 상태 머신

### 3단계: 로봇 워크플로우
1. **Pickee 작업 할당**: ROS2 서비스 호출
2. **이벤트 처리**: 로봇 상태에 따른 주문 상태 전환
3. **알림 발송**: EventBus를 통한 푸시 알림

### 4단계: 고급 기능
- LLM 재시도 로직 + Fallback
- 푸시 알림 (클라이언트 레지스트리)
- 메트릭/모니터링

---

## 🎓 사용 예제

### Config 사용
```python
from shopee_main_service.config import MainServiceConfig

# 환경 변수에서 로드
config = MainServiceConfig.from_env()

# 개발 환경용
config = MainServiceConfig.for_development()
```

### 예외 처리
```python
from shopee_main_service.exceptions import OrderNotFoundError, AuthenticationError

try:
    order = await order_service.get_order(order_id)
except OrderNotFoundError as e:
    return e.to_dict()  # {"error_code": "ORDER_001", ...}
```

### 재시도
```python
from shopee_main_service.utils import retry_async

result = await retry_async(
    lambda: llm_client.generate_query(text),
    max_retries=3,
    backoff=0.5
)
```