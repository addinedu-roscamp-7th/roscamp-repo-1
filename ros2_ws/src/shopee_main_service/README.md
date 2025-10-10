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

---

**현재 상태**: 완벽한 스켈레톤! 기반이 탄탄하여 구현 준비 완료 ✅  
각 모듈의 TODO 주석과 examples를 참고하여 기능을 하나씩 구현하세요!
