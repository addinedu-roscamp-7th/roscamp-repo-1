# Shopee Main Service - 테스트 가이드

로봇과 다른 파트 구현 없이 Main Service를 테스트하는 방법입니다.

## 📋 개요

Mock 컴포넌트를 사용하여 실제 로봇, LLM, 데이터베이스 없이도 Main Service의 주요 기능을 테스트할 수 있습니다.

### Mock/테스트 도구
1. **Test Client** (`scripts/test_client.py`) – App ↔ Main Service TCP 흐름을 자동/수동으로 검증
2. **Interactive Mock App** (`scripts/interactive_mock_app.py`) – App 측 메시지를 단계적으로 수동 전송
3. **Mock Robot Node** (`ros2 run main_service mock_robot_node`) – ROS2 상호작용을 자동으로 흉내 내는 기존 일괄 시뮬레이터

## 🚀 빠른 시작

### 1. 패키지 빌드

```bash
cd ~/dev_ws/Shopee/ros2_ws
colcon build --packages-select main_service
source install/setup.bash
```

### 2. 환경 설정

`.env` 파일 생성 (데이터베이스 없이 테스트하려면 이 단계 생략 가능):

```bash
cd src/main_service
cp .env.example .env
```

### 3. Mock 컴포넌트 실행

**터미널 1 - Mock LLM Server 시작 (자동 모드):**
```bash
ros2 run main_service mock_llm_server
```

**터미널 2 - Mock Robot Node 시작 (자동 모드):**
```bash
# Pickee와 Packee를 모두 시뮬레이션
ros2 run main_service mock_robot_node

# Pickee만 모의 (Packee는 실제 노드와 연동 시)
ros2 run main_service mock_pickee_node
# 또는
ros2 run main_service mock_robot_node --mode pickee

# Packee만 모의 (Pickee는 실제 노드와 연동 시)
ros2 run main_service mock_packee_node
# 또는
ros2 run main_service mock_robot_node --mode packee
```

**터미널 3 - Main Service 시작:**
```bash
ros2 run main_service main_service_node
```

### 4. 테스트 실행

**터미널 4 - Test Client 실행:**

전체 워크플로우 테스트 (자동):
```bash
python3 src/main_service/scripts/test_client.py
```

전체 워크플로우 테스트 (수동 - 단계별):
```bash
python3 src/main_service/scripts/test_client.py --interactive
```

텍스트 기반 상품 선택 문장 변경:
```bash
python3 src/main_service/scripts/test_client.py --speech-selection "사과 가져다줘"
```

bbox 기반만 확인하려면:
```bash
python3 src/main_service/scripts/test_client.py --no-speech-selection
```

재고 관리 테스트 (자동):
```bash
python3 scripts/test_client.py inventory
```

재고 관리 테스트 (수동 - 단계별):
```bash
python3 scripts/test_client.py inventory --interactive
```

**옵션:**
- `-i`, `--interactive`: 인터랙티브 모드 - 각 단계마다 Enter를 눌러야 진행
- `--no-speech-selection`: 텍스트 기반 상품 선택 단계를 비활성화하고 bbox 기반으로만 테스트
- `--speech-selection`: 텍스트 기반 선택 시 사용할 문장 지정

### 5. 인터랙티브 Mock 활용

보다 세밀하게 흐름을 조정하고 싶다면 다음 스크립트를 활용하세요.

```bash
# App ↔ Main Service TCP 메시지를 한 단계씩 전송
python3 src/main_service/scripts/interactive_mock_app.py
```

각 스크립트는 명령어 도움말을 콘솔에 안내하며, SequenceDiagram 및 InterfaceSpecification에 명시된 메시지들을 그대로 트리거할 수 있습니다.

## 🔍 상세 가이드

### Mock Robot Node

Pickee와 Packee의 모든 ROS2 인터페이스를 구현한 시뮬레이터입니다.

**제공 기능:**
- ✅ 작업 시작 (Start Task)
- ✅ 섹션 이동 시뮬레이션 (0.5초 후 도착)
- ✅ 상품 인식 시뮬레이션 (0.3초 후 완료)
- ✅ 상품 선택 처리 (0.3초 후 완료)
- ✅ 포장대 이동 (0.5초 후 장바구니 전달)
- ✅ Packee 포장 시뮬레이션 (1초 후 완료)
- ✅ 영상 스트림 시작/중지

**동작 방식:**
- ROS2 서비스 요청을 받으면 즉시 성공 응답
- 타이머를 사용하여 비동기적으로 토픽 발행
- 실제 로봇과 동일한 메시지 시퀀스 재현

### Mock LLM Server

LLM API를 시뮬레이션하는 HTTP 서버입니다.

**제공 기능:**
- ✅ 상품 검색 쿼리 생성 (`POST /search_query`)
- ✅ 음성 명령 인텐트 분석 (`POST /detect_intent`)

**동작 방식:**
- 간단한 규칙 기반 처리
- "비건 사과" → `name LIKE '%사과%' AND is_vegan_friendly = true`
- "사과 가져다줘" → `{"intent": "fetch_product", "entities": {"product_name": "사과"}}`

**포트:** 5001 (설정 파일에서 변경 가능)


### Test Client

Main Service의 TCP API를 테스트하는 클라이언트입니다.

**실행 모드:**

- **자동 모드**: 모든 테스트 단계를 자동으로 순차 실행
- **인터랙티브 모드** (`-i` 옵션): 각 단계마다 Enter를 눌러야 진행
  - 각 단계를 천천히 확인하면서 테스트 가능
  - 로봇/서비스 상태를 확인하기 좋음

**테스트 시나리오:**

1. **전체 워크플로우** (기본):
   - 로그인
   - 상품 검색
   - 주문 생성
   - 영상 스트림 시작
   - 상품 선택
   - 쇼핑 종료
   - 영상 스트림 중지
   - 재고/히스토리 조회
   - 음성 명령

2. **재고 관리** (`inventory` 옵션):
   - 재고 추가
   - 재고 검색
   - 재고 수정
   - 재고 삭제

### 인터랙티브 Mock 스크립트

- `interactive_mock_app.py`: App에서 전송하는 TCP 메시지를 직접 구성하여 Main Service 응답과 비동기 알림을 확인할 수 있습니다.

### 시나리오별 자동화 유틸리티

`main_service/scenario_suite.py`에는 SequenceDiagram 명세에 맞춘 비동기 실행 함수가 정리돼 있습니다. 개별 시나리오만 빠르게 검증하고 싶을 때 활용하세요.

- `run_sc_02_4_product_selection`: 상품 선택 및 장바구니 반영 (SC_02_4)
- `run_sc_02_5_shopping_end`: 쇼핑 종료 플로우 (SC_02_5)
- `run_sc_05_2_1_inventory_search` ~ `run_sc_05_2_4_inventory_delete`: 관리자 재고 관리 시나리오들 (SC_05_2_x)
- `run_sc_05_3_robot_history_search`: 관리자 작업 이력 조회 (SC_05_3)

실행 예시는 아래와 같습니다.

```bash
python3 - <<'PY'
import asyncio
from main_service.scenario_suite import run_sc_02_4_product_selection

asyncio.run(run_sc_02_4_product_selection())
PY
```

CLI 형태로 바로 실행하고 싶다면 `scripts/scenarios` 경로의 실행기를 사용할 수 있습니다.

- `python3 scripts/scenarios/sc_02_4_product_selection.py`
- `python3 scripts/scenarios/sc_02_5_shopping_end.py`
- `python3 scripts/scenarios/sc_05_2_1_inventory_search.py`
- `python3 scripts/scenarios/sc_05_2_2_inventory_update.py`
- `python3 scripts/scenarios/sc_05_2_3_inventory_create.py`
- `python3 scripts/scenarios/sc_05_2_4_inventory_delete.py`
- `python3 scripts/scenarios/sc_05_3_robot_history_search.py`

Mock 환경에서는 비동기 알림을 검증하기 위해 `MainServiceClient.drain_notifications()`가 사용되므로, ROS2 토픽 이벤트와 TCP 응답이 모두 도착할 시간을 확보한 뒤 호출해주세요.

## 🔗 컴포넌트별 통신 테스트 체크리스트 (Main Service 기준)

### Shopee App ↔ Main Service (TCP)
- 명세: `docs/InterfaceSpecification/App_vs_Main.md`
- 도구: `main_service/client_utils.py` (`MainServiceClient`)
- 절차:
  1. `ros2 run main_service main_service_node`
  2. 별도 터미널에서 `python3 -m main_service.client_utils` 또는 시나리오 스크립트 실행
  3. `user_login`, `product_search`, `order_create`, `product_selection`, `shopping_end`, `video_stream_start/stop`, `inventory_*`, `robot_status_request` 등 메시지 전송
  4. `MainServiceClient.drain_notifications()`로 `robot_moving_notification`, `cart_update_notification` 등 푸시 이벤트 수신 확인

### Main Service ↔ LLM 서비스 (HTTP)
- 명세: `docs/InterfaceSpecification/Main_vs_LLM.md`
- 도구: `ros2 run main_service mock_llm_server` 또는 실제 LLM 엔드포인트
- 검증 포인트:
- `LLMClient.generate_search_query("비건 사과")` → SQL WHERE 절 응답
- `LLMClient.extract_bbox_number("2번 집어줘")` → `{"bbox": 2}`
- `LLMClient.detect_intent("피키야, A존으로 이동해줘")` → 이동 의도/엔티티 응답
  - 실패 시 fallback 검색(`ProductService._basic_keyword_search`)이 호출되는지 로그 확인

### Main Service ↔ Pickee Main (ROS2)
- 명세: `docs/InterfaceSpecification/Main_vs_Pic_Main.md`
- 도구:
  - Mock 환경: `ros2 run main_service mock_robot_node` (또는 `mock_pickee_node`)
  - 실제/시뮬레이션 로봇: Pickee Main 노드
- 테스트 항목:
  - `/pickee/workflow/start_task` 서비스 호출 (주문 생성 시 자동)
  - `/pickee/moving_status`, `/pickee/arrival_notice`, `/pickee/product_detected`, `/pickee/product/selection_result`, `/pickee/cart_handover_complete`
  - 각 토픽을 `ros2 topic echo`로 모니터링하면서 OrderService 핸들러 동작(`handle_moving_status`, `handle_arrival_notice` 등) 확인

### Main Service ↔ Packee Main (ROS2)
- 명세: `docs/InterfaceSpecification/Main_vs_Pac_Main.md`
- 도구:
  - Mock 환경: `mock_robot_node` (Packee 흐름 포함) 또는 `mock_packee_node`
  - 실제/시뮬레이션 로봇: Packee Main 노드
- 테스트 항목:
  - `/packee/packing/check_availability`, `/packee/packing/start` 서비스 호출
  - `/packee/packing_complete` 토픽 수신 후 `OrderService.handle_packee_complete`에서 상태 전환/알림 확인

### UDP 영상 스트림 (App ↔ Main)
- 명세: `docs/InterfaceSpecification/App_vs_Main_UDP.md`
- 절차:
  1. UDP 포트 6000에서 수신하는 간단한 socket 스크립트를 준비
  2. App용 TCP 핸들러에서 `video_stream_start` 전송
  3. `/pickee/video_stream/start` 서비스 성공 시 `StreamingService`가 6000/UDP로 프레임 헤더 송신
  4. `video_stream_stop` 호출 후 스트림 중단 확인

### 내부 이벤트/헬스 모니터
- EventBus 토픽: `app_push`, `robot_failure`, `reservation_timeout`
- 확인 방법:
  - `tests/test_dashboard_controller.py` 예시처럼 EventBus에 mock listener 등록
  - `RobotStateStore.list_states()`와 `OrderService.get_active_orders_snapshot()`으로 현재 상태 스냅샷 공유
  - `settings.ROS_STATUS_HEALTH_TIMEOUT`을 줄여 헬스 체크 타임아웃을 빠르게 재현

## 📊 예상 출력

### Mock Robot Node
```
[INFO] [mock_robot_node]: Mock Robot Node initialized
[INFO] [mock_robot_node]: [MOCK] Start task: Order=1, Robot=1
[INFO] [mock_robot_node]: [MOCK] Moving to section: Location=10, Section=1000
[INFO] [mock_robot_node]: [MOCK] Arrived at section 1000
[INFO] [mock_robot_node]: [MOCK] Detecting products: [1, 2]
[INFO] [mock_robot_node]: [MOCK] Detected 2 products
```

### Main Service (로그)
```
INFO:main_service.api_controller:→ Received [user_login] from ('127.0.0.1', 54321): {"user_id": "admin", "password": "admin123"}
INFO:main_service.api_controller:← Sending [user_login_response] result=True (15.3ms): Login successful

INFO:main_service.api_controller:→ Received [order_create] from ('127.0.0.1', 54321): {"user_id": "admin", "cart_items": [{"product_id": 1, "quantity": 2}]}
INFO:main_service.api_controller:← Sending [order_create_response] result=True (8.7ms): Order successfully created
```

### Test Client (자동 모드)
```
→ Sent: user_login
  Data: {"user_id": "admin", "password": "admin123"}
← Received: user_login_response
  Result: True
  Message: Login successful

→ Sent: order_create
  Data: {"user_id": "admin", "cart_items": [...]}
← Received: order_create_response
  Result: True
  Data: {
    "order_id": 1,
    "robot_id": 1
  }
```

### Test Client (인터랙티브 모드)
```
[1] Testing Login...
→ Press Enter to continue... [사용자가 Enter 입력]

→ Sent: user_login
  Data: {"user_id": "admin", "password": "admin123"}
← Received: user_login_response
  Result: True
  Message: Login successful

[2] Testing Product Search...
→ Press Enter to continue... [사용자가 Enter 입력]
...
```

## 🛠️ 트러블슈팅

### 연결 실패
```
✗ Error: Could not connect to Main Service
```
→ Main Service가 실행 중인지 확인

### Mock 컴포넌트 미응답
```
[ERROR] Service /pickee/workflow/start_task unavailable
```
→ Mock Robot Node가 실행 중인지 확인

### LLM 타임아웃
```
[WARNING] LLM query generation failed
```
→ Mock LLM Server가 포트 8000에서 실행 중인지 확인

## 🧪 데이터베이스 없이 테스트

데이터베이스가 없어도 대부분의 기능을 테스트할 수 있습니다:

1. **Mock 환경 사용**: Robot과 LLM Mock만으로 충분
2. **에러 무시**: DB 연결 에러는 발생하지만 테스트는 진행됨
3. **제한사항**:
   - 실제 주문 데이터 저장 안됨
   - 사용자 인증 실패 (Mock 데이터 사용)
   - 재고 조회/수정 불가

## 📝 다음 단계

Mock 환경 테스트 성공 후:

1. **데이터베이스 연결**: PostgreSQL/MySQL 설정
2. **실제 LLM 연동**: OpenAI/Anthropic API 연동
3. **실제 로봇 연동**: Pickee/Packee 하드웨어 연결
4. **통합 테스트**: 전체 시스템 통합

## 🔗 관련 파일

- `main_service/mock_robot_node.py` - Mock 로봇 (Pickee/Packee 선택 가능)
- `main_service/mock_pickee_node.py` - Pickee 전용 Mock 노드
- `main_service/mock_packee_node.py` - Packee 전용 Mock 노드
- `main_service/mock_llm_server.py` - Mock LLM
- `scripts/test_client.py` - 테스트 클라이언트
- `.env.example` - 환경 설정 템플릿

---

**주의**: Mock 컴포넌트는 개발/테스트 전용입니다. 프로덕션 환경에서는 사용하지 마세요.
