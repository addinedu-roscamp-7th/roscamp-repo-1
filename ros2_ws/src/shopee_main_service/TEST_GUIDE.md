# Shopee Main Service - 테스트 가이드

로봇과 다른 파트 구현 없이 Main Service를 테스트하는 방법입니다.

## 📋 개요

Mock 컴포넌트를 사용하여 실제 로봇, LLM, 데이터베이스 없이도 Main Service의 모든 기능을 테스트할 수 있습니다.

### Mock 컴포넌트
1. **Mock Robot Node** - Pickee/Packee 로봇 시뮬레이터
2. **Mock LLM Server** - LLM API 시뮬레이터
3. **Test Client** - TCP API 테스트 클라이언트

## 🚀 빠른 시작

### 1. 패키지 빌드

```bash
cd ~/dev_ws/Shopee/ros2_ws
colcon build --packages-select shopee_main_service
source install/setup.bash
```

### 2. 환경 설정

`.env` 파일 생성 (데이터베이스 없이 테스트하려면 이 단계 생략 가능):

```bash
cd src/shopee_main_service
cp .env.example .env
```

### 3. Mock 컴포넌트 실행

**터미널 1 - Mock LLM Server 시작:**
```bash
ros2 run shopee_main_service mock_llm_server
```

**터미널 2 - Mock Robot Node 시작:**
```bash
ros2 run shopee_main_service mock_robot_node
```

**터미널 3 - Main Service 시작:**
```bash
ros2 run shopee_main_service main_service_node
```

### 4. 테스트 실행

**터미널 4 - Test Client 실행:**

전체 워크플로우 테스트 (자동):
```bash
python3 src/shopee_main_service/scripts/test_client.py
```

전체 워크플로우 테스트 (수동 - 단계별):
```bash
python3 scripts/test_client.py -i
```

재고 관리 테스트 (자동):
```bash
python3 scripts/test_client.py inventory
```

재고 관리 테스트 (수동 - 단계별):
```bash
python3 scripts/test_client.py inventory -i
```

**옵션:**
- `-i`, `--interactive`: 인터랙티브 모드 - 각 단계마다 Enter를 눌러야 진행

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

**포트:** 8000 (설정 파일에서 변경 가능)

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
INFO:shopee_main_service.api_controller:→ Received [user_login] from ('127.0.0.1', 54321): {"user_id": "admin", "password": "admin123"}
INFO:shopee_main_service.api_controller:← Sending [user_login_response] result=True (15.3ms): Login successful

INFO:shopee_main_service.api_controller:→ Received [order_create] from ('127.0.0.1', 54321): {"user_id": "admin", "cart_items": [{"product_id": 1, "quantity": 2}]}
INFO:shopee_main_service.api_controller:← Sending [order_create_response] result=True (8.7ms): Order successfully created
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

- `shopee_main_service/mock_robot_node.py` - Mock 로봇
- `shopee_main_service/mock_llm_server.py` - Mock LLM
- `scripts/test_client.py` - 테스트 클라이언트
- `.env.example` - 환경 설정 템플릿

---

**주의**: Mock 컴포넌트는 개발/테스트 전용입니다. 프로덕션 환경에서는 사용하지 마세요.
