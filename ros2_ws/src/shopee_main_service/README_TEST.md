# 🧪 Shopee Main Service - 테스트 가이드

로봇, 데이터베이스, LLM 없이 Main Service를 완전히 테스트할 수 있습니다!

## ✅ 준비 완료!

모든 Mock 컴포넌트가 독립 스크립트로 구현되어 패키지 의존성 문제가 없습니다.

## 🚀 실행 방법

### 1. 빌드 (최초 1회만)

```bash
cd ~/dev_ws/Shopee/ros2_ws
colcon build --packages-select shopee_interfaces shopee_main_service
source install/setup.bash
```

### 2. 테스트 실행 (4개 터미널)

#### 📺 터미널 1 - Mock LLM Server
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/run_mock_llm.py
```
**예상 출력:**
```
╔══════════════════════════════════════════════════════════╗
║             Mock LLM Server Starting                     ║
╚══════════════════════════════════════════════════════════╝

Endpoints:
  POST http://localhost:8000/search_query
  POST http://localhost:8000/detect_intent

Press Ctrl+C to stop
```

#### 🤖 터미널 2 - Mock Robot Node
```bash
source ~/dev_ws/Shopee/ros2_ws/install/setup.bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/run_mock_robot.py
```
**예상 출력:**
```
╔══════════════════════════════════════════════════════════╗
║          Mock Robot Node Starting                        ║
╚══════════════════════════════════════════════════════════╝

Simulating:
  - Pickee Robot (피킹 로봇)
  - Packee Robot (포장 로봇)

Services & Topics initialized
Press Ctrl+C to stop
```

#### 🌐 터미널 3 - Main Service
```bash
source ~/dev_ws/Shopee/ros2_ws/install/setup.bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/run_main_service.py
```
**예상 출력:**
```
Starting Shopee Main Service
Config: API=0.0.0.0:5000, LLM=http://localhost:8000, DB=...
APIController listening on 0.0.0.0:5000
```
⚠️ DB 연결 에러는 무시하세요 (정상)

#### 🧪 터미널 4 - Test Client
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/test_client.py
```

## 📊 테스트 시나리오

### 전체 워크플로우 (기본)
```bash
python3 scripts/test_client.py
```

**테스트 순서:**
1. ✅ 사용자 로그인
2. ✅ 상품 검색 (LLM)
3. ✅ 주문 생성
4. ✅ 영상 스트림 시작
5. ✅ 상품 선택
6. ✅ 쇼핑 종료
7. ✅ 영상 스트림 중지
8. ✅ 재고 검색
9. ✅ 로봇 히스토리 조회
10. ✅ 음성 명령 분석

### 재고 관리 테스트
```bash
python3 scripts/test_client.py inventory
```

**테스트 순서:**
1. ✅ 재고 추가
2. ✅ 재고 검색
3. ✅ 재고 수정
4. ✅ 재고 삭제

## 🎯 예상 로그

### Main Service
```
2025-10-12 - shopee_main_service - INFO - Starting Shopee Main Service
2025-10-12 - shopee_main_service.api_controller - INFO - APIController listening on 0.0.0.0:5000
2025-10-12 - shopee_main_service.order_service - INFO - Creating order for user 'admin' with 2 items
2025-10-12 - shopee_main_service.order_service - INFO - Order 1 created and dispatched to robot 1
```

### Mock Robot
```
2025-10-12 - mock_robot_node - INFO - [MOCK] Start task: Order=1, Robot=1
2025-10-12 - mock_robot_node - INFO - [MOCK] Moving to section: Location=10, Section=1000
2025-10-12 - mock_robot_node - INFO - [MOCK] Arrived at section 1000
2025-10-12 - mock_robot_node - INFO - [MOCK] Detecting products: [1, 2]
2025-10-12 - mock_robot_node - INFO - [MOCK] Detected 2 products
2025-10-12 - mock_robot_node - INFO - [MOCK] Processing selection: Product=1, BBox=1
2025-10-12 - mock_robot_node - INFO - [MOCK] Product 1 selected
2025-10-12 - mock_robot_node - INFO - [MOCK] Moving to packaging area
2025-10-12 - mock_robot_node - INFO - [MOCK] Cart handover complete
2025-10-12 - mock_robot_node - INFO - [MOCK] Packing started: Order=1, Robot=10
2025-10-12 - mock_robot_node - INFO - [MOCK] Packing complete for order 1
```

### Test Client
```
============================================================
Starting Full Workflow Test
============================================================

[1] Testing Login...
→ Sent: user_login
  Data: {"user_id": "admin", "password": "admin123"}
← Received: user_login_response
  Result: True
  Message: Login successful

[3] Testing Order Creation...
→ Sent: order_create
  Data: {"user_id": "admin", "cart_items": [...]}
← Received: order_create_response
  Result: True
  Data: {
    "order_id": 1,
    "robot_id": 1
  }

============================================================
Full Workflow Test Completed!
============================================================
```

## 🛠️ 트러블슈팅

### ❌ "Could not connect to Main Service"
**원인:** Main Service가 실행되지 않음
**해결:** 터미널 3에서 Main Service 실행 확인

### ❌ "Service unavailable"
**원인:** Mock Robot이 실행되지 않음
**해결:** 터미널 2에서 Mock Robot 실행 확인

### ❌ "LLM query generation failed"
**원인:** Mock LLM Server가 실행되지 않음
**해결:**
- 터미널 1에서 Mock LLM 실행 확인
- 포트 8000 충돌 확인: `lsof -i :8000`

### ⚠️ DB 연결 에러
**원인:** 데이터베이스가 설정되지 않음
**상태:** 정상! Mock 환경에서는 DB 없이도 대부분 기능 테스트 가능

## 📋 체크리스트

테스트 전 확인:
- [ ] ROS2 환경 source (`source install/setup.bash`)
- [ ] shopee_interfaces 빌드 완료
- [ ] shopee_main_service 빌드 완료
- [ ] 포트 5000, 8000 사용 가능

테스트 항목:
- [ ] Mock LLM 응답 확인
- [ ] Mock Robot 이벤트 확인
- [ ] 전체 워크플로우 통과
- [ ] 재고 관리 기능 통과

## 🎯 파일 구조

```
scripts/
├── run_mock_llm.py      # Mock LLM Server (독립 실행)
├── run_mock_robot.py    # Mock Robot Node (독립 실행)
└── test_client.py       # TCP API 테스트 클라이언트

shopee_main_service/
├── main_service_node.py # Main Service (ROS2 실행)
└── ...                  # 기타 서비스 모듈
```

## 🚀 빠른 시작 (복사해서 실행)

### 터미널 1
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service && python3 scripts/run_mock_llm.py
```

### 터미널 2
```bash
cd ~/dev_ws/Shopee/ros2_ws && source install/setup.bash && cd src/shopee_main_service && python3 scripts/run_mock_robot.py
```

### 터미널 3
```bash
cd ~/dev_ws/Shopee/ros2_ws && source install/setup.bash && cd src/shopee_main_service && python3 scripts/run_main_service.py
```

### 터미널 4
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service && python3 scripts/test_client.py
```

---

**성공하면 다음 단계:**
1. 데이터베이스 연결 (.env 설정)
2. 실제 LLM API 연동
3. 실제 로봇 하드웨어 테스트
