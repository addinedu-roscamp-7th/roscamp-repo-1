# Quick Test Guide - 빠른 테스트 가이드

로봇 없이 Main Service를 테스트하는 가장 간단한 방법입니다.

## 🎯 준비물

- ✅ Python 환경 (이미 설정됨)
- ✅ ROS2 환경 (설치됨)
- ❌ 데이터베이스 (없어도 테스트 가능)
- ❌ 실제 로봇 (Mock으로 대체)
- ❌ LLM API (Mock으로 대체)

## 🚀 실행 순서

### 1. 빌드 (한 번만)

```bash
cd ~/dev_ws/Shopee/ros2_ws
colcon build --packages-select shopee_interfaces shopee_main_service
source install/setup.bash
```

### 2. Mock 환경 실행 (4개 터미널)

#### 터미널 1️⃣ - Mock LLM Server
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/run_mock_llm.py
```

✅ 성공 시: `Mock LLM Server Starting...` 출력

#### 터미널 2️⃣ - Mock Robot Node
```bash
source ~/dev_ws/Shopee/ros2_ws/install/setup.bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/run_mock_robot.py
```

✅ 성공 시: `Mock Robot Node Starting...` 출력

#### 터미널 3️⃣ - Main Service
```bash
source ~/dev_ws/Shopee/ros2_ws/install/setup.bash
ros2 run shopee_main_service main_service_node
```

⚠️ 데이터베이스 에러 무시: DB 연결 실패 메시지가 나와도 계속 진행됨
✅ 성공 시: `Starting Shopee Main Service` 출력

#### 터미널 4️⃣ - Test Client
```bash
cd ~/dev_ws/Shopee/ros2_ws/src/shopee_main_service
python3 scripts/test_client.py
```

✅ 성공 시: 전체 워크플로우가 자동으로 테스트됨

## 📊 예상 결과

### Test Client 출력:
```
============================================================
Starting Full Workflow Test
============================================================

[1] Testing Login...
→ Sent: user_login
← Received: user_login_response
  Result: True

[2] Testing Product Search...
→ Sent: product_search
← Received: product_search_response
  Result: True

[3] Testing Order Creation...
→ Sent: order_create
← Received: order_create_response
  Result: True
  Data: {
    "order_id": 1,
    "robot_id": 1
  }

... (계속)

============================================================
Full Workflow Test Completed!
============================================================
```

### Mock Robot 출력:
```
[INFO] [mock_robot_node]: [MOCK] Start task: Order=1, Robot=1
[INFO] [mock_robot_node]: [MOCK] Moving to section: Location=10, Section=1000
[INFO] [mock_robot_node]: [MOCK] Arrived at section 1000
[INFO] [mock_robot_node]: [MOCK] Detecting products: [1, 2]
[INFO] [mock_robot_node]: [MOCK] Detected 2 products
[INFO] [mock_robot_node]: [MOCK] Processing selection: Product=1, BBox=1
[INFO] [mock_robot_node]: [MOCK] Moving to packaging area
[INFO] [mock_robot_node]: [MOCK] Cart handover complete
[INFO] [mock_robot_node]: [MOCK] Packing complete for order 1
```

## 🔧 개별 컴포넌트 테스트

### Mock LLM만 테스트:
```bash
# 터미널 1
python3 scripts/run_mock_llm.py

# 터미널 2
curl -X POST http://localhost:8000/search_query \
  -H "Content-Type: application/json" \
  -d '{"query": "비건 사과"}'

# 응답: {"where_clause": "(name LIKE '%사과%') AND is_vegan_friendly = true"}
```

### Mock Robot만 테스트:
```bash
source install/setup.bash
ros2 run shopee_main_service mock_robot_node

# 다른 터미널에서
ros2 topic list        # 토픽 목록 확인
ros2 service list      # 서비스 목록 확인
```

### Main Service API 직접 테스트:
```bash
# Main Service 실행 후
nc localhost 5000
{"type": "user_login", "data": {"user_id": "admin", "password": "admin123"}}
# Enter 누르면 응답 확인
```

## ❌ 문제 해결

### "Could not connect to Main Service"
→ Main Service가 실행 중인지 확인 (터미널 3)

### "Service unavailable"
→ Mock Robot Node가 실행 중인지 확인 (터미널 2)

### "LLM query generation failed"
→ Mock LLM Server가 실행 중인지 확인 (터미널 1)
→ 포트 8000이 사용 중인지 확인: `lsof -i :8000`

### 데이터베이스 에러
→ 정상입니다. DB 없이도 대부분 기능 테스트 가능
→ 주문/사용자 데이터는 저장되지 않지만 플로우는 동작함

## 📝 테스트 항목 체크리스트

- [ ] Mock LLM 응답 확인
- [ ] Mock Robot 토픽 발행 확인
- [ ] Main Service TCP 연결 확인
- [ ] 로그인 API 테스트
- [ ] 상품 검색 API 테스트
- [ ] 주문 생성 API 테스트
- [ ] 로봇 이동 이벤트 확인
- [ ] 상품 인식 이벤트 확인
- [ ] 상품 선택 처리 확인
- [ ] 포장 완료 이벤트 확인

## 🎯 다음 단계

Mock 테스트 성공 후:

1. **데이터베이스 연결**
   - PostgreSQL/MySQL 설치
   - `.env` 파일 설정
   - 스키마 생성

2. **실제 LLM 연동**
   - OpenAI/Anthropic API 키 설정
   - `config.py`에서 LLM_BASE_URL 변경

3. **실제 로봇 테스트**
   - Pickee/Packee 하드웨어 연결
   - Mock Robot 대신 실제 로봇 사용

---

**참고**: 더 자세한 내용은 `TEST_GUIDE.md`를 참고하세요.
