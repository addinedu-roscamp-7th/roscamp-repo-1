# Shopee Main Service Monitoring Dashboard 설계서

**작성일**: 2025-10-17
**버전**: 5.0
**담당**: Main Service Team
**용도**: 개발자 및 운영자용 실시간 모니터링 및 관찰

---

## 1. 개요

### 1.1 목표
메인 서비스를 실행할 때 PyQt6 기반 대시보드를 자동으로 함께 기동시켜, 로봇 상태·주문 흐름·시스템 성능·장애 상황을 실시간으로 확인할 수 있도록 한다. 개발 및 운영 환경에서 시스템 동작을 관찰하고 문제를 조기에 발견하기 위한 **통합 모니터링 도구**이다.

### 1.2 범위
- **표시**: 로봇 상태, 주문/상태 머신 흐름, 시스템 성능 메트릭스, 에러 및 장애 추적, 네트워크 연결 상태, 이벤트 로그
- **제외**: 설정 변경, 주문 조작, Mock 이벤트 발행 등 쓰기 작업
- **환경**: 개발 및 운영 환경 (설정으로 활성화/비활성화 가능, 기본값: 비활성화)

---

## 2. 요구사항

| 구분 | 항목 | 설명 | 우선순위 |
| --- | --- | --- | --- |
| 기능 | 로봇 상태 | `RobotStateStore` 스냅샷, 상태, 배터리, 예약 정보, 활성 주문, 현재 위치, 장바구니 상태 | High |
| 기능 | 주문 흐름 | 진행 중 주문 목록, 상태 머신 단계, 경과 시간, 할당 로봇, 총 상품 수, 총 금액 | High |
| 기능 | 시스템 성능 메트릭스 | 평균 처리 시간, 시간당 처리량, 성공률, 로봇 활용률, 시스템 부하 | High |
| 기능 | 에러 및 장애 추적 | 최근 실패 주문 (사유별 분류), 로봇 오류 현황, LLM 서비스 상태, ROS 서비스 재시도 통계 | High |
| 기능 | 네트워크 상태 | App 연결 세션 수, ROS2 토픽 상태, LLM 응답 시간, 데이터베이스 커넥션 풀 | Medium |
| 기능 | 이벤트 로그 | EventBus `app_push`, `robot_failure` 등 이벤트 실시간 표시 | Medium |
| 기능 | 재고 정보 | 품절 임박 상품, 인기 상품 TOP 5, 예약된 재고량 | Low |
| 비기능 | 실행 조건 | `settings.GUI_ENABLED=True`일 때만 GUI 기동 (기본 False) | - |
| 비기능 | 자원 격리 | GUI 스레드 예외가 Core Service에 영향 주지 않음 | - |
| 비기능 | 읽기 전용 | 모든 표시는 읽기 전용, 시스템 상태 변경 불가 | - |
| 비기능 | 성능 | 1초 주기 스냅샷 수집 시 메인 서비스 지연 최소화 (< 10ms) | - |

---

## 3. 아키텍처 개요

```
┌────────────────────────────────────────────────────────────────┐
│ Shopee Main Service 프로세스                                    │
│                                                                │
│  ┌─────────────────────┐      ┌─────────────────────────────┐ │
│  │ asyncio + ROS2 Loop │      │ Qt GUI Thread (PyQt6)       │ │
│  │ (MainServiceApp)    │◄─────┤                             │ │
│  └────────┬────────────┘      └────────┬────────────────────┘ │
│           │                             │                     │
│           │ 데이터 수집 (읽기 전용)        │                     │
│           ▼                             ▼                     │
│   RobotStateStore          DashboardBridge                    │
│   OrderService             DashboardController                │
│   EventBus                 DashboardWindow                    │
│   ProductService           ├─MetricsPanel (NEW)               │
│   APIController            ├─RobotPanel (ENHANCED)            │
│   DatabaseManager          ├─OrderPanel (ENHANCED)            │
│   RobotCoordinator         ├─ErrorPanel (NEW)                 │
│                            ├─NetworkPanel (NEW)               │
│                            ├─EventLogPanel                    │
│                            └─InventoryPanel (NEW, Optional)   │
└────────────────────────────────────────────────────────────────┘
```

### 주요 컴포넌트

1. **DashboardBridge**: asyncio 루프와 Qt GUI 스레드 간 thread-safe 통신
2. **DashboardController**: 주기적 데이터 수집 및 이벤트 포워딩
3. **DashboardDataProvider**: 스냅샷 데이터 수집 헬퍼
4. **DashboardWindow**: PyQt6 메인 윈도우 및 패널 레이아웃

---

## 4. UI 구성

### 4.1 전체 레이아웃 (탭 구조)

```
┌────────────────────────────────────────────────────────────────────┐
│ Shopee Main Service Dashboard                                      │
│ 상태: 연결됨 | App 세션: 5 | 로봇: 10 | 진행중 주문: 3 | 갱신: 14:30:25│
├────────────────────────────────────────────────────────────────────┤
│ [개요] [로봇 상태] [주문 관리] [시스템 진단] [이벤트 로그]         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  (탭별 컨텐츠 영역)                                                 │
│                                                                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 탭 1: 개요 (Overview) - 기본 화면 ⭐
가장 중요한 정보를 한눈에 파악

```
┌────────────────────────────────────────────────────────────────────┐
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ 시스템 성능 메트릭스                                             │ │
│ │ ┌──────────────┬──────────────┬──────────────┬──────────────┐  │ │
│ │ │평균 처리시간 │ 시간당 처리량 │   성공률     │ 로봇 활용률  │  │ │
│ │ │    42s       │    48건      │   95.2%      │   60%        │  │ │
│ │ │  🟢 정상     │              │   🟢 양호    │  🟢 적정     │  │ │
│ │ └──────────────┴──────────────┴──────────────┴──────────────┘  │ │
│ │ ┌──────────────┐                                                │ │
│ │ │ 시스템 부하  │                                                │ │
│ │ │    15%       │                                                │ │
│ │ │  🟢 여유     │                                                │ │
│ │ └──────────────┘                                                │ │
│ └────────────────────────────────────────────────────────────────┘ │
├──────────────────────────┬─────────────────────────────────────────┤
│  활성 로봇 요약           │  진행 중 주문 요약                      │
│                          │                                         │
│  Pickee: 6/10            │  PAID → MOVING: 2건                     │
│  ├─ WORKING: 4           │  PICKING: 3건                           │
│  ├─ IDLE: 2              │  PACKING: 1건                           │
│  └─ ERROR: 0             │  평균 진행률: 45%                       │
│                          │                                         │
│  Packee: 4/5             │  최근 1시간 완료: 48건                  │
│  ├─ WORKING: 2           │  실패: 3건 (FAIL_PICKUP: 2, FAIL_PACK: 1)│
│  ├─ IDLE: 2              │                                         │
│  └─ OFFLINE: 0           │                                         │
├──────────────────────────┴─────────────────────────────────────────┤
│  최근 알림 (최신 5건)                                               │
│  🔴 [14:30:25] Robot 5 통신 두절 (3분 경과)                        │
│  🟡 [14:28:10] Order 15 처리 지연 (타임아웃 임박)                  │
│  🟢 [14:25:00] Order 14 포장 완료                                  │
│  🟢 [14:23:45] Robot 1 배터리 충전 완료 (100%)                     │
│  🔴 [14:20:30] Order 13 피킹 실패 (상품 미감지)                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 탭 2: 로봇 상태 (Robot Status)
로봇별 상세 정보

```
┌────────────────────────────────────────────────────────────────────┐
│  로봇 상태 모니터링                                    필터: [전체▼]│
├────────────────────────────────────────────────────────────────────┤
│ ID │Type   │Status  │Battery│Location   │Cart │Order│Offline│Updated│
│ 1  │Pickee │WORKING │ 85%🟢│PACKING_A  │Full │ 15  │  -    │ 1s ago│
│ 2  │Pickee │IDLE    │ 92%🟢│HOME       │Empty│  -  │  -    │ 2s ago│
│ 3  │Packee │IDLE    │ 78%🟡│HOME       │Empty│  -  │  -    │ 1s ago│
│ 4  │Pickee │WORKING │ 68%🟡│SHELF_A    │Full │ 16  │  -    │ 3s ago│
│ 5  │Pickee │OFFLINE │ 15%🔴│UNKNOWN    │  ?  │  -  │ 3m    │ 3m ago│
│ 6  │Packee │WORKING │ 95%🟢│PACKING_A  │Full │ 17  │  -    │ 1s ago│
│ 7  │Pickee │ERROR   │ 45%🟡│SHELF_B    │Full │ 18  │  -    │ 5s ago│
│ 8  │Pickee │IDLE    │100%🟢│CHARGING   │Empty│  -  │  -    │ 2s ago│
│ 9  │Packee │IDLE    │ 88%🟢│HOME       │Empty│  -  │  -    │ 1s ago│
│ 10 │Packee │WORKING │ 72%🟡│PACKING_B  │Full │ 19  │  -    │ 2s ago│
├────────────────────────────────────────────────────────────────────┤
│ 통계                                                               │
│ 총 로봇: 10 | 작업중: 4 | 대기중: 4 | 오류: 1 | 오프라인: 1        │
│ 평균 배터리: 73.8% | 저전력 경고: 1대 (Robot 5)                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 탭 3: 주문 관리 (Order Management)
주문별 상세 정보 및 진행도

```
┌────────────────────────────────────────────────────────────────────┐
│  진행 중 주문                                    필터: [진행중▼]    │
├────────────────────────────────────────────────────────────────────┤
│ Order│Customer│Status   │Items│Amount │Progress    │Time│Pickee│Packee│
│ 15   │user_01 │MOVING   │ 5   │₩25000 │██░░░░░░░░ │ 30s│  1   │  -   │
│ 16   │user_02 │PICKING  │ 8   │₩48000 │████░░░░░░ │ 45s│  4   │  -   │
│ 17   │user_03 │PACKING  │ 3   │₩12000 │████████░░ │ 78s│  -   │  6   │
│ 18   │user_04 │PICKING  │12   │₩65000 │███░░░░░░░ │ 22s│  7   │  -   │
│ 19   │user_05 │PACKING  │ 6   │₩38000 │█████████░ │ 95s│  -   │ 10  │
│ 20   │user_06 │PAID     │ 4   │₩18000 │█░░░░░░░░░ │  5s│  -   │  -   │
├────────────────────────────────────────────────────────────────────┤
│ 선택된 주문 상세 정보: Order #15                                    │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ 고객: user_01 | 상태: MOVING | 시작: 14:30:00                  │ │
│ │ 상품: 5개 | 총액: ₩25,000 | 진행률: 20%                        │ │
│ │                                                                │ │
│ │ 타임라인:                                                      │ │
│ │ 14:30:00 ✅ PAID (결제 완료)                                  │ │
│ │ 14:30:05 ✅ Robot #1 할당                                     │ │
│ │ 14:30:10 🔄 MOVING (이동 중...)                               │ │
│ │ 14:30:?? ⏳ PICKING (예정)                                    │ │
│ │                                                                │ │
│ │ 할당 로봇: Pickee #1 (배터리: 85%, 위치: PACKING_A)           │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

#### 탭 4: 시스템 진단 (System Diagnostics) 🔧
에러 추적 및 네트워크 상태

```
┌────────────────────────────────────────────────────────────────────┐
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ 에러 및 장애 추적                                               │ │
│ │                                                                │ │
│ │ 최근 실패 주문 (30분 이내)                                      │ │
│ │ ┌──────────────────┬──────┬─────────────────────────────────┐ │ │
│ │ │ 실패 사유         │ 건수 │ 최근 발생 시각                  │ │ │
│ │ ├──────────────────┼──────┼─────────────────────────────────┤ │ │
│ │ │ FAIL_PICKUP      │  2   │ 14:20:30 (Order #13)            │ │ │
│ │ │ FAIL_PACK        │  1   │ 14:15:45 (Order #10)            │ │ │
│ │ └──────────────────┴──────┴─────────────────────────────────┘ │ │
│ │                                                                │ │
│ │ 로봇 오류 현황                                                  │ │
│ │ ┌──────────────────────────────────────────────────────────┐  │ │
│ │ │ 🔴 Robot #5: OFFLINE (마지막 통신: 3분 전)                │  │ │
│ │ │ 🟡 Robot #7: ERROR (그리퍼 오류, 5초 전)                  │  │ │
│ │ └──────────────────────────────────────────────────────────┘  │ │
│ │                                                                │ │
│ │ LLM 서비스 상태                                                 │ │
│ │ 성공률: 97.5% (39/40) | 평균 응답시간: 850ms | 폴백: 3회      │ │
│ │                                                                │ │
│ │ ROS 서비스 재시도                                               │ │
│ │ 최근 1시간: 12회 (정상 범위)                                   │ │
│ └────────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ 네트워크 및 연결 상태                                           │ │
│ │                                                                │ │
│ │ App 연결 세션: 5 / 200 (2.5%) 🟢 여유                         │ │
│ │ ━━░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                 │ │
│ │                                                                │ │
│ │ ROS2 토픽 상태: 🟢 정상 (5/5 수신)                            │ │
│ │ ├─ /pickee/robot_status: ✅ 1s ago                            │ │
│ │ ├─ /packee/robot_status: ✅ 1s ago                            │ │
│ │ ├─ /pickee/moving_status: ✅ 2s ago                           │ │
│ │ ├─ /packee/packing_complete: ✅ 3s ago                        │ │
│ │ └─ /pickee/product/selection_result: ✅ 1s ago                │ │
│ │                                                                │ │
│ │ LLM 서비스: 🟢 정상 (응답시간: 850ms / 1500ms)                │ │
│ │ ██████████████████████████████████████████░░░░░░░░░░          │ │
│ │                                                                │ │
│ │ DB 커넥션 풀: 3 / 10 (30%) 🟢 여유                            │ │
│ │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                   │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

#### 탭 5: 이벤트 로그 (Event Log)
전체 이벤트 히스토리 및 검색

```
┌────────────────────────────────────────────────────────────────────┐
│  이벤트 로그                                                        │
│  검색: [___________] 필터: [전체▼] 날짜: [오늘▼]                   │
├────────────────────────────────────────────────────────────────────┤
│ Time     │ Level│ Type              │ Message                       │
│──────────┼──────┼───────────────────┼───────────────────────────────│
│ 14:30:25 │ ERROR│ robot_failure     │ Robot 5 통신 두절             │
│ 14:30:20 │ INFO │ robot_moving      │ Robot 1 → PACKING_A 이동 중   │
│ 14:30:15 │ INFO │ product_detected  │ Order 15, 3개 상품 감지       │
│ 14:30:10 │ INFO │ cart_add_success  │ Order 15, Product 42 담기 완료│
│ 14:30:05 │ ERROR│ robot_failure     │ Robot 5 timeout (10s)         │
│ 14:30:00 │ INFO │ order_created     │ Order 15 생성 (user_01)       │
│ 14:29:55 │ INFO │ packing_complete  │ Order 14 포장 완료            │
│ 14:29:50 │ WARN │ llm_fallback      │ LLM 타임아웃, 폴백 검색 사용  │
│ 14:29:45 │ INFO │ robot_arrived     │ Robot 3 HOME 도착             │
│ 14:29:40 │ INFO │ app_push          │ User user_02 알림 전송        │
│ ...                                                                 │
├────────────────────────────────────────────────────────────────────┤
│ 총 200건 표시 중 | 최신순 정렬 | [내보내기 CSV]                     │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 탭별 데이터 요구사항 요약

| 탭 | 주요 데이터 | 수집 주기 | 우선순위 |
|---|------------|----------|---------|
| **개요** | 성능 메트릭스, 로봇/주문 요약, 최근 알림 | 1초 | ⭐ High |
| **로봇 상태** | 전체 로봇 목록, 위치, 배터리, 상태 | 1초 | ⭐ High |
| **주문 관리** | 진행 중 주문, 상세 타임라인 | 1초 | ⭐ High |
| **시스템 진단** | 에러 통계, 네트워크 상태 | 1초 | 🔶 Medium |
| **이벤트 로그** | 전체 이벤트 히스토리 | 즉시 | 🔶 Medium |

---

### 4.3 패널 상세 (기존 참고용)

#### MetricsPanel - ⭐ High Priority
**목적**: 시스템 전체 성능 및 효율성 한눈에 파악
**위치**: 탭 1 (개요) 상단

| 표시 항목 | 설명 | 데이터 소스 | 색상 구분 |
|----------|------|-----------|---------|
| 평균 처리 시간 | 주문 생성 → 포장 완료까지 평균 시간 | `Order.start_time` ~ `Order.end_time` 평균 | 목표 2초 이하: 녹색, 2~5초: 노랑, 5초 이상: 빨강 |
| 시간당 처리량 | 지난 1시간 동안 완료된 주문 수 | `Order.end_time` 카운트 (최근 1시간) | - |
| 성공률 | 전체 주문 대비 성공한 주문 비율 | `(완료 주문 / 전체 주문) * 100` | 95% 이상: 녹색, 90~95%: 노랑, 90% 미만: 빨강 |
| 로봇 활용률 | 작업 중인 로봇 / 전체 로봇 비율 | `(WORKING 로봇 수 / 전체 로봇 수) * 100` | 60% 이상: 녹색, 30~60%: 노랑, 30% 미만: 빨강 |
| 시스템 부하 | 진행 중 주문 수 / 최대 세션 수 | `(진행 중 주문 / 200) * 100` | 70% 미만: 녹색, 70~90%: 노랑, 90% 이상: 빨강 |

**갱신 주기**: 1초

---

#### RobotPanel (좌측 중단) - ENHANCED 🔧
**목적**: 각 로봇의 상태 및 작업 현황 모니터링

| 컬럼 | 설명 | 데이터 소스 | 추가/변경 |
|------|------|-----------|----------|
| Robot ID | 로봇 고유 ID | `RobotState.robot_id` | 기존 |
| Type | PICKEE / PACKEE | `RobotState.robot_type` | 기존 |
| Status | IDLE / WORKING / ERROR / OFFLINE | `RobotState.status` | 기존 |
| Battery | 배터리 잔량 (%) | `RobotState.battery_level` | 기존 (색상: <20% 빨강, 20~50% 노랑) |
| Location | 현재 위치 | `RobotState.current_location` | **NEW** (SR_19 요구사항) |
| Cart | 장바구니 상태 (Empty/Full) | `RobotState.cart_status` | **NEW** (SR_19 요구사항) |
| Reserved | 예약 여부 | `RobotState.reserved` | 기존 |
| Order ID | 현재 처리 중인 주문 | `RobotState.active_order_id` | 기존 |
| Offline Time | 마지막 통신 후 경과 시간 | `now - RobotState.last_update` | **NEW** (타임아웃 감지) |

**갱신 주기**: 1초
**특수 표시**: OFFLINE 상태 로봇은 빨간색 배경

---

#### OrderPanel (우측 중단) - ENHANCED 🔧
**목적**: 진행 중 주문의 상세 정보 및 진행도 추적

| 컬럼 | 설명 | 데이터 소스 | 추가/변경 |
|------|------|-----------|----------|
| Order ID | 주문 고유 ID | `Order.order_id` | 기존 |
| Customer | 고객 ID | `Order.customer_id` | 기존 |
| Status | 주문 상태 (PAID ~ PACKED) | `Order.order_status` | 기존 |
| Items | 주문 상품 개수 | `COUNT(OrderItem)` | **NEW** (작업 복잡도 파악) |
| Amount | 주문 총액 (₩) | `SUM(OrderItem.price * quantity)` | **NEW** (매출 모니터링) |
| Progress | 진행률 (10~100%) | `status_progress_map` | 기존 (진행 바 추가) |
| Started | 주문 시작 시간 | `Order.start_time` | 기존 |
| Elapsed | 경과 시간 | `now - Order.start_time` | 기존 |
| Pickee | 할당된 Pickee 로봇 ID | `order_service._pickee_assignments` | 기존 |
| Packee | 할당된 Packee 로봇 ID | `order_service._packee_assignments` | 기존 |
| Timeout | 예약 타임아웃까지 남은 시간 | `30s - elapsed` | **NEW** (30초 타임아웃 카운트다운) |

**갱신 주기**: 1초
**필터**: `status < 8` (PACKED 전까지)

---

#### ErrorPanel (중하단) - NEW ⭐ High Priority
**목적**: 시스템 장애 및 에러 조기 발견

| 표시 항목 | 설명 | 데이터 소스 |
|----------|------|-----------|
| 최근 실패 주문 | 사유별 실패 주문 수 (FAIL_PICKUP, FAIL_PACK 등) | `OrderService.get_recent_failed_orders()` (최근 30분) |
| 로봇 오류 현황 | ERROR 상태 로봇 목록 및 오류 사유 | `RobotState.status == ERROR` |
| 오프라인 로봇 | 통신 두절 로봇 및 경과 시간 | `RobotHealthMonitor` (타임아웃 > 10초) |
| LLM 서비스 상태 | LLM 호출 성공률, 평균 응답 시간, 폴백 발생 횟수 | `ProductService` 메트릭 수집 |
| ROS 재시도 통계 | ROS 서비스 호출 재시도 총 횟수 (최근 1시간) | `RobotCoordinator` 재시도 카운터 |

**갱신 주기**: 1초
**알림**: 신규 에러 발생 시 빨간색 깜빡임

---

#### NetworkPanel (중하단) - NEW 🔶 Medium Priority
**목적**: 네트워크 및 외부 서비스 연결 상태 모니터링

| 표시 항목 | 설명 | 데이터 소스 | 임계값 |
|----------|------|-----------|-------|
| App 세션 수 | 현재 연결된 App 클라이언트 수 / 최대 수 | `APIController.active_connections` | 200 초과 시 경고 |
| ROS2 토픽 상태 | 구독 중인 토픽 메시지 수신 여부 | `RobotCoordinator` 토픽 구독 상태 | 1개라도 미수신 시 경고 |
| LLM 응답 시간 | 최근 LLM 호출 평균 응답 시간 (ms) | `ProductService` LLM 호출 메트릭 | 1500ms 초과 시 경고 |
| DB 커넥션 풀 | 사용 중 커넥션 / 전체 커넥션 | `DatabaseManager` 풀 상태 | 90% 이상 사용 시 경고 |
| 토픽 수신률 | 정상 수신 토픽 수 / 전체 토픽 수 | `RobotCoordinator` | 100% 미만 시 경고 |

**갱신 주기**: 1초

---

#### EventLogPanel (하단) - 기존 유지
**목적**: 실시간 이벤트 추적

| 컬럼 | 설명 |
|------|------|
| Timestamp | 이벤트 발생 시각 (HH:MM:SS) |
| Event Type | `app_push`, `robot_failure`, `reservation_timeout` 등 |
| Message | 이벤트 상세 내용 |
| Metadata | Order ID, Robot ID 등 추가 정보 |

**갱신 주기**: 즉시 (이벤트 발생 시)
**최대 저장**: 200건 (FIFO)

---

#### InventoryPanel (Optional) - 🔵 Low Priority
**목적**: 재고 관리 및 인기 상품 파악

| 표시 항목 | 설명 | 데이터 소스 |
|----------|------|-----------|
| 품절 임박 상품 | 재고 10개 이하 상품 목록 | `Product.stock <= 10` |
| 인기 상품 TOP 5 | 오늘/이번주 가장 많이 주문된 상품 | `OrderItem` 집계 (최근 24시간) |
| 예약된 재고량 | 진행 중 주문에 예약된 재고 총량 | `InventoryService` 예약 정보 |

**갱신 주기**: 5초 (느린 주기 허용)
**구현 우선순위**: 낮음 (v2.0 이후 고려)

---

## 5. 데이터 연동

### 5.1 스냅샷 수집 (1초 주기)

`DashboardController`가 다음 데이터를 수집:

```python
# DashboardDataProvider.collect_snapshot() - ENHANCED
{
    # 기존 데이터
    'orders': {
        'orders': [...],  # OrderService.get_active_orders_snapshot()
        'summary': {...}
    },
    'robots': [...],      # RobotStateStore.list_states()

    # 확장된 메트릭스 데이터
    'metrics': {
        # 성능 메트릭스
        'avg_processing_time': 42.5,        # 평균 처리 시간 (초)
        'hourly_throughput': 48,            # 시간당 처리량
        'success_rate': 95.2,               # 성공률 (%)
        'robot_utilization': 60.0,          # 로봇 활용률 (%)
        'system_load': 15.0,                # 시스템 부하 (%)

        # 에러 및 장애
        'failed_orders': [...],             # OrderService.get_recent_failed_orders()
        'failed_orders_by_reason': {        # 사유별 실패 주문 수
            'FAIL_PICKUP': 2,
            'FAIL_PACK': 1,
        },
        'error_robots': [...],              # ERROR 상태 로봇 목록
        'offline_robots': [...],            # OFFLINE 로봇 목록 (타임아웃)
        'llm_stats': {
            'success_rate': 97.5,
            'avg_response_time': 850,       # ms
            'fallback_count': 3,
        },
        'ros_retry_count': 12,              # ROS 재시도 총 횟수

        # 네트워크 상태
        'network': {
            'app_sessions': 5,              # 현재 App 세션 수
            'app_sessions_max': 200,        # 최대 세션 수
            'ros_topics_healthy': True,     # ROS 토픽 정상 여부
            'ros_topic_health': {           # 토픽별 수신 상태
                '/pickee/robot_status': True,
                '/packee/robot_status': True,
                # ...
            },
            'llm_response_time': 850,       # ms
            'db_connections': 3,            # 사용 중 커넥션
            'db_connections_max': 10,       # 최대 커넥션
            'topic_receive_rate': 100.0,    # 토픽 수신률 (%)
        },

        # 재고 정보 (Optional)
        'inventory': {
            'low_stock_products': [...],    # 품절 임박 상품
            'popular_products': [...],      # 인기 상품 TOP 5
            'reserved_stock': 120,          # 예약된 재고 총량
        }
    }
}
```

### 5.2 메트릭스 수집 구현 상세

#### 5.2.1 성능 메트릭스
```python
# OrderService 내 메서드 추가
def get_performance_metrics(self, time_window_hours: int = 1) -> dict:
    """
    시스템 성능 메트릭스를 계산한다.

    Args:
        time_window_hours: 통계 수집 시간 범위 (시간)

    Returns:
        평균 처리 시간, 처리량, 성공률 등
    """
    now = datetime.now()
    window_start = now - timedelta(hours=time_window_hours)

    # 완료된 주문 조회 (최근 N시간)
    completed_orders = self._db_session.query(Order).filter(
        Order.order_status == 8,  # PACKED
        Order.end_time >= window_start
    ).all()

    # 평균 처리 시간 계산
    if completed_orders:
        processing_times = [
            (o.end_time - o.start_time).total_seconds()
            for o in completed_orders if o.end_time
        ]
        avg_processing_time = sum(processing_times) / len(processing_times)
    else:
        avg_processing_time = 0

    # 시간당 처리량
    hourly_throughput = len(completed_orders)

    # 성공률 계산
    all_orders = self._db_session.query(Order).filter(
        Order.start_time >= window_start
    ).all()
    success_rate = (len(completed_orders) / len(all_orders) * 100) if all_orders else 0

    # 로봇 활용률
    all_robots = self._robot_state_store.list_states()
    working_robots = [r for r in all_robots if r.status == 'WORKING']
    robot_utilization = (len(working_robots) / len(all_robots) * 100) if all_robots else 0

    # 시스템 부하
    active_orders = len(self.get_active_orders_snapshot())
    system_load = (active_orders / 200 * 100)  # 최대 200 세션 기준

    return {
        'avg_processing_time': round(avg_processing_time, 1),
        'hourly_throughput': hourly_throughput,
        'success_rate': round(success_rate, 1),
        'robot_utilization': round(robot_utilization, 1),
        'system_load': round(system_load, 1),
    }
```

#### 5.2.2 네트워크 상태 수집
```python
# APIController 내 메서드 추가
def get_connection_stats(self) -> dict:
    """현재 활성 App 연결 통계를 반환한다."""
    return {
        'app_sessions': len(self._active_connections),
        'app_sessions_max': 200,
    }

# RobotCoordinator 내 메서드 추가
def get_topic_health(self) -> dict:
    """ROS2 토픽별 수신 상태를 반환한다."""
    now = datetime.now()
    status_topics = {
        '/pickee/robot_status': self._last_pickee_status_time,
        '/packee/robot_status': self._last_packee_status_time,
        '/pickee/moving_status': self._last_pickee_move_time,
    }
    event_topics = {
        '/pickee/arrival_notice': self._last_pickee_arrival_time,
        '/pickee/product/selection_result': self._last_pickee_selection_time,
        '/pickee/cart_handover_complete': self._last_pickee_handover_time,
        '/packee/availability_result': self._last_packee_availability_time,
        '/packee/packing_complete': self._last_packee_complete_time,
        # ...
    }

    status_health = {}
    for topic, last_time in status_topics.items():
        if self._status_timeout <= 0:
            status_health[topic] = True
            continue
        if not last_time:
            status_health[topic] = False
            continue
        elapsed = (now - last_time).total_seconds()
        status_health[topic] = elapsed <= self._status_timeout

    event_activity = {}
    for topic, last_time in event_topics.items():
        if not last_time:
            event_activity[topic] = {
                'seconds_since_last': None,
                'overdue': False,
            }
            continue
        elapsed = (now - last_time).total_seconds()
        event_activity[topic] = {
            'seconds_since_last': round(elapsed, 1),
            'overdue': self._event_timeout > 0 and elapsed > self._event_timeout,
        }

    healthy_count = sum(1 for healthy in status_health.values() if healthy)
    total = len(status_health) or 1
    return {
        'ros_topics_healthy': all(status_health.values()) if status_health else True,
        'ros_topic_health': status_health,
        'topic_receive_rate': healthy_count / total * 100,
        'event_topic_activity': event_activity,
    }

- 상태 토픽 타임아웃은 `settings.ROS_STATUS_HEALTH_TIMEOUT`, 이벤트 토픽 타임아웃은 `settings.ROS_EVENT_TOPIC_TIMEOUT`을 따른다.
- 이벤트 토픽은 헬스 지표에서 제외하고, 마지막 수신 시각과 지연 여부만 네트워크 패널에 노출한다.

# DatabaseManager 내 메서드 추가
def get_pool_stats(self) -> dict:
    """데이터베이스 커넥션 풀 통계를 반환한다."""
    pool = self._engine.pool
    return {
        'db_connections': pool.checkedout(),
        'db_connections_max': pool.size(),
    }
```

### 5.3 EventBus 연동

- `app_push`, `robot_failure` 이벤트를 구독
- `DashboardController._forward_event()`로 GUI에 전달
- `DashboardBridge`를 통해 thread-safe 전송

### 5.4 데이터 흐름

```
asyncio 루프                           Qt GUI 스레드
    │                                      │
    │  1초마다 collect_snapshot (확장)     │
    │  - 기존: orders, robots              │
    │  - NEW: metrics, network, errors     │
    ├─────────────────────────────────────►│ MetricsPanel 갱신
    │                                      │ RobotPanel 갱신 (확장)
    │                                      │ OrderPanel 갱신 (확장)
    │                                      │ ErrorPanel 갱신 (NEW)
    │                                      │ NetworkPanel 갱신 (NEW)
    │                                      │
    │  이벤트 발생                          │
    ├─────────────────────────────────────►│ EventLogPanel 추가
    │                                      │
```

---

## 6. 안전성 및 종료 처리

### 6.1 예외 격리
- GUI 스레드 예외는 로그로만 남기고 메인 서비스는 계속 동작
- `try-except`로 각 패널 갱신 로직 보호

### 6.2 종료 처리
1. 메인 서비스 종료 시 `DashboardController.stop()` 호출
2. 이벤트 리스너 제거
3. 브릿지 종료 (`DashboardBridge.close()`)
4. Qt 이벤트 루프 종료

### 6.3 GUI 창 닫기
- 사용자가 창 닫기 버튼 클릭 시 GUI만 종료 (메인 서비스는 계속 실행)
- `closeEvent()`에서 적절히 처리

---

## 7. 구현 세부사항

### 7.1 의존성
```bash
pip install PyQt6>=6.9.0
```

### 7.2 파일 구조
```
shopee_main_service/
├── dashboard/
│   ├── __init__.py
│   ├── controller.py       # DashboardController, Bridge, DataProvider
│   ├── window.py           # DashboardWindow (메인 윈도우)
│   ├── panels.py           # RobotPanel, OrderPanel, EventLogPanel
│   └── launcher.py         # start_dashboard_gui() 함수
```

### 7.3 실행 흐름

```python
# main_service_node.py의 MainServiceApp.run()
async def run(self):
    self._install_handlers()
    self._robot.set_asyncio_loop(asyncio.get_running_loop())

    # 대시보드 시작
    if settings.GUI_ENABLED:
        await self._start_dashboard_controller()
        start_dashboard_gui(self._dashboard_controller)  # 별도 스레드

    await self._api.start()

    try:
        while rclpy.ok():
            rclpy.spin_once(self._robot, timeout_sec=0.1)
            await asyncio.sleep(0)
    finally:
        if self._dashboard_controller:
            await self._dashboard_controller.stop()
        # ... 정리
```

### 7.4 GUI 부트스트랩

```python
# dashboard/launcher.py
import threading
from PyQt6.QtWidgets import QApplication

def start_dashboard_gui(controller: DashboardController):
    """
    별도 스레드에서 Qt GUI를 실행한다.

    Args:
        controller: 초기화된 DashboardController 인스턴스
    """
    def gui_thread_main():
        app = QApplication([])
        window = DashboardWindow(controller.bridge)
        window.show()
        app.exec()

    thread = threading.Thread(target=gui_thread_main, daemon=True)
    thread.start()
```

---

## 8. 테스트

### 8.1 수동 테스트
1. `.env`에 `SHOPEE_GUI_ENABLED=1` 설정
2. 메인 서비스 실행: `ros2 run shopee_main_service main`
3. 대시보드 창이 자동으로 표시되는지 확인
4. 로봇 상태, 주문 목록이 실시간으로 갱신되는지 확인
5. 이벤트 로그가 정상적으로 추가되는지 확인
6. 창 닫기 후에도 메인 서비스가 계속 실행되는지 확인

### 8.2 자동 테스트 (선택)
- `pytest-qt`를 이용한 GUI 컴포넌트 테스트
- `DashboardController` 유닛 테스트

---

## 9. 구현 일정

### Phase 1: 기존 기능 유지 (완료)
| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 1.1 | 메인 윈도우 및 레이아웃 구현 (`window.py`) | ✅ 완료 |
| 1.2 | RobotPanel 구현 (로봇 상태 테이블) | ✅ 완료 |
| 1.3 | OrderPanel 구현 (주문 목록) | ✅ 완료 |
| 1.4 | EventLogPanel 구현 (이벤트 로그) | ✅ 완료 |
| 1.5 | GUI 런처 구현 (`launcher.py`) | ✅ 완료 |
| 1.6 | 메인 서비스 통합 및 테스트 | ✅ 완료 |

### Phase 2: 탭 기반 UI 구조 개편 (v5.0)

#### 2.1 기본 인프라 (3-4시간) ⭐ High
| 세부 작업 | 예상 소요 |
|----------|----------|
| QTabWidget 기반 메인 레이아웃 구현 | 1시간 |
| 탭별 위젯 클래스 구조 설계 (OverviewTab, RobotTab 등) | 1시간 |
| 상태바 확장 (App 세션, 로봇 수, 진행중 주문 수) | 1시간 |
| 탭 간 데이터 공유 메커니즘 구현 | 1시간 |

#### 2.2 탭 1: 개요 (Overview) 구현 (4-5시간) ⭐ High
| 세부 작업 | 예상 소요 |
|----------|----------|
| 시스템 성능 메트릭스 카드 UI | 1.5시간 |
| OrderService.get_performance_metrics() 구현 | 1.5시간 |
| 활성 로봇/주문 요약 위젯 | 1시간 |
| 최근 알림 목록 (최신 5건) | 1시간 |
| 색상 임계값 적용 | 30분 |

#### 2.3 탭 2: 로봇 상태 (Robot Status) 구현 (3-4시간) ⭐ High
| 세부 작업 | 예상 소요 |
|----------|----------|
| 로봇 테이블 위젯 (Location, Cart 컬럼 추가) | 1.5시간 |
| Offline Time 계산 로직 | 1시간 |
| 상태별 색상 표시 (OFFLINE 빨간색 배경) | 1시간 |
| 필터링 기능 (전체/IDLE/WORKING/ERROR) | 1시간 |
| 하단 통계 요약 | 30분 |

#### 2.4 탭 3: 주문 관리 (Order Management) 구현 (4-5시간) ⭐ High
| 세부 작업 | 예상 소요 |
|----------|----------|
| 주문 테이블 (Items, Amount, Progress Bar 추가) | 2시간 |
| 주문 선택 시 상세 정보 패널 | 1.5시간 |
| 타임라인 UI 구현 | 1.5시간 |
| 필터링 기능 (진행중/완료/실패) | 1시간 |

#### 2.5 탭 4: 시스템 진단 (System Diagnostics) 구현 (5-6시간) 🔶 Medium
| 세부 작업 | 예상 소요 |
|----------|----------|
| 에러 및 장애 추적 패널 UI | 1.5시간 |
| 실패 주문 사유별 집계 로직 | 1시간 |
| LLM 메트릭 수집 (ProductService) | 1.5시간 |
| 네트워크 상태 패널 UI | 1시간 |
| APIController.get_connection_stats() | 30분 |
| RobotCoordinator.get_topic_health() | 1시간 |
| DatabaseManager.get_pool_stats() | 30분 |
| 프로그레스 바 및 임계값 표시 | 1시간 |

#### 2.6 탭 5: 이벤트 로그 (Event Log) 구현 (2-3시간) 🔶 Medium
| 세부 작업 | 예상 소요 |
|----------|----------|
| 기존 EventLogPanel을 탭으로 이동 | 1시간 |
| 검색 기능 추가 (키워드, Order ID, Robot ID) | 1시간 |
| 필터 기능 (전체/INFO/WARN/ERROR) | 1시간 |
| CSV 내보내기 기능 | 30분 |

#### 2.7 데이터 수집 로직 확장 (3-4시간) ⭐ High
| 세부 작업 | 예상 소요 |
|----------|----------|
| DashboardDataProvider.collect_snapshot() 확장 | 2시간 |
| 메트릭스 데이터 통합 및 검증 | 1시간 |
| 성능 최적화 (캐싱, 쿼리 최적화) | 1시간 |

**Phase 2 총 예상 시간**: 24-31시간 (3-4일)

---

### Phase 3: 고급 기능 추가 (선택사항)
탭 구조가 완성된 후 필요 시 추가

### Phase 4: 저우선순위 기능 (v2.0 이후)
| 단계 | 내용 | 예상 소요 | 우선순위 |
| --- | --- | --- | --- |
| 4.1 | **InventoryPanel 구현** | 4-5시간 | 🔵 Low |
|     | - ProductService.get_inventory_stats() | 2시간 | |
|     | - 인기 상품 집계 로직 | 1.5시간 | |
|     | - UI 레이아웃 | 1.5시간 | |
| 4.2 | **타임라인 뷰** | 8-10시간 | 🔵 Low |
|     | - 주문 진행 타임라인 차트 | 4시간 | |
|     | - 로봇 스케줄 Gantt 차트 | 4-6시간 | |

**Phase 4 총 예상 시간**: 12-15시간 (2-3일)

---

### 전체 구현 로드맵

```
v4.0 (현재)          v5.0 (Phase 2)              v6.0+ (Phase 3-4)
기본 단일 화면  ─►   5개 탭 구조로 개편      ─►   고급 기능 추가
(완료)               탭1: 개요 (성능 요약)        - 실시간 차트
                     탭2: 로봇 상태               - 타임라인 뷰
                     탭3: 주문 관리               - 알림/경고
                     탭4: 시스템 진단             - 데이터 내보내기
                     탭5: 이벤트 로그             - 원격 접속 지원
                     (3-4일)                      (필요시)
```

**탭별 구현 우선순위**:
1. **1차**: 기본 인프라 + 탭1(개요) + 탭2(로봇) + 탭3(주문) → 핵심 모니터링 기능
2. **2차**: 탭4(시스템 진단) → 장애 조기 발견
3. **3차**: 탭5(이벤트 로그) → 상세 분석 및 디버깅

**권장 구현 순서**:
1. Phase 2-1: 인프라 구축 (QTabWidget, 상태바)
2. Phase 2-2~2.4: 핵심 탭 3개 완성 (개요, 로봇, 주문)
3. **중간 테스트**: 실전 환경에서 1주일 사용
4. Phase 2-5~2.7: 나머지 탭 및 데이터 확장
5. **최종 검증**: 성능 측정 및 피드백 수렴
6. Phase 3-4: 필요성 검증 후 선택적 진행

---

## 10. 배포/운영

### 10.1 개발 환경
```bash
# .env
SHOPEE_GUI_ENABLED=1
SHOPEE_GUI_SNAPSHOT_INTERVAL=1.0
```

### 10.2 운영 환경
```bash
# .env
SHOPEE_GUI_ENABLED=0  # 또는 설정하지 않음 (기본값)
```

### 10.3 README 문서화
- 대시보드 활성화 방법
- 화면 구성 설명
- 문제 해결 가이드

---

## 11. 향후 확장 가능성

### v5.0에서 구현 (Phase 2 + 3)
- ✅ **시스템 성능 메트릭스 패널**: 평균 처리 시간, 성공률, 로봇 활용률
- ✅ **에러 및 장애 추적 패널**: 실패 주문 분석, LLM/ROS 상태 모니터링
- ✅ **네트워크 상태 패널**: App 세션, ROS 토픽, DB 커넥션 상태
- ✅ **로봇/주문 패널 확장**: 위치, 장바구니, 총 금액 등 추가 정보

### v6.0 이후 고려 사항 (Phase 4+)
추후 필요 시 추가 가능한 기능:

#### 인터랙티브 기능
- **주문 상세 팝업**: 실패한 주문 클릭 시 상세 정보 모달 표시
- **로봇 상세 팝업**: 로봇 클릭 시 작업 이력, 배터리 히스토리 차트
- **필터링 기능**:
  - 로봇 상태별 필터 (IDLE만 보기, WORKING만 보기)
  - 주문 상태별 필터
  - 시간 범위 필터 (최근 1시간/24시간/1주일)
- **이벤트 로그 검색**: 키워드, Order ID, Robot ID로 검색

#### 고급 시각화
- **실시간 차트**:
  - 시간대별 처리량 추이 (라인 차트)
  - 주문 상태 분포 (파이 차트)
  - 로봇 배터리 히스토리 (영역 차트)
- **타임라인 뷰**:
  - 주문 진행 타임라인 (Gantt 차트)
  - 로봇 스케줄 시각화
- **히트맵**:
  - 시간대별 시스템 부하 히트맵
  - 로봇별 작업량 히트맵

#### 알림 및 경고
- **소리 알림**: 치명적 에러 발생 시 알림음
- **데스크톱 알림**: 시스템 트레이 알림
- **이메일/Slack 연동**: 특정 임계값 초과 시 자동 알림

#### 운영 편의 기능
- **설정 조정 패널** (읽기/쓰기):
  - 스냅샷 수집 주기 조정
  - 로그 레벨 변경
  - 로봇 할당 전략 변경
- **데이터 내보내기**:
  - CSV/Excel 형식으로 로그 내보내기
  - 스크린샷 저장 기능
- **원격 접속 지원**:
  - VNC/RDP를 통한 원격 모니터링
  - 웹 기반 대시보드 (Flask/FastAPI + React)

#### 성능 최적화
- **메트릭스 캐싱**: 1초마다가 아닌 변경 감지 기반 갱신
- **가상 스크롤**: 로봇/주문 목록이 많을 때 성능 개선
- **백그라운드 집계**: 무거운 통계 쿼리는 별도 스레드에서 수행

---

## 12. 요구사항 추적표

| 요구사항 ID | 설명 | 구현 패널 | Phase |
|-----------|------|----------|-------|
| SR_17 | 작업 정보 모니터링 | OrderPanel | Phase 1 ✅ |
| SR_19 | 로봇 상태 조회 | RobotPanel | Phase 1 ✅ |
| SR_19 (확장) | 로봇 위치/장바구니 상태 | RobotPanel (ENHANCED) | Phase 2 |
| - | 시스템 성능 목표 달성 여부 | MetricsPanel | Phase 2 |
| - | 장애 조기 발견 | ErrorPanel | Phase 2 |
| - | 네트워크 건강성 체크 | NetworkPanel | Phase 3 |
| SR_21 | 상품 정보 조회 | InventoryPanel | Phase 4 |

---

## 13. 변경 이력

| 버전 | 날짜 | 변경 내용 | 담당자 |
|-----|------|----------|-------|
| 4.0 | 2025-10-16 | 초기 대시보드 구현 완료 (단일 화면) | Main Service Team |
|     |            | - RobotPanel, OrderPanel, EventLogPanel |  |
| 5.0 | 2025-10-17 | **탭 기반 UI 구조로 전면 개편** | Main Service Team |
|     |            | - 5개 탭 구조 설계 (개요/로봇/주문/진단/로그) |  |
|     |            | - 탭1: 시스템 성능 메트릭스 + 요약 대시보드 |  |
|     |            | - 탭2: 로봇 상태 모니터링 (Location, Cart 추가) |  |
|     |            | - 탭3: 주문 관리 + 타임라인 뷰 |  |
|     |            | - 탭4: 에러 추적 + 네트워크 진단 |  |
|     |            | - 탭5: 이벤트 로그 + 검색/필터 기능 |  |
|     |            | - 데이터 수집 로직 확장 설계 |  |
|     |            | - Phase별 구현 일정 재수립 (24-31시간) |  |

---

**현재 상태**: v5.0 탭 구조 설계 완료, Phase 2 구현 대기 중

**설계 변경 배경**:
- 기존 단일 화면 방식은 정보량이 많아 가독성 저하
- 사용자 피드백: 탭 구조로 정보를 분류하여 집중도 향상 필요
- 향후 확장성을 고려하여 탭 추가가 용이한 구조로 개편

**주요 개선점**:
1. **정보 밀도 최적화**: 탭별로 관련 정보 그룹화
2. **사용자 경험 향상**: 필요한 정보만 선택적으로 확인 가능
3. **확장성 확보**: 새로운 기능을 별도 탭으로 추가 용이
4. **성능 개선**: 현재 활성 탭만 렌더링하여 리소스 절감
