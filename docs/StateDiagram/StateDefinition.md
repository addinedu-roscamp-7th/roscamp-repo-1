# State Definition

Shopee 프로젝트의 로봇 상태(State) 정의 문서입니다.

## 목차
- [State Definition](#state-definition)
  - [목차](#목차)
  - [Pickee State Definition](#pickee-state-definition)
    - [공통 상태 (Common States)](#공통-상태-common-states)
    - [쇼핑 모드 (Shopping Mode)](#쇼핑-모드-shopping-mode)
    - [재고 보충 모드 (Restocking Mode)](#재고-보충-모드-restocking-mode)
  - [Packee State Definition](#packee-state-definition)
  - [State Transition Rules](#state-transition-rules)
    - [Pickee State Transitions](#pickee-state-transitions)
      - [공통 전환 규칙](#공통-전환-규칙)
      - [쇼핑 모드 전환 규칙](#쇼핑-모드-전환-규칙)
      - [재고 보충 모드 전환 규칙](#재고-보충-모드-전환-규칙)
    - [Packee State Transitions](#packee-state-transitions)
  - [버전 히스토리](#버전-히스토리)

---

## Pickee State Definition

### 공통 상태 (Common States)

로봇의 기본적인 충전 및 초기화 상태입니다.

| State ID | 한글명 | 영문명 | 설명 |
|----------|--------|--------|------|
| PK_S00 | 초기화중 | INITIALIZING | 시스템 초기화 중 |
| PK_S01 | 충전중_작업불가 | CHARGING_UNAVAILABLE | 배터리 30% 미만, 작업 불가능 상태로 충전 중 |
| PK_S02 | 충전중_작업가능 | CHARGING_AVAILABLE | 배터리 30% 이상, 작업 가능 상태로 충전 중 |

### 쇼핑 모드 (Shopping Mode)

사용자의 원격 쇼핑 서비스를 위한 상태입니다.

| State ID | 한글명 | 영문명 | 설명 |
|----------|--------|--------|------|
| PK_S10 | 상품위치이동중 | MOVING_TO_SHELF | 상품이 있는 매대로 이동 중 |
| PK_S11 | 상품인식중 | DETECTING_PRODUCT | 매대 도착 후 Vision AI가 상품 위치 인식 중 |
| PK_S12 | 상품선택대기중 | WAITING_SELECTION | 신선식품 등 사용자의 수동 선택 대기 중 |
| PK_S13 | 상품담기중 | PICKING_PRODUCT | 매대에서 상품을 픽업하여 장바구니에 담는 중 (Pick & Place) |
| PK_S14 | 포장대이동중 | MOVING_TO_PACKING | 쇼핑 완료 후 포장대로 이동 중 |
| PK_S15 | 장바구니전달대기중 | WAITING_HANDOVER | 포장대에서 장바구니 교체 대기 중 |
| PK_S16 | 대기장소이동중 | MOVING_TO_STANDBY | 작업 완료 후 대기 장소로 이동 중 |

### 재고 보충 모드 (Restocking Mode)

직원의 재고 보충 작업을 지원하는 상태입니다.

| State ID | 한글명 | 영문명 | 설명 |
|----------|--------|--------|------|
| PK_S20 | 직원등록중 | REGISTERING_STAFF | 직원의 정면/후면 외형 특징 등록 중 |
| PK_S21 | 직원추종중 | FOLLOWING_STAFF | Vision을 통해 직원 위치를 추적하며 추종 중 |
| PK_S22 | 창고이동중 | MOVING_TO_WAREHOUSE | 상품 가져오기 요청 받고 창고로 이동 중 |
| PK_S23 | 적재대기중 | WAITING_LOADING | 창고에서 직원이 상품을 적재할 때까지 대기 중 |
| PK_S24 | 매대이동중 | MOVING_TO_SHELF | 적재 완료 후 저장된 매대(직원) 위치로 복귀 중 |
| PK_S25 | 하차대기중 | WAITING_UNLOADING | 직원 위치 도착 후 직원이 상품을 하차할 때까지 대기 중 |

---

## Packee State Definition

포장 작업을 수행하는 Packee 로봇의 상태입니다.

| State ID | 한글명 | 영문명 | 설명 |
|----------|--------|--------|------|
| PA_S00 | 초기화중 | INITIALIZING | 시스템 초기화 중 |
| PA_S01 | 작업대기중 | STANDBY | 장바구니 도착 및 포장 요청 대기 중 |
| PA_S02 | 장바구니확인중 | CHECKING_CART | 장바구니 유무 확인 및 자세 변경 중 |
| PA_S03 | 상품인식중 | DETECTING_PRODUCTS | Vision을 통해 장바구니 내 상품 위치 인식 중 |
| PA_S04 | 작업계획중 | PLANNING_TASK | Task Allocation, Scheduling, Collision Avoidance 계획 중 |
| PA_S05 | 상품담기중 | PACKING_PRODUCTS | 양팔을 이용하여 상품을 픽업하고 포장 박스에 담는 중 (Pick & Place) |

---

## State Transition Rules

### Pickee State Transitions

#### 공통 전환 규칙
- `PK_S00` → `PK_S01`: 초기화 완료
- `PK_S01` → `PK_S02`: 배터리 30% 진입
- `PK_S02` → `PK_S10`: 쇼핑 시작 (쇼핑 요청)
- `PK_S02` → `PK_S20`: 재고 보충 모드 시작
- `PK_S16` → `PK_S01`: 대기장소 도착

#### 쇼핑 모드 전환 규칙
- `PK_S10` → `PK_S11`: 상품 인식 시작 (매대 도착)
- `PK_S11` → `PK_S13`: 일반 상품 자동 선택 완료
- `PK_S11` → `PK_S12`: 신선식품 수동 선택 시작
- `PK_S12` → `PK_S13`: 사용자 선택 완료
- `PK_S13` → `PK_S14`: 원격쇼핑 종료
- `PK_S13` → `PK_S10`: 다음 매대로 이동 시작
- `PK_S14` → `PK_S15`: 포장대 도착 완료
- `PK_S15` → `PK_S16`: 장바구니 전달 완료

#### 재고 보충 모드 전환 규칙
- `PK_S20` → `PK_S21`: 직원 등록 완료
- `PK_S21` → `PK_S22`: 상품 가져오기 요청
- `PK_S22` → `PK_S23`: 창고 도착 완료
- `PK_S23` → `PK_S24`: 적재 완료
- `PK_S24` → `PK_S25`: 매대 도착 완료
- `PK_S25` → `PK_S21`: 하차 완료, 추가 요청 대기
- `PK_S25` → `PK_S15`: 상품 전달 완료

### Packee State Transitions

- `PA_S00` → `PA_S01`: 초기화 완료
- `PA_S01` → `PA_S02`: 장바구니 교체 완료
- `PA_S02` → `PA_S03`: 장바구니 유무 확인 완료
- `PA_S03` → `PA_S04`: 상품 위치 인식 완료
- `PA_S04` → `PA_S05`: 작업 계획 완료
- `PA_S05` → `PA_S03`: Pick & Place 완료 [다음 상품 존재]
- `PA_S05` → `PA_S01`: 모든 상품 포장 완료

---

## 버전 히스토리

- v1.0 (2025-10-09): 초기 문서 생성
  - Pickee State 정의 (공통/쇼핑/재고보충 모드)
  - Packee State 정의
  - State 전환 규칙 추가
