# Robot Fleet Management Plan

이 문서는 `shopee_main_service` 패키지에 가용 로봇을 선별·예약·모니터링하는 로봇 관제 로직을 추가하기 위한 설계 제안이다. 현재 주문 생성 시 Pickee 로봇이 고정 ID(1번)으로 할당되고, Packee 역시 단순 가용 여부만 확인하는 수준에 머물러 있어 다중 로봇 환경이나 장애 상황을 처리할 수 없다. 아래 제안은 상태 캐시를 기반으로 한 일관된 자원 관리 계층을 도입해 이 문제를 해결한다.

## 목표

- Pickee/Packee 로봇의 최신 상태를 주기적으로 수집하고, 가용 여부를 빠르게 조회할 수 있는 캐시를 유지한다.
- 주문/포장 작업을 시작하기 전에 로봇을 예약(reserve)하고, 작업 종료 시 해제(release)하여 중복 할당을 방지한다.
- 로봇 상태 변화(오프라인, 오류 등)를 감지해 예약을 무효화하고 대체 로봇을 자동으로 선택할 수 있는 훅을 마련한다.
- 기존 모듈(APIController, OrderService, RobotCoordinator, EventBus)와 자연스럽게 연계되도록 계층 구조를 정의한다.

## 전반 구조

```
┌────────────────────┐        ┌────────────────────┐
│  RobotStateStore   │◄───────│  RobotCoordinator  │
│  (새로 추가)        │  상태   │  (ROS 통신)         │
└──────┬─────────────┘        └────────┬───────────┘
       │ 예약/해제 API                        │ 토픽 콜백
┌──────▼─────────────┐        ┌──────────▼─────────┐
│  RobotAllocator    │◄───────│  로봇 상태 메시지   │
│  (전략/정책)         │        └────────────────────┘
└──────┬─────────────┘
       │ 선택 API
┌──────▼─────────────┐
│  OrderService      │
│  (주문/포장 단계)   │
└────────────────────┘
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 | 주요 책임 |
| --- | --- | --- |
| `RobotStateStore` (새 클래스) | 로봇 ID → 상태 데이터 캐시 | 로봇 상태 등록/갱신, 예약 플래그 관리, 타임스탬프 기록, 동시성 제어 |
| `RobotAllocator` (새 클래스) | 로봇 선택 전략 | 상태 스토어에서 후보 로봇을 조회하고, 정책에 따라 하나를 선택/예약 |
| `RobotCoordinator` (기존) | ROS 토픽/서비스 인터페이스 | 상태 토픽 수신 시 `RobotStateStore` 업데이트, 서비스 호출 실패 시 예약 해제 |
| `OrderService` (기존) | 비즈니스 플로우 | 주문 생성 및 장바구니 인계 시 로봇 선택/해제 API 호출, 실패 시 롤백 |

## 상태 모델

### Pickee 상태

```json
{
  "robot_id": 1,
  "type": "pickee",
  "status": "IDLE" | "WORKING" | "ERROR" | "OFFLINE",
  "reserved": false,
  "battery_level": 87.5,
  "active_order_id": 123,
  "last_update": "2024-01-01T12:34:56Z"
}
```

### Packee 상태

Packee 역시 동일 구조를 사용하되 `status` 값과 예약 조건이 다를 수 있다(`AVAILABLE`, `PACKING`, 등). 중복 코드를 줄이기 위해 공통 `RobotState` 데이터 클래스를 두고, 타입·상태 값만 Enum으로 분리한다.

## 예약 알고리즘

### 예약 절차

1. `OrderService.create_order` 호출 시 `RobotAllocator.reserve_pickee(order_context)` 실행.
2. `RobotAllocator` 는 `RobotStateStore`에서 `type=pickee`, `status=IDLE`, `reserved=False` 인 후보 리스트를 가져온다.
3. 특정 전략(예: 라운드 로빈, 최소 작업 횟수, 최대 배터리)을 적용하여 대상으로 삼는다.
4. 선택된 로봇에 대해 `RobotStateStore.try_reserve(robot_id, order_id)` 호출 → 내부적으로 Lock을 사용해 원자적으로 예약.
5. 예약 성공 시 로봇 ID를 반환하고 주문 생성 로직을 지속한다. 실패하면 다른 후보를 시도하거나 `RobotUnavailableError` 발생.

### 해제 절차

- 주문 취소·실패·타임아웃: `RobotStateStore.release(robot_id, order_id)` 호출.
- Pickee 가 장바구니 전달을 마쳤을 때, Packee 예약이 성공하면 Pickee는 `ReturnToBase` 명령과 함께 해제된다.
- `RobotCoordinator` 가 `PickeeRobotStatus` 에서 `status=IDLE` 을 수신하면 자동으로 `active_order_id` 를 지우고 `reserved=False` 로 갱신한다(단, 주문이 아직 진행 중인 상태에서는 OrderService 가 해제를 늦출 수 있도록 플래그로 제어).

## 전략 플러그

다양한 선택 정책을 지원하기 위해 전략 패턴을 적용한다.

```python
class BaseAllocationStrategy(Protocol):
    async def select(self, candidates: list[RobotState], context: AllocationContext) -> RobotState | None: ...
```

기본 전략은 아래를 우선 구현한다.

1. **RoundRobinStrategy** – 마지막으로 할당한 로봇 ID 기억.
2. **LeastWorkloadStrategy** – 진행 중인 주문 수/누적 작업 시간 기반.
3. **BatteryAwareStrategy** – 배터리 임계치 이하 로봇 제외.

전략은 설정(`settings` 또는 `.env`)에서 선택할 수 있도록 하고, 필요 시 런타임 중에도 교체 가능하게 만든다.

## RobotCoordinator 연동

`RobotCoordinator` 의 토픽 콜백에서 `RobotStateStore` 를 갱신한다.

- `PickeeRobotStatus`:
  - `status` 업데이트
  - 배터리/위치 등 부가 정보 저장
  - status 가 `ERROR` / `OFFLINE` 이면 예약 해제 후 EventBus 로 알림 발행
- `PackeeRobotStatus`:
  - 동일하게 처리
- `PickeeCartHandover`:
  - Packee 예약 성공 여부에 따라 Pickee 해제 타이밍 결정

서비스 호출 실패 시(예: ROS 서비스 타임아웃)에도 예약을 원복해야 하므로 `OrderService` 에서 예외를 캐치해 `RobotStateStore.release` 를 호출한다.

## 데이터 지속성

초기 버전에서는 인메모리 캐시로 충분하지만, 여러 프로세스/컨테이너가 OrderService 를 실행한다면 외부 스토리지(예: Redis, PostgreSQL)로 옮겨야 한다. 설계 시 인터페이스를 추상화하여 저장소를 교체할 수 있도록 한다.

```python
class RobotStateRepository(Protocol):
    async def get(self, robot_id: int) -> RobotState | None: ...
    async def save(self, state: RobotState) -> None: ...
    async def reserve(self, robot_id: int, order_id: int) -> bool: ...
```

## 실패 처리 & 재시도

- 예약 이후 실제 명령(예: `/pickee/workflow/start_task`) 이 실패하면 즉시 해제한다.
- 예약 후 일정 시간(예: 5초) 동안 로봇 상태가 `WORKING` 으로 전환되지 않으면 타임아웃 → EventBus 에 알림 발행 + 예약 해제.
- Packee 예약 실패 시, Pickee 예약은 유지하되 사용자가 다시 시도할 수 있도록 알림을 보낸다.
- 로봇이 `ERROR` 상태로 전환되면, 관제 로직이 자동으로 다음 가용 로봇을 찾아 재시도할 수 있도록 훅을 노출한다.

## API 확장

추후 관리자 UI 나 외부 서비스에서 로봇 상태를 조회할 수 있도록 TCP API 또는 ROS 서비스 확장을 고려한다.

- `/main/get_available_robots` ROS 서비스
- `robot_status_request` TCP 메시지에 예약 상태 포함
- 로봇별 작업 히스토리와 연동해 리포트 생성

## 통합 단계

1. **Skeleton 추가**: `RobotStateStore`, `RobotAllocator`, `AllocationStrategy` 인터페이스를 새 파일에 정의한다.
2. **RobotCoordinator 연결**: 토픽 콜백에서 상태 스토어를 갱신하도록 수정한다.
3. **OrderService 연계**: 주문 생성/장바구니 인계/종료 경로에서 예약·해제 호출을 추가하고 에러 처리 경로를 정리한다.
4. **Packee 로직 통합**: `handle_cart_handover` 에서 Packee 예약 로직을 동일 패턴으로 바꾼다.
5. **테스트 작성**: 
   - 상태 스토어 단위 테스트 (예약 충돌, 해제, 상태 갱신)
   - 전략 테스트 (라운드 로빈, 배터리 고려)
   - OrderService 플로우 테스트에서 가용 로봇 유무에 따른 분기 검증
6. **문서/설정 업데이트**: `.env`/`config.py` 에 전략 선택, 타임아웃, 풀 크기 등의 설정을 추가하고 README/TEST_GUIDE 갱신.

## 열려 있는 과제

- 실제 ROS 토픽의 상태 정보가 충분한지 확인 필요 (배터리, 에러 코드 등).
- Packee 의 가용성 판단을 ROS 서비스 호출 대신 상시 상태 모니터링으로 전환할지 결정.
- 멀티 프로세스 환경에서 상태 스토어를 공유하는 방법 결정 (Redis vs. DB).
- 장애 알림을 어떤 채널(앱 push, 관리자 콘솔, ROS 로그)로 보낼지 결정.
- 로봇 유지보수/점검 모드(수동에서 제외) 같은 운영 정책 정의.

위 설계를 바탕으로 단계별 구현을 진행하면 다중 로봇 환경에서도 안정적으로 주문을 배정하고, 로봇 상태에 따라 자동으로 복구하는 관제 시스템을 구축할 수 있다.

