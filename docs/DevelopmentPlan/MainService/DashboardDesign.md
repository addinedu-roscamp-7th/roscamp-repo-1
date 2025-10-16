# Shopee Main Service Embedded Dev Dashboard 설계서

**작성일**: 2025-10-16  
**버전**: 3.0  
**담당**: Main Service Team  
**용도**: 개발자용 실시간 관찰 및 디버깅

---

## 1. 개요

### 1.1 목표
메인 서비스를 실행할 때 PyQt6 기반 대시보드를 자동으로 함께 기동시켜, 로봇 상태·주문 흐름·이벤트 로그·설정 파라미터를 바로 확인하고 실험할 수 있도록 한다. 운영 환경이 아닌 개발 환경에서 사용한다.

### 1.2 범위
- **표시**: 로봇 상태, 주문/상태 머신 흐름, 이벤트 로그, LLM 요청 통계, 재시도/헬스체크 현황, 주요 설정 값.
- **조작(개발용)**: mock 이벤트 발행, 특정 주문 강제 진행/취소, 설정 값(예: 배터리 임계치, 재시도 횟수) 임시 조정.
- **제외**: 운영자 승인 기능, 영구 설정 변경, 외부 배포.

---

## 2. 요구사항

| 구분 | 항목 | 설명 |
| --- | --- | --- |
| 기능 | 로봇 상태 | `RobotStateStore` 스냅샷, 헬스 모니터 타임아웃 상태, 현재 할당 주문 |
| 기능 | 주문 흐름 | 진행 중 주문 목록, 상태 머신 단계별 체류 시간, 재시도 횟수 |
| 기능 | 이벤트 로그 | EventBus `app_push` 및 `robot_failure` 이벤트 실시간 표시 |
| 기능 | Mock 컨트롤 | Mock 로봇/LLM 토글, 특정 이벤트 발행, 샘플 주문 생성 버튼 |
| 기능 | 설정 패널 | `settings` 파라미터(예: `ROS_SERVICE_RETRY_ATTEMPTS`)를 세션 한정으로 조정하고 반영 |
| 기능 | LLM/재시도 통계 | 최근 LLM 요청 성공/실패, 서비스 재시도 횟수, 헬스 타임아웃 발생 로그 |
| 기능 | ROS 상태 | Pickee/Packee 토픽 수신 시간, 마지막 상태 메시지, 헬스 타임아웃 카운트 |
| 비기능 | 실행 조건 | `settings.GUI_ENABLED=True`일 때만 GUI 기동 (기본 False) |
| 비기능 | 자원 격리 | GUI 스레드 예외가 Core Service에 영향 주지 않음 |
| 비기능 | 권한 | 개발 환경/Mock 모드에서만 조작 기능 노출 |

---

## 3. 아키텍처 개요

```
┌───────────────────────────────────────────────────────┐
│ Shopee Main Service 프로세스                           │
│                                                       │
│  ┌─────────────────────┐      ┌────────────────────┐  │
│  │ asyncio + ROS2 Loop │      │ Qt GUI Thread      │  │
│  │ (MainServiceApp)    │      │ (PyQt6)            │  │
│  └────────┬────────────┘      └────────┬───────────┘  │
│           │ 공유 객체                    │             │
│           ▼                             ▼             │
│   RobotStateStore (async)      DashboardController     │
│   OrderService                 ├─RobotPanel            │
│   EventBus                     ├─OrderPanel            │
│   Config(Settings)             ├─EventLogPanel         │
│   Mock Services                └─ControlPanel          │
└───────────────────────────────────────────────────────┘
```

- `DashboardController`: 메인 스레드 데이터를 thread-safe 방식으로 읽고 Qt 시그널로 전달.
- `DashboardEventListener`: EventBus에 등록돼 GUI에 이벤트 전달.
- GUI 스레드는 PyQt6 `QApplication` + `QMainWindow` 구성.

---

## 4. UI 및 기능 구성

| 영역 | 위젯 | 상세 |
| --- | --- | --- |
| 상단 상태바 | StatusBar | GUI 연결 여부, 마지막 갱신 시각, 헬스 타임아웃 카운트 |
| 좌측 상단 | RobotPanel | Pickee/Packee 테이블, 상태, battery, active_order_id, 헬스 타임아웃 여부 |
| 좌측 하단 | ROSHealthPanel | 마지막 토픽 수신 시각, 서비스 재시도 통계, 헬스 타임아웃 로그 |
| 중앙 | OrderPanel | 진행 중 주문, 상태 머신 단계, 시작/갱신/완료 시간, 재시도 횟수 |
| 우측 상단 | LLMPanel | 최근 LLM 쿼리, 성공/실패, fallback 사용 여부 |
| 우측 하단 | ControlPanel | Mock 로봇/주문 생성, 특정 이벤트 발행, 설정 값 조절 슬라이더 |
| 하단 전체 | EventLogPanel | `app_push`, `robot_failure`, Mock 이벤트 로그 (최신 200건) |

Mock 관련 조작 버튼은 `settings.MOCK_ENABLED`가 True일 때만 활성화.

---

## 5. 데이터 연동

### 5.1 Robot/Order 스냅샷
- `DashboardController`가 1초 주기로 `RobotStateStore.list_states()`와 주문 헬퍼(예: `OrderService.debug_list_active_orders()`, 필요 시 새 헬퍼 추가)를 호출.
- 이 호출은 `asyncio.run_coroutine_threadsafe` 혹은 thread-safe 큐로 처리해 Qt 스레드와 충돌을 피함.

### 5.2 EventBus 연동
- `EventBus.register_listener`로 GUI 전용 listener를 추가.
- 각 이벤트는 `DashboardController.enqueue_event()`로 전달, Qt 스레드에서 로그 갱신.

### 5.3 설정 조절
- GUI에서 설정 값을 조정하면 `DashboardController.apply_temp_setting(key, value)`가 호출되고, 이는 `settings` 객체의 세션 범위 변수에 반영(예: `ROS_SERVICE_RETRY_ATTEMPTS` 조정).
- 엔지니어가 쉽게 원상 복구할 수 있도록 “Reset Settings” 버튼 제공.

### 5.4 Mock 컨트롤
- `ControlPanel` 버튼이 눌리면 `order_service.create_order()` 등 내부 API를 호출하거나 Mock 로봇 이벤트를 발행 (Mock 모드에서만 허용).
- 허용되지 않는 환경(운영)에서는 버튼 비활성화.

---

## 6. 안전성 및 종료 처리

- GUI 스레드에서 발생한 예외는 로그로만 남기고, 메인 서비스는 계속 동작.
- 메인 서비스 종료 시 `DashboardController.stop()`을 호출해 타이머·listener 제거 후 Qt 이벤트 루프 종료.
- GUI 창에서 닫기 버튼을 누르면 메인 서비스에도 종료 신호를 전달하거나, GUI만 종료할 수 있도록 옵션 제공.

---

## 7. 설정 및 실행

### 7.1 의존성
- `PyQt6>=6.7.0` (이미 개발 환경에서 설치).
- (선택) `pytest-qt` 테스트 목적으로 개발 requirements에 추가.

### 7.2 실행 흐름 (의사코드)

```python
def main():
    rclpy.init()
    app = MainServiceApp()

    if settings.GUI_ENABLED:
        start_dashboard_gui(app)  # Qt 스레드 생성

    asyncio.run(app.run())
```

`start_dashboard_gui` 내부:
- Qt QApplication 생성.
- `DashboardController` 인스턴스에 `state_store`, `event_bus`, `order_service`, `settings` 등을 주입.
- `DashboardWindow`를 띄우고 별도 스레드에서 `app.exec()` 실행.

---

## 8. 테스트

- **수동**: GUI가 자동으로 표시되는지, 각 패널이 갱신되는지, 설정 변경과 Mock 버튼이 실제로 작동하는지 확인.
- **자동(선택)**: `pytest-qt`를 이용해 `DashboardController`의 데이터 처리 로직과 이벤트 처리 로직 테스트.

---

## 9. 일정 및 담당

| 단계 | 내용 | 담당 |
| --- | --- | --- |
| 1 | Qt 스레드 부트스트랩 및 기본 레이아웃 | Main Service Team |
| 2 | Robot/Order/ROS 패널 데이터 연동 | Main Service Team |
| 3 | EventBus listener + 이벤트 로그 | Main Service Team |
| 4 | ControlPanel 기능(설정 조정·Mock) 추가 | Main Service Team |
| 5 | 종료 처리, 테스트 정리 | Main Service Team |

총 소요: 개발자의 경험에 따라 2~3일 예상.

---

## 10. 배포/운영

- 개발 환경에서는 `.env`에 `SHOPEE_GUI_ENABLED=1` 설정.
- 운영 환경에서는 기본 False 유지.
- README에 “개발용 대시보드” 섹션 추가(설정 방법, 기능 요약).

