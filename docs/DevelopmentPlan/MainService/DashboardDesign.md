# Shopee Main Service Monitoring Dashboard 설계서

**작성일**: 2025-10-16
**버전**: 4.0
**담당**: Main Service Team
**용도**: 개발자용 실시간 모니터링 및 관찰

---

## 1. 개요

### 1.1 목표
메인 서비스를 실행할 때 PyQt6 기반 대시보드를 자동으로 함께 기동시켜, 로봇 상태·주문 흐름·이벤트 로그를 실시간으로 확인할 수 있도록 한다. 개발 환경에서 시스템 동작을 관찰하기 위한 **읽기 전용 모니터링 도구**이다.

### 1.2 범위
- **표시**: 로봇 상태, 주문/상태 머신 흐름, 이벤트 로그, 통계 정보
- **제외**: 설정 변경, 주문 조작, Mock 이벤트 발행 등 쓰기 작업
- **환경**: 개발 환경 전용 (운영 환경에서는 기본적으로 비활성화)

---

## 2. 요구사항

| 구분 | 항목 | 설명 |
| --- | --- | --- |
| 기능 | 로봇 상태 | `RobotStateStore` 스냅샷, 상태, 배터리, 예약 정보, 활성 주문 |
| 기능 | 주문 흐름 | 진행 중 주문 목록, 상태 머신 단계, 경과 시간, 할당 로봇 |
| 기능 | 이벤트 로그 | EventBus `app_push`, `robot_failure` 등 이벤트 실시간 표시 |
| 기능 | 통계 정보 | 최근 실패 주문, 예약 타임아웃 모니터 상태 |
| 비기능 | 실행 조건 | `settings.GUI_ENABLED=True`일 때만 GUI 기동 (기본 False) |
| 비기능 | 자원 격리 | GUI 스레드 예외가 Core Service에 영향 주지 않음 |
| 비기능 | 읽기 전용 | 모든 표시는 읽기 전용, 시스템 상태 변경 불가 |

---

## 3. 아키텍처 개요

```
┌───────────────────────────────────────────────────────┐
│ Shopee Main Service 프로세스                           │
│                                                       │
│  ┌─────────────────────┐      ┌────────────────────┐  │
│  │ asyncio + ROS2 Loop │      │ Qt GUI Thread      │  │
│  │ (MainServiceApp)    │◄─────┤ (PyQt6)            │  │
│  └────────┬────────────┘      └────────┬───────────┘  │
│           │                             │             │
│           │ 데이터 수집 (읽기 전용)        │             │
│           ▼                             ▼             │
│   RobotStateStore          DashboardBridge            │
│   OrderService             DashboardController        │
│   EventBus                 DashboardWindow            │
│                            ├─RobotPanel               │
│                            ├─OrderPanel               │
│                            └─EventLogPanel            │
└───────────────────────────────────────────────────────┘
```

### 주요 컴포넌트

1. **DashboardBridge**: asyncio 루프와 Qt GUI 스레드 간 thread-safe 통신
2. **DashboardController**: 주기적 데이터 수집 및 이벤트 포워딩
3. **DashboardDataProvider**: 스냅샷 데이터 수집 헬퍼
4. **DashboardWindow**: PyQt6 메인 윈도우 및 패널 레이아웃

---

## 4. UI 구성

### 4.1 전체 레이아웃

```
┌─────────────────────────────────────────────────────┐
│ Shopee Main Service Dashboard                       │
│ 상태: 연결됨 | 마지막 갱신: 2025-10-16 14:30:25      │
├──────────────────────┬──────────────────────────────┤
│  RobotPanel          │  OrderPanel                  │
│  (로봇 상태 테이블)    │  (진행 중 주문)               │
│                      │                              │
│  ID | Type  | Status │  ID | Status  | Robot | Time │
│  1  | Pickee| IDLE   │  15 | MOVING  | 1     | 30s  │
│  2  | Pickee| WORKING│  16 | PACKING | 3     | 45s  │
│  3  | Packee| IDLE   │                              │
│                      │                              │
├──────────────────────┴──────────────────────────────┤
│  EventLogPanel (이벤트 로그)                         │
│  [14:30:20] robot_moving: Robot 1 → Location 5      │
│  [14:30:15] product_detected: Order 15, 3 products  │
│  [14:30:10] cart_add_success: Order 15, Product 42  │
└─────────────────────────────────────────────────────┘
```

### 4.2 패널 상세

#### RobotPanel (좌측 상단)
- Pickee/Packee 로봇 목록 테이블
- 컬럼: Robot ID, Type, Status, Battery, Reserved, Active Order, Last Update
- 1초 주기 자동 갱신

#### OrderPanel (우측 상단)
- 진행 중 주문 목록 (status < 8)
- 컬럼: Order ID, Customer, Status, Progress(%), Started, Elapsed, Pickee, Packee
- 타임아웃 모니터 활성 여부 표시

#### EventLogPanel (하단 전체)
- `app_push`, `robot_failure`, `reservation_timeout` 등 이벤트 로그
- 최신 200건 표시, 자동 스크롤
- 타임스탬프, 이벤트 타입, 메시지

---

## 5. 데이터 연동

### 5.1 스냅샷 수집 (1초 주기)

`DashboardController`가 다음 데이터를 수집:

```python
# DashboardDataProvider.collect_snapshot()
{
    'orders': {
        'orders': [...],  # OrderService.get_active_orders_snapshot()
        'summary': {...}
    },
    'robots': [...],      # RobotStateStore.list_states()
    'metrics': {
        'failed_orders': [...]  # OrderService.get_recent_failed_orders()
    }
}
```

### 5.2 EventBus 연동

- `app_push`, `robot_failure` 이벤트를 구독
- `DashboardController._forward_event()`로 GUI에 전달
- `DashboardBridge`를 통해 thread-safe 전송

### 5.3 데이터 흐름

```
asyncio 루프                Qt GUI 스레드
    │                           │
    │  1초마다 collect_snapshot  │
    ├──────────────────────────►│ 테이블 갱신
    │                           │
    │  이벤트 발생               │
    ├──────────────────────────►│ 로그 추가
    │                           │
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

| 단계 | 내용 | 예상 소요 |
| --- | --- | --- |
| 1 | 메인 윈도우 및 레이아웃 구현 (`window.py`) | 1-2시간 |
| 2 | RobotPanel 구현 (로봇 상태 테이블) | 1-2시간 |
| 3 | OrderPanel 구현 (주문 목록) | 1-2시간 |
| 4 | EventLogPanel 구현 (이벤트 로그) | 1시간 |
| 5 | GUI 런처 구현 (`launcher.py`) | 1시간 |
| 6 | 메인 서비스 통합 및 테스트 | 1-2시간 |

**총 예상 시간**: 6-10시간 (1일 이내)

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

## 11. 향후 확장 가능성 (현재 범위 외)

추후 필요 시 추가 가능한 기능:
- 실패한 주문 상세 정보 팝업
- 로봇 상태 필터링 (예: IDLE만 보기)
- 이벤트 로그 필터링 및 검색
- 통계 차트 (주문 처리 시간, 성공률 등)
- 설정 조정 패널 (읽기/쓰기)

**현재는 최소 기능으로 구현하여 빠르게 활용 가능하도록 한다.**
