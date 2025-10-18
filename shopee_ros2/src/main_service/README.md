# Shopee Main Service

ROS2 패키지로 구현된 Shopee 중앙 백엔드 서비스입니다. App, 로봇(Pickee/Packee), LLM 서비스 간의 모든 통신을 중계하고 비즈니스 로직을 총괄하는 핵심 컴포넌트입니다.

##  주요 기능

- **사용자 인증**: App 사용자의 로그인 및 인증을 처리합니다.
- **주문 관리**: 상품 검색, 주문 생성, 상품 선택 등 고객의 쇼핑 전 과정을 관리합니다.
- **로봇 오케스트레이션**:
    - 주문에 따라 Pickee 로봇에게 피킹(picking) 작업을 지시합니다.
    - 피킹이 완료되면 Packee 로봇에게 포장(packing) 작업을 지시합니다.
    - 로봇의 상태(이동, 도착, 작업 완료 등)를 실시간으로 수신하여 처리합니다.
- **LLM 연동**: 자연어 상품 검색 요청을 LLM 서비스를 통해 SQL 쿼리로 변환하여 처리합니다.
- **실시간 알림**: 로봇의 상태 변경 등 주요 이벤트를 App에 실시간으로 알립니다.
- **영상 스트리밍**: App의 요청에 따라 특정 로봇의 카메라 영상을 UDP를 통해 실시간으로 중계합니다.
- **ROS2 서비스 제공**: 로봇이 상품의 위치를 조회할 수 있도록 `/main/get_product_location` 서비스를 제공합니다.
- **모니터링 대시보드**: PyQt6 기반 GUI로 시스템 성능, 로봇 상태, 주문 진행 상황을 실시간으로 모니터링합니다 (v5.0 탭 구조).

## 아키텍처

- **하이브리드 이벤트 루프**: `asyncio`와 `rclpy`를 함께 사용하여, 비동기 웹 요청(TCP/UDP)과 ROS2 통신을 동시에 효율적으로 처리합니다.
- **모듈형 설계**: `APIController`, `RobotCoordinator`, `StreamingService`, `OrderService` 등 기능별로 클래스를 분리하여 유지보수성과 확장성을 높였습니다.
- **통신 인터페이스**:
    - **TCP (포트 5000):** App과의 명령/응답 및 알림 채널
    - **UDP (포트 6000):** 로봇 → App 영상 데이터 중계 채널
    - **ROS2:** Pickee/Packee 로봇과의 토픽/서비스 통신 채널
    - **HTTP:** LLM 서비스와의 REST API 통신 채널

## 설치 및 의존성

1.  **워크스페이스 빌드**
    ROS2 워크스페이스의 루트(`ros2_ws`)에서 아래 명령어를 실행하여 모든 패키지를 빌드합니다.
    ```bash
    colcon build
    ```

2.  **의존성 관리**
    - Python 의존성은 `setup.py`의 `install_requires`에 정의되어 있으며, `pip install -e .` 또는 `colcon build` 시 자동으로 설치됩니다.
    - ROS2 패키지 의존성은 `package.xml`에 정의되어 있습니다.

## 설정

서비스 실행에 필요한 설정은 워크스페이스 루트의 `.env` 파일을 통해 관리됩니다 (`/home/addinedu/dev_ws/Shopee/shopee_ros2/.env`). 함께 제공된 `.env.example` 파일을 `.env`로 복사하여 환경에 맞게 수정하십시오.

**주요 환경 변수:**
- `SHOPEE_API_HOST`: API 서버 호스트 (기본: `0.0.0.0`)
- `SHOPEE_API_PORT`: API 서버 TCP 포트 (기본: `5000`)
- `SHOPEE_LLM_BASE_URL`: LLM 서비스 URL (기본: `http://localhost:8000`)
- `SHOPEE_DB_URL`: 데이터베이스 URL (예: `mysql+pymysql://user:pass@host:3306/dbname`)
- `SHOPEE_LOG_LEVEL`: 로그 레벨 (기본: `INFO`)
- `SHOPEE_GUI_ENABLED`: 대시보드 활성화 (기본: `false`, 활성화: `true`)
- `SHOPEE_GUI_SNAPSHOT_INTERVAL`: 대시보드 갱신 주기 초 단위 (기본: `1.0`)

## 실행 방법

1.  워크스페이스 환경 설정 파일을 소싱(source)합니다.
    ```bash
    cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
    source install/setup.bash
    ```

2.  `ros2 run` 명령어로 `main_service_node`를 실행합니다.
    ```bash
    ros2 run main_service main_service_node
    ```

## 모니터링 대시보드

### 활성화 방법

`.env` 파일에서 대시보드를 활성화합니다:
```bash
SHOPEE_GUI_ENABLED=true
SHOPEE_GUI_SNAPSHOT_INTERVAL=1.0  # 1초마다 갱신
```

### 대시보드 구성 (v5.0 탭 구조)

대시보드는 5개의 탭으로 구성되어 있습니다:

1. **개요 (Overview)** ⭐
   - 시스템 성능 메트릭스 (평균 처리 시간, 성공률, 로봇 활용률, 시스템 부하)
   - 활성 로봇 요약 (Pickee/Packee 상태별 분류)
   - 진행 중 주문 요약
   - 최근 알림 (최신 5건)

2. **로봇 상태 (Robot Status)**
   - 전체 로봇 목록 (ID, Type, Status, Battery, Location, Cart, Order)
   - 로봇별 상세 정보 및 오프라인 감지
   - 통계 요약

3. **주문 관리 (Order Management)**
   - 진행 중 주문 목록 (Items, Amount, Progress Bar)
   - 주문 선택 시 상세 정보 및 타임라인
   - 할당된 로봇 정보

4. **시스템 진단 (System Diagnostics)**
   - 에러 및 장애 추적 (최근 실패 주문, 로봇 오류, LLM/ROS 상태)
   - 네트워크 연결 상태 (App 세션, ROS 토픽, DB 커넥션)

5. **이벤트 로그 (Event Log)**
   - 전체 이벤트 히스토리 (최대 200건)
   - 실시간 이벤트 추가

### 특징

- **실시간 업데이트**: 1초 주기로 모든 정보 갱신
- **탭 구조**: 정보를 분류하여 가독성 향상
- **상태바**: App 세션 수, 로봇 수, 진행 중 주문 수, 마지막 갱신 시각 표시
- **독립 실행**: 대시보드 창을 닫아도 메인 서비스는 계속 동작

## 테스트 방법

`main_service`는 단위 테스트와 Mock 컴포넌트를 활용한 통합 테스트를 모두 지원합니다.

### 단위 테스트 (Unit Tests)

핵심 로직의 개별 단위를 테스트합니다. `pytest`를 사용하여 실행할 수 있습니다.

```bash
# main_service 패키지 디렉토리에서 실행
cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws/src/main_service
pytest
```

### 통합 테스트 (Integration Tests)

실제 로봇이나 외부 서비스 없이 `main_service`의 전체 워크플로우를 테스트할 수 있는 가장 효과적인 방법입니다. Mock 컴포넌트(`Mock Robot`, `Mock LLM`)를 사용하여 실제 운영 환경과 유사한 시나리오를 시뮬레이션합니다.

**자세한 실행 방법은 `TEST_GUIDE.md` 파일을 참고하세요.** 이 가이드에는 Mock 컴포넌트 실행, 테스트 클라이언트 사용법, 문제 해결 팁이 상세히 설명되어 있습니다.
