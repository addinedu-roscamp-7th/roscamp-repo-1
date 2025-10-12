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

서비스 실행에 필요한 설정은 프로젝트 루트의 `.env` 파일을 통해 관리됩니다. 함께 제공된 `.env.example` 파일을 `.env`로 복사하여 환경에 맞게 수정하십시오.

**주요 환경 변수:**
- `SHOPEE_API_HOST`: API 서버 호스트 (기본: `0.0.0.0`)
- `SHOPEE_API_PORT`: API 서버 TCP 포트 (기본: `5000`)
- `SHOPEE_LLM_BASE_URL`: LLM 서비스 URL (기본: `http://localhost:8000`)
- `SHOPEE_DB_URL`: 데이터베이스 URL (예: `mysql+pymysql://user:pass@host:3306/dbname`)
- `SHOPEE_LOG_LEVEL`: 로그 레벨 (기본: `INFO`)

## 실행 방법

1.  워크스페이스 환경 설정 파일을 소싱(source)합니다.
    ```bash
    cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
    source install/setup.bash
    ```

2.  `ros2 run` 명령어로 `main_service_node`를 실행합니다.
    ```bash
    ros2 run shopee_main_service main_service_node
    ```

## 테스트 방법

1.  워크스페이스 환경 설정 파일을 소싱(source)합니다.
    ```bash
    cd /home/jinhyuk2me/dev_ws/Shopee/ros2_ws
    source install/setup.bash
    ```

2.  `pytest`를 실행합니다. `shopee_main_service` 패키지 디렉토리에서 실행하거나, 워크스페이스 루트에서 전체 테스트를 실행할 수 있습니다.
    ```bash
    # shopee_main_service 패키지 디렉토리에서 실행
    cd src/shopee_main_service
    pytest
    ```

현재 총 14개의 단위 테스트가 구현되어 있으며, 모든 테스트가 통과하는 것을 확인했습니다.
