# Shopee Main Service

ROS2 패키지로 구현된 Shopee 중앙 백엔드 서비스입니다. App, Pickee/Packee 로봇, LLM 서비스 간 통신을 중계하고 주문·로봇 오케스트레이션을 담당합니다.

## 사전 준비

- **ROS2**: Jazzy (자세한 규정은 `docs/CodingStandard/standard.md` 참고). ROS 환경 설정 및 DDS 네트워크 구성이 선행되어야 합니다.
- **Python**: 3.10 이상 권장. `pip`, `colcon` 등 빌드 도구가 준비돼 있어야 합니다.
- **데이터베이스**: MySQL 8.x. 접속 URI는 `config.py`의 `MainServiceConfig.DB_URL` 값으로 관리하므로 환경에 맞게 수정하거나 환경 변수로 덮어씁니다.
- **LLM 서비스**: `config.py`의 `LLM_BASE_URL`이 가리키는 엔드포인트에서 `search_query`, `bbox`, `intent_detection` API를 제공해야 합니다 (자세한 흐름은 Main Service 설계 문서 참고).
- **시스템 요구사항**: 모듈 책임과 인터페이스는 `docs/DevelopmentPlan/MainService` 하위 설계 문서 내용과 일치해야 합니다.

## 환경 설정

`main_service/config.py`의 `MainServiceConfig` 클래스 기본값을 프로젝트 환경에 맞게 수정합니다. 아래 항목은 빈번히 조정되는 값입니다 (main_service/main_service/config.py:56-104).

```python
class MainServiceConfig(BaseSettings):
    API_HOST: str = '0.0.0.0'
    API_PORT: int = 5000
    DB_URL: str = 'mysql+pymysql://shopee:shopee@localhost:3306/shopee'
    LLM_BASE_URL: str = 'http://localhost:5001'
    ROBOT_ALLOCATION_STRATEGY: str = 'round_robin'
    GUI_ENABLED: bool = True
    GUI_SNAPSHOT_INTERVAL: float = 1.0
    LOG_LEVEL: str = 'INFO'
```

필요하다면 런타임 환경 변수(`export SHOPEE_API_PORT=6000` 등)로 일부 값을 덮어쓸 수 있지만, 기본적인 설정은 코드에 직접 반영한다는 가정으로 유지합니다. 또한 ROS2 관련 환경 변수(`ROS_DOMAIN_ID`, 네임스페이스 등)가 다른 패키지와 충돌하지 않도록 조정하세요.

## 빌드 및 설치

1. 워크스페이스에서 전체 패키지를 빌드합니다.

   ```bash
   cd /home/jinhyuk2me/dev_ws/Shopee/shopee_ros2
   colcon build --packages-select main_service
   ```

2. 빌드 후 환경을 소싱합니다.

   ```bash
   source install/setup.bash
   ```

3. Python 의존성은 `setup.py`의 `install_requires`와 `extras_require`에 정의되어 있으며 `colcon build` 또는 `pip install -e .` 시 자동 설치됩니다.

## 데이터베이스 초기화

`scripts` 폴더에 제공된 스크립트를 사용해 MySQL을 준비합니다 (shopee_ros2/src/main_service/scripts/README.md:21-158).

```bash
cd /home/jinhyuk2me/dev_ws/Shopee/shopee_ros2/src/main_service/scripts
./setup_database.sh        # 최초 1회
# 또는 데이터만 복원할 때
./reset_database.sh
```

- 관리자 계정: `admin / admin123`
- Pickee 2대, Packee 1대 샘플 데이터 포함
- 수동 초기화가 필요하면 `init_schema.sql`, `sample_data.sql`을 직접 실행할 수 있습니다.

## Mock 및 외부 서비스

- **Mock LLM**: `ros2 run main_service mock_llm_server`
- **Mock Robot**: `ros2 run main_service mock_robot_node` (또는 `--mode pickee`, `--mode packee`)
- **대시보드**: `config.py`에서 `GUI_ENABLED=True`로 유지하면 PyQt6 GUI가 활성화됩니다.
- **테스트 클라이언트**: `python3 scripts/test_client.py`로 App↔Main 시나리오를 검증합니다.

Mock 구성 요소는 docs/DevelopmentPlan/MainService/RobotFleetManagement.md:24-89, shopee_ros2/src/main_service/TEST_GUIDE.md:14-109에 정의된 플릿 관리 흐름과 일관되게 동작합니다.

## 실행 절차

1. ROS2 환경 소싱
2. LLM 서버 및 Mock 로봇 노드 기동 (실 로봇 사용 시 해당 노드 대신 실제 ROS 노드를 사용)
3. Main Service 실행

```bash
cd /home/jinhyuk2me/dev_ws/Shopee/shopee_ros2
source install/setup.bash
ros2 run main_service main_service_node
```

서비스 시작 시 `RobotStateStore`가 DB의 로봇 목록을 불러와 `RobotStatus.OFFLINE`으로 초기화합니다 (main_service/main_service/main_service_node.py:98-140). ROS 토픽 수신 후 상태가 갱신되며, GUI가 활성화되어 있으면 대시보드가 따라 올라옵니다.

### 통합 Mock 환경 실행

App 개발자가 빠르게 테스트하려면 아래 런치 파일을 사용할 수 있습니다.

```bash
ros2 launch main_service mock_environment.launch.py \
  api_host:=0.0.0.0 \
  api_port:=5000 \
  llm_base_url:=http://localhost:5001 \
  db_url:=mysql+pymysql://shopee:shopee@localhost:3306/shopee \
  mock_robot_mode:=all
```

- Mock LLM 서버와 Mock Robot 노드가 함께 실행되며, Main Service는 `config.py`와 동일한 환경 변수로 설정됩니다.
- GUI는 기본적으로 비활성화(`SHOPEE_GUI_ENABLED=false`)되어 있으므로 GUI가 필요하면 런치에서 환경 변수를 별도로 지정하세요.
- DB 초기화는 런치 실행 전에 수동으로 완료해야 합니다 (`scripts/setup_database.sh`).

## 모니터링 대시보드

- 5개 탭(개요, 로봇 상태, 주문 관리, 시스템 진단, 이벤트 로그)으로 구성되어 시스템 상태를 1초 주기로 갱신합니다.
- `ROBOT_ALLOCATION_STRATEGY`, `ROS_SERVICE_RETRY_*`, `ROS_STATUS_HEALTH_TIMEOUT` 등 주요 설정 값은 `config.py`를 참고하세요 (main_service/main_service/config.py:56-104).

## 테스트 및 품질 관리

- **단위 테스트**: `pytest`
- **통합 테스트**: Mock LLM/로봇과 함께 `scripts/test_client.py`, `tests/mocks/` 활용
- **시나리오 실행**: `main_service/scenario_suite.py`의 헬퍼 함수를 통해 SequenceDiagram에 정의된 흐름을 재현할 수 있습니다.
- **타입 체크**: `mypy main_service/`
- **커버리지**: `pytest --cov=main_service --cov-report=term --cov-report=html`

자세한 옵션은 `TEST_GUIDE.md`를 참고하세요.

## 설계 문서 맵

- **전체 구조**: docs/DevelopmentPlan/MainService/MainServiceDesign.md
- **로봇 플릿 관리**: docs/DevelopmentPlan/MainService/RobotFleetManagement.md
- **요구사항 추적**: docs/DevelopmentPlan/MainService/MainServiceDesign.md:96-109
- **ROS 인터페이스 명세**: docs/InterfaceSpecification/Main_vs_Pic_Main.md, Main_vs_Pack_Main.md
- **시퀀스**: docs/SequenceDiagram/SC_02_4.md, SC_02_5.md, SC_03_x.md 등

위 문서와 README 내용이 항상 동기화되도록 유지해야 하며, 변경 시 요구사항-설계-구현 간 정합성을 검증하십시오.
