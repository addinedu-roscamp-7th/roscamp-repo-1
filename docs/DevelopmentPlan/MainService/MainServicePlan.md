# Shopee Main Service 개발 계획

본 문서는 Shopee Main Service의 단계별 개발 계획을 정의합니다.
안정적인 개발을 위해, 가장 기본적인 부분부터 시작하여 점진적으로 기능을 확장하는 방식으로 계획을 구성했습니다.

---

## 1단계: 프로젝트 기반 설정 (Foundation)

- **목표**: `Main Service` 개발을 위한 기본 환경을 구축합니다.
- **세부 작업**:
  1. **프로젝트 초기화**: Python 기반의 FastAPI 웹 프레임워크를 사용하여 프로젝트 구조를 생성합니다. (`requirements.txt` 파일 포함)
  2. **환경 설정 관리**: 데이터베이스 접속 정보, 서버 포트 등 주요 설정값을 관리하기 위한 설정 파일을 구성합니다.
  3. **데이터베이스 연동**: `ERDiagram.md`를 기반으로 SQLAlchemy와 같은 ORM(객체 관계 매핑)을 사용하여 데이터베이스 모델을 정의하고, DB 연결을 설정합니다.
  4. **기본 서버 실행**: "Hello, World!"를 응답하는 가장 간단한 API 엔드포인트를 만들어, 서버가 정상적으로 실행되는지 확인합니다.

---

## 2단계: 핵심 API 구현 (Core APIs)

- **목표**: 로봇 연동 전, App과의 핵심적인 상호작용 및 데이터 관리 기능을 구현합니다.
- **세부 작업**:
  1. **사용자 인증 API**: `user_login` 기능을 구현합니다. (App ↔ Main)
  2. **상품 및 재고 API**:
     - `product_search` (상품 검색) 기능을 구현합니다. 이 과정에서 `Main_vs_LLM` 인터페이스를 구현하여 LLM 서버와 연동합니다.
     - `inventory_search`, `inventory_create`, `inventory_update`, `inventory_delete` 등 재고 관리 API를 구현합니다. (App ↔ Main)

---

## 3단계: 주문 및 로봇 연동 구현 (Order & Robot Orchestration)

- **목표**: 프로젝트의 핵심인 주문 생성부터 로봇(Pickee, Packee) 제어까지의 로직을 구현합니다.
- **세부 작업**:
  1. **ROS2 연동 설정**: Python ROS2 클라이언트 라이브러리(`rclpy`)를 프로젝트에 통합하여 로봇과 통신할 준비를 합니다.
  2. **주문 생성 및 Pickee 작업 할당**:
     - `order_create` API를 구현합니다.
     - 주문 정보를 DB에 저장하고, 가용한 Pickee 로봇을 선택하는 로직을 구현합니다.
     - 선택된 Pickee에게 `/pickee/workflow/start_task` ROS2 서비스를 호출하여 작업을 지시합니다.
  3. **Pickee 상태 수신 및 전달**: Pickee로부터 오는 ROS2 토픽 메시지(`moving_status`, `arrival_notice` 등)를 구독(subscribe)하고, 해당 상태를 App에 TCP 이벤트로 전달하는 로직을 구현합니다.
  4. **실시간 상품 선택 로직**: 사용자가 App에서 상품을 선택(`product_selection`)하면, 이를 Pickee에게 전달하여 피킹을 지시하는 전체 흐름을 구현합니다.
  5. **쇼핑 종료 및 Packee 연동**: `shopping_end` 요청을 처리하고, 포장 로봇(`Packee`)에게 작업을 넘기기 위한 `/packee/packing/check_availability` 서비스 호출 로직을 구현합니다.

---

## 4단계: 모니터링 및 고급 기능 구현 (Monitoring & Advanced Features)

- **목표**: 관리자 기능 및 부가 기능을 구현하여 서비스의 완성도를 높입니다.
- **세부 작업**:
  1. **실시간 영상 스트리밍**: 로봇의 카메라 영상을 App으로 전달하는 UDP 스트림 중계 기능을 구현합니다.
  2. **관리자용 API**: `robot_history_search` (작업 이력 조회) 등 관리자용 API를 구현합니다.
  3. **관리자용 알림**: `work_info_notification`, `packing_info_notification` 등 관리자 App에 작업 현황을 알려주는 이벤트 전송 로직을 구현합니다.
