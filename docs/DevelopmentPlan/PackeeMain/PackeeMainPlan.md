# Shopee Packee Main Controller 구현 계획

이 문서는 `DetailedDesign.md`에 기술된 설계를 바탕으로 `packee_main_controller` ROS2 노드를 구현하기 위한 단계별 계획을 정의합니다.

## Phase 1: 프로젝트 설정 및 핵심 구조 구현

**목표**: 코드 작성을 위한 기본적인 프로젝트 구조와 상태 기계의 뼈대를 완성합니다.

- **Step 1.1: ROS2 패키지 생성 및 설정**
  - `shopee_packee_main` 폴더 내에 `package.xml`, `setup.py`, `setup.cfg` 등 ROS2 Python 패키지 필수 파일들을 설정합니다.
  - 의존성 패키지(`rclpy`, `shopee_interfaces`)를 명시합니다.

- **Step 1.2: 메인 노드 파일 생성**
  - `shopee_packee_main/` 디렉터리에 `main_controller.py` 파일을 생성합니다.
  - `rclpy`를 사용하여 `packee_main_controller`라는 이름의 ROS2 노드를 초기화하고, `spin`을 통해 노드가 종료되지 않도록 기본 코드를 작성합니다.

- **Step 1.3: 상태 기계(State Machine) 프레임워크 구현**
  - 상태의 기반이 될 추상 기본 클래스 `State`를 정의합니다. (`on_enter`, `execute`, `on_exit` 메소드 포함)
  - 현재 상태를 관리하고 상태 간의 전환을 처리하는 `StateMachine` 클래스를 구현합니다.

- **Step 1.4: 모든 상태(State) 클래스 정의**
  - `DetailedDesign.md`의 '3. 상태 관리' 표에 명시된 모든 상태(`INITIALIZING`, `CHARGING_AVAILABLE` 등)에 대한 클래스 파일을 생성합니다.
  - 각 클래스는 `State` 클래스를 상속받으며, 내부는 일단 비워두거나 간단한 로그 출력만 추가합니다.

## Phase 2: 내부 컴포넌트 연동 구현

**목표**: Mobile, Arm, Vision 등 Packee 로봇의 하위 컴포넌트들과 통신하기 위한 ROS2 인터페이스를 구현합니다.

- **Step 2.1: Subscriber 구현**
  - Mobile, Arm, Vision에서 발행하는 토픽을 수신하기 위한 Subscriber를 `main_controller` 노드에 추가합니다.
  - 예: `/packee/mobile/arrival`, `/packee/arm/place_status`, `/packee/vision/detection_result` 등
  - 각 Subscriber의 콜백 함수에서는 수신된 데이터를 처리하여 상태 기계의 상태 전환 트리거로 사용될 수 있도록 구현합니다.

- **Step 2.2: Service Client 구현**
  - Mobile, Arm, Vision에 명령을 내리기 위한 Service Client를 `main_controller` 노드에 추가합니다.
  - 예: `/packee/mobile/move_to_location`, `/packee/arm/pick_product`, `/packee/vision/detect_products` 등
  - 서비스 호출을 쉽게 할 수 있도록 비동기(async/await) 래퍼(wrapper) 함수를 구현하는 것을 권장합니다.

## Phase 3: 외부 시스템(Main Service) 연동 구현

**목표**: Shopee Main Service와 통신하기 위한 ROS2 인터페이스를 구현합니다.

- **Step 3.1: Publisher 구현**
  - 로봇의 현재 상태를 Main Service에 보고하기 위한 Publisher를 `main_controller` 노드에 추가합니다.
  - 예: `/packee/packing_complete`, `/packee/robot_status`, `/packee/availability_result` 등
  - `/packee/robot_status`는 주기적으로 발행하도록 타이머를 사용해 구현합니다.

- **Step 3.2: Service Server 구현**
  - Main Service로부터 명령을 수신하기 위한 Service Server를 `main_controller` 노드에 추가합니다.
  - 예: `/packee/packing/check_availability`, `/packee/packing/start` 등
  - 각 서비스의 콜백 함수에서는 수신된 요청을 분석하여 상태 기계에 시작 이벤트를 전달하도록 구현합니다.

## Phase 4: 시나리오 워크플로우 및 상태별 로직 구현

**목표**: 앞서 구현한 인터페이스들을 활용하여 각 상태의 세부 동작과 전체 시나리오 워크플로우를 완성합니다.

- **Step 4.1: 상태별 세부 로직 구현**
  - Phase 1에서 생성한 각 상태 클래스 내부에 실제 동작 로직을 채워 넣습니다.
  - 예: `CHECKING_CART` 상태의 `on_enter` 메소드에서는 `/packee/vision/check_cart_presence` 서비스를 호출하고, `DETECTING_PRODUCT` 상태의 `on_enter`에서는 `/packee/vision/detect_products_in_cart` 서비스를 호출하는 식입니다.

- **Step 4.2: 포장 시나리오 구현**
  - `DetailedDesign.md`의 '5.1. 포장 시나리오 워크플로우'에 명시된 흐름에 따라 상태들이 올바르게 전환되고 각 상태의 로직이 실행되는지 확인하며 전체 시나리오를 완성합니다.
  - 상태 전환에 필요한 조건(e.g., 서비스 요청, 토픽 수신)을 명확하게 구현합니다.

- **Step 4.3: 파라미터 처리 로직 구현**
  - `DetailedDesign.md`의 '6. 파라미터'에 명시된 설정값들을 노드 실행 시 불러와 사용할 수 있도록 ROS2 파라미터 처리 로직을 추가합니다.

## Phase 5: 테스트 및 안정화

**목표**: 구현된 기능의 신뢰성과 안정성을 확보합니다.

- **Step 5.1: 단위 테스트 (Unit Test)**
  - 상태 기계의 상태 전환 로직, 각 상태 클래스의 개별 로직 등 핵심 기능에 대한 단위 테스트 코드를 작성합니다.

- **Step 5.2: 통합 테스트 (Integration Test)**
  - `ros2 launch`를 사용하여 `main_controller`와 각 하위 컴포넌트의 모의 노드(mock node)를 함께 실행하여 인터페이스가 정상적으로 동작하는지 테스트합니다.
  - 모의 노드는 실제 컴포넌트처럼 특정 서비스 요청에 응답하거나 토픽을 발행하는 역할을 합니다.

- **Step 5.3: 시스템 테스트 (System Test)**
  - 실제 로봇 또는 시뮬레이션 환경에서 `packee_main_controller`를 실행하여 전체 시나리오가 의도대로 동작하는지 검증합니다.
  - 예외 상황(e.g., 서비스 응답 실패, 경로 이동 실패)에 대한 처리 로직을 테스트하고 보강합니다.
