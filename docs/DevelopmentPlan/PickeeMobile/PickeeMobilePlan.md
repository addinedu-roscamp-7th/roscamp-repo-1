# Shopee Pickee Mobile 구현 계획

이 문서는 `PickeeMobileDesign2.md`에 기술된 설계를 바탕으로 Pickee Mobile의 ROS2 노드 및 컴포넌트를 구현하기 위한 단계별 계획을 정의합니다.

## Phase 1: 프로젝트 설정 및 핵심 구조 구현

**목표**: 코드 작성을 위한 기본적인 프로젝트 구조와 상태 기계의 뼈대를 완성합니다.

- **Step 1.1: ROS2 패키지 생성 및 설정**
  - `shopee_pickee_mobile` 폴더 내에 `package.xml`, `setup.py`, `setup.cfg` 등 ROS2 Python 패키지 필수 파일들을 설정합니다.
  - 의존성 패키지(`rclpy`, `shopee_interfaces`, `numpy`, `scipy`, `tf2_ros` 등 `PickeeMobileDesign2.md`에 명시된 라이브러리)를 명시합니다.

- **Step 1.2: 메인 노드 파일 생성**
  - `shopee_pickee_mobile/` 디렉터리에 `mobile_controller.py` 파일을 생성합니다.
  - `rclpy`를 사용하여 `pickee_mobile_controller`라는 이름의 ROS2 노드를 초기화하고, `spin`을 통해 노드가 종료되지 않도록 기본 코드를 작성합니다.

- **Step 1.3: 상태 기계(State Machine) 프레임워크 구현**
  - 상태의 기반이 될 추상 기본 클래스 `State`를 정의합니다. (`on_enter`, `execute`, `on_exit` 메소드 포함)
  - 현재 상태를 관리하고 상태 간의 전환을 처리하는 `StateMachine` 클래스를 구현합니다.

- **Step 1.4: 모든 상태(State) 클래스 정의**
  - `PickeeMobileDesign2.md`의 '3. 상태 관리' 표에 명시된 모든 상태(`IDLE`, `MOVING`, `STOPPED`, `CHARGING`, `ERROR`)에 대한 클래스 파일을 생성합니다.
  - 각 클래스는 `State` 클래스를 상속받으며, 내부는 일단 비워두거나 간단한 로그 출력만 추가합니다.

## Phase 2: 내부 컴포넌트 연동 구현

**목표**: Pickee Mobile 내부의 Localization, Path Planning, Motion Control 컴포넌트들을 구현하고 연동합니다.

- **Step 2.1: Localization Node 구현**
  - 센서 데이터(LiDAR, IMU, Encoder)를 ROS2 토픽으로 수집하고 전처리하는 로직을 구현합니다.
  - AMCL 또는 EKF 기반의 2D 위치 추정 알고리즘을 구현합니다. (기존 ROS2 내비게이션 스택의 `amcl` 노드 활용 또는 자체 구현 고려)
  - 추정된 로봇의 `Pose2D` 정보를 `pickee_mobile_controller`가 구독할 수 있는 내부 토픽으로 발행하는 로직을 구현합니다.

- **Step 2.2: Path Planning Node 구현**
  - `pickee_mobile_controller`로부터 내부 토픽 또는 서비스 요청을 통해 전역 경로(`global_path`)를 수신하는 로직을 구현합니다.
  - Localization Node 또는 Vision Node로부터 실시간 장애물 정보를 수신하는 로직을 구현합니다.
  - DWA(Dynamic Window Approach) 또는 TEB(Timed-Elastic Band)와 같은 지역 경로 계획 알고리즘을 구현합니다. (기존 ROS2 내비게이션 스택의 `local_planner` 노드 활용 또는 자체 구현 고려)
  - 계획된 지역 경로를 Motion Control Node가 구독할 수 있는 내부 토픽으로 발행하는 로직을 구현합니다.

- **Step 2.3: Motion Control Node 구현**
  - Path Planning Node로부터 지역 경로를 수신하는 로직을 구현합니다.
  - PID 제어기 또는 Model Predictive Control(MPC)과 같은 제어 기법을 적용하여 로봇의 선속도 및 각속도를 계산하는 로직을 구현합니다.
  - 계산된 속도 명령을 Motor Driver 및 Steering Actuator에 전달하는 로직을 구현합니다 (예: `/cmd_vel` 토픽 발행).
  - `pickee_mobile_controller`로부터 수신된 속도 제어 명령(`speed_control`) 및 장애물 정보를 기반으로 비상 정지, 감속 등의 안전 제어 로직을 구현합니다.

## Phase 3: 외부 시스템(Pickee Main Controller) 연동 구현

**목표**: Pickee Main Controller와 통신하기 위한 ROS2 인터페이스를 구현합니다.

- **Step 3.1: Publisher 구현**
  - `/pickee/mobile/pose` (`shopee_interfaces/msg/PickeeMobilePose` 타입) 토픽을 100ms 주기로 발행하는 로직을 구현합니다.
  - `/pickee/mobile/arrival` (`shopee_interfaces/msg/PickeeMobileArrival` 타입) 토픽을 목적지 도착 시 1회 발행하는 로직을 구현합니다.

- **Step 3.2: Service Server 구현**
  - `/pickee/mobile/move_to_location` (`shopee_interfaces/srv/PickeeMobileMoveToLocation` 타입) 서비스 서버를 구현합니다. 요청 수신 시 상태 전이 및 내부 컴포넌트 제어 로직을 호출합니다.
  - `/pickee/mobile/update_global_path` (`shopee_interfaces/srv/PickeeMobileUpdateGlobalPath` 타입) 서비스 서버를 구현합니다. 요청 수신 시 내부 Path Planning Node에 경로 업데이트를 지시합니다.

- **Step 3.3: Subscriber 구현**
  - `/pickee/mobile/speed_control` (`shopee_interfaces/msg/PickeeMobileSpeedControl` 타입) 토픽을 구독하고, 수신된 속도 제어 명령을 내부 Motion Control Node에 전달하는 로직을 구현합니다.

## Phase 4: 상태별 로직 및 파라미터 처리 구현

**목표**: 앞서 구현한 인터페이스들을 활용하여 각 상태의 세부 동작과 파라미터 처리 로직을 완성합니다.

- **Step 4.1: 상태별 세부 로직 구현**
  - Phase 1에서 생성한 각 상태 클래스 내부에 실제 동작 로직을 채워 넣습니다.
  - `IDLE` 상태: `move_to_location` 서비스 요청을 대기하고, 요청 수신 시 `MOVING` 상태로 전이.
  - `MOVING` 상태: 경로 계획 및 추종, 위치 및 상태를 주기적으로 보고. 목적지 도착 시 `IDLE` 또는 `STOPPED` 상태로 전이.
  - `STOPPED` 상태: 정지 상태를 유지하고, `speed_control` 토픽을 통해 재시작 명령을 대기.
  - `CHARGING` 상태: 배터리 충전 로직을 수행하고, 배터리 임계값 이상 충전 완료 시 `IDLE` 상태로 전이.
  - `ERROR` 상태: 오류 발생 시 진입하며, 오류 처리 및 복구 로직을 수행하거나 사용자 개입을 대기.

- **Step 4.2: 파라미터 처리 로직 구현**
  - `PickeeMobileDesign2.md`의 '6. 파라미터'에 명시된 설정값들을 노드 실행 시 불러와 사용할 수 있도록 ROS2 파라미터 처리 로직을 추가합니다.

## Phase 5: 테스트 및 안정화

**목표**: 구현된 기능의 신뢰성과 안정성을 확보합니다.

- **Step 5.1: 단위 테스트 (Unit Test)**
  - 상태 기계의 상태 전환 로직, 각 상태 클래스의 개별 로직, Localization/Path Planning/Motion Control 알고리즘 등 핵심 기능에 대한 단위 테스트 코드를 작성합니다.

- **Step 5.2: 통합 테스트 (Integration Test)**
  - `ros2 launch`를 사용하여 `mobile_controller`와 각 내부 컴포넌트 노드들을 함께 실행하여 인터페이스가 정상적으로 동작하는지 테스트합니다.
  - Mock 노드를 사용하여 Pickee Main Controller 또는 센서/액추에이터를 시뮬레이션하여 Pickee Mobile의 외부 인터페이스를 테스트합니다.

- **Step 5.3: 시스템 테스트 (System Test)**
  - 실제 로봇 또는 시뮬레이션 환경(Gazebo 등)에서 Pickee Mobile을 실행하여 `PickeeMobileDesign2.md`에 정의된 전체 시나리오가 의도대로 동작하는지 검증합니다.
  - 예외 상황(e.g., 서비스 응답 실패, 경로 이동 실패, 센서 오류)에 대한 처리 로직을 테스트하고 보강합니다.

## 6. 코딩 표준 (Coding Standards)

`PickeeMobileDesign2.md`에 명시된 ROS2 및 Python 코딩 표준을 준수하여 개발을 진행한다.

*   **ROS2 표준:**
    *   Package Names: `snake_case`
    *   Node/Topic/Service/Action/Parameter Names: `snake_case`
    *   Type Names: `PascalCase`
    *   Type Field Names: `snake_case`
    *   Type Constants Names: `SCREAMING_SNAKE_CASE`
*   **Python 표준:**
    *   Package 및 module 이름: `snake_case`
    *   Class 및 exception 이름: `PascalCase`
    *   Function, method, parameter, local/instance/global 변수 이름: `snake_case`
    *   Global/Class constants: `SCREAMING_SNAKE_CASE`
*   **공통 규칙:**
    *   주석은 한국어로 작성 (`#`)
    *   함수와 함수 사이 1줄 간격
    *   Import문은 한줄에 하나씩
    *   제어문은 반드시 중괄호 사용 (Python에서는 해당 없음, C++에 적용)
    *   문자열은 작은따옴표 사용