# Shopee Packee Arm Controller 구현 계획

이 문서는 `PackeeArmDesign.md` 상세 설계를 바탕으로 Packee Arm Controller ROS2 노드를 구현하기 위한 단계별 계획을 정의한다. 상위 컴포넌트인 Packee Main Controller 계획(`docs/DevelopmentPlan/PackeeMain/PackeeMainPlan.md`)과 동일한 Phase 체계를 유지하며, 상·하위 컴포넌트 간 의존성을 명시한다.

## Phase 1: 프로젝트 준비 및 기본 구조 확립

**목표**: 패키지 환경과 노드 뼈대를 갖추고 Packee Main과 공유되는 파라미터/코딩 표준을 정비한다.

- **Step 1.1: ROS2 패키지 정비**
  - `packee_arm` 패키지의 `package.xml`, `CMakeLists.txt` 의존성 확인 (`rclcpp`, `shopee_interfaces`).
  - 코딩 표준(`docs/CodingStandard/standard.md`) 적용 상태 점검, 코드 포맷 통일.
- **Step 1.2: 노드 및 파라미터 뼈대 구현**
  - `PackeeArmController` 클래스 구성, `main()` 진입점/노드 초기화.
  - 설계 8장 파라미터 선언(`robot_id`, `arm_sides`, 속도 스케일 등)과 기본값 로드.
- **Step 1.3: 실행 상태·로깅 기본 구조 마련**
  - 내부 상태(`IDLE`, `PICKING` 등) 열거형 정의, Diagnostics/로깅 초기 훅 마련.
  - Packee Main Controller 로그 포맷과 일관성 유지.

## Phase 2: ROS 인터페이스 구현

**목표**: Packee Main과 합의된 서비스/토픽 기반 인터페이스를 구현하고 기본적인 명령 흐름을 검증한다.

- **Step 2.1: 서비스 서버 구현**
  - `/packee/arm/move_to_pose`, `/packee/arm/pick_product`, `/packee/arm/place_product` 서비스 콜백 작성.
  - 파라미터 유효성 검사, 표준 응답 메시지 템플릿 구성.
- **Step 2.2: 상태 토픽 퍼블리셔 구현**
  - `/packee/arm/pose_status`, `/packee/arm/pick_status`, `/packee/arm/place_status` 발행 로직 구현.
  - 진행률/단계 상수 정의, Packee Main 상태 전이 조건과 조율.
- **Step 2.3: Packee Main Mock 연동 테스트**
  - C++ Mock 노드(`ros2 run packee_arm mock_packee_main`)를 활용하여 정상/거부 시나리오를 검증.
  - QoS와 타임아웃 설정을 Packee Main Controller와 합의한 값으로 맞춤.

## Phase 3: 실행 매니저 및 하드웨어 추상화

**목표**: 듀얼 암 동작을 위한 명령 큐, 동시성 제어, 하드웨어 인터페이스를 구현한다.

- **Step 3.1: Execution Manager 구현**
  - 좌/우 팔 명령 큐, Future 관리, 진행률 업데이트 로직 작성.
  - 충돌 방지를 위한 동시성 제어(세마포어/락) 적용.
- **Step 3.2: Hardware Interface Stub 구축**
  - `MotionPlannerAdapter`, `ArmDriverProxy`, `GripperController` 추상 클래스 정의.
  - 하드웨어 팀과 API 규격을 확정하고, 시뮬레이터/Mock과 연동 가능한 스텁 구현.
- **Step 3.3: 단계별 상태 매핑**
  - 설계 6장 단계(planning, grasping 등)를 Execution Manager에 반영.
  - 실패 시도 시 Packee Main이 재시도 여부를 판단할 수 있도록 상태 토픽에 오류 코드 포함.

## Phase 4: 진단·예외 처리 및 확장 기능

**목표**: 안정성 확보를 위해 예외 처리, 진단 로깅, 파라미터 동적 조정을 완성한다.

- **Step 4.1: 예외 처리 로직 보강**
  - 파라미터 오류, 하드웨어 타임아웃, 센서 이상에 대한 응답/토픽 일관성 유지.
  - 에러 코드 체계 정의, Packee Main에 전달될 메시지 포맷 합의.
- **Step 4.2: Diagnostics/알림 연계**
  - `/diagnostics` 또는 별도 진단 토픽 발행 구현.
  - Shopee App 알림 요건(UR_04/SR_11) 충족을 위한 데이터 포인트 정의.
- **Step 4.3: 파라미터 동적 조정**
  - 속도/힘 한계 파라미터를 ROS2 런타임에서 수정 가능하도록 구현.
  - 변경 시 Validation 및 실행 중 반영 로직 조정.

## Phase 5: 테스트, 문서화 및 인수

**목표**: 다양한 테스트 레벨을 수행하고 산출물을 Packee Main Controller 일정과 맞춰 인수한다.

- **Step 5.1: 단위 테스트**
  - gtest + rclcpp 기반 테스트로 서비스 입력 검증, 상태 전이, 진행률 계산 확인.
- **Step 5.2: 통합/시뮬레이션 테스트**
  - Packee Main Mock, Vision 좌표 데이터와 연계해 `SC_03_3`, `SC_03_4` 시퀀스를 재현.
  - Gazebo/Isaac Sim에서 듀얼 암 충돌 회피 및 동기화 검증.
- **Step 5.3: HIL 테스트 및 문서 인수**
  - 실제 하드웨어와 연결해 최소 3회 연속 성공 시나리오 수행.
  - 테스트 리포트, 운영 매뉴얼, 장애 대응 가이드 업데이트.
  - 완료 기준(DoD): 서비스 성공/실패 재현 가능, 상태 토픽 모니터링, 예외 로그 확인, 문서 최신화.

## 공통 관리 항목
- **의존성**: Packee Main/Packee Vision 인터페이스 확정, 하드웨어 API 제공, 테스트 환경(Mock/시뮬레이터) 준비.
- **리스크 및 대응**  
  - 하드웨어 API 변경 → 어댑터 추상화, 변경시 영향 최소화  
  - Vision 좌표 오차 → 보정 파라미터 도입, 로그 기반 보정  
  - 듀얼 암 충돌 → 시뮬레이션 검증, 충돌 영역 제한 로직  
  - ROS 통신 지연 → QoS/타임아웃 조정, 재시도 로직  
- **커뮤니케이션**: 주간 스탠드업(Arm SW 팀), 스프린트 리뷰(Arm/Main/Vision/HW 팀), QA 리뷰. 모든 문서/리포트는 `docs/DevelopmentPlan/PackeeArm/` 경로에 저장한다.
