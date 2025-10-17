# Shopee Packee Arm Controller 구현 계획

이 문서는 `PackeeArmDesign.md` 상세 설계를 바탕으로 Packee Arm Controller ROS2 노드를 구현하기 위한 단계별 계획을 정의한다. 상위 컴포넌트인 Packee Main Controller 계획(`docs/DevelopmentPlan/PackeeMain/PackeeMainPlan.md`)과 동일한 Phase 체계를 유지하며, 상·하위 컴포넌트 간 의존성을 명시한다.

## Phase 1: 프로젝트 준비 및 기본 구조 확립

**목표**: 패키지 환경과 노드 뼈대를 갖추고 Packee Main과 공유되는 파라미터/코딩 표준을 정비한다.

- **Step 1.1: ROS2 패키지 정비**
  - `packee_arm` 패키지의 `package.xml`, `CMakeLists.txt` 의존성 확인 (`rclcpp`, `shopee_interfaces`).
  - 코딩 표준(`docs/CodingStandard/standard.md`) 적용 상태 점검, 코드 포맷 통일.
- **Step 1.2: 노드 및 파라미터 뼈대 구현**
  - `PackeeArmController` 클래스 구성, `main()` 진입점/노드 초기화.
  - 설계 9장 파라미터 선언(`servo_gain_xy`, `servo_gain_z`, `servo_gain_yaw`, `cnn_confidence_threshold` 등)과 기본값 로드.
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

## Phase 3: Visual Servo 모듈 및 하드웨어 연동

**목표**: Two-stream CNN 기반 VisualServoModule을 구현하고, ExecutionManager와 하드웨어 계층에 통합한다.

- **Step 3.1: VisualServoModule 뼈대 구축**
  - 목표/현재 이미지 버퍼 관리, CNN 추론 트리거, P 제어기 틀을 구현한다.
  - 4자유도 오차 계산(`r* - r_c`)과 게인/클리핑 적용 로직을 작성한다.
- **Step 3.2: Two-stream CNN 통합**
  - 학습된 ONNX/TensorRT 모델 로딩, 추론 파이프라인, 신뢰도 산출을 구현한다.
  - 224×224 리사이즈, 정규화 등 전처리와 추론 후 포즈/신뢰도 후처리를 연결한다.
- **Step 3.3: ExecutionManager/Hardware 연계**
  - 명령 큐와 VisualServoModule을 연결해 서비스별 상태(`servoing`, `grasping`, `placing`)를 관리한다.
  - `ArmDriverProxy`에 속도 명령 인터페이스를 구현하고, 그리퍼 제어와 타임아웃 응답을 연동한다.

## Phase 4: 진단·예외 처리 및 파라미터 튜닝

**목표**: CNN 신뢰도 기반 예외 처리와 Diagnostics를 완성하고, P 제어 파라미터를 런타임에서 조정 가능하도록 한다.

- **Step 4.1: 예외 처리 로직 보강**
  - CNN 신뢰도 저하, 시야 이탈, 그리퍼 오류에 대한 상태 메시지/재시도 정책을 구현한다.
  - 에러 코드 체계와 상태 토픽 payload 확장을 Packee Main과 합의한다.
- **Step 4.2: Diagnostics/알림 연계**
  - `/diagnostics` 또는 별도 진단 토픽에 CNN 신뢰도, 수렴 스텝 수, 최종 오차를 발행한다.
  - Shopee App 알림 요건(UR_04/SR_11)을 충족하도록 이벤트 매핑을 정의한다.
- **Step 4.3: 파라미터 동적 조정**
  - `servo_gain_*`, `cnn_confidence_threshold`, `max_translation_speed` 등을 ROS2 파라미터 서버로 노출한다.
  - 변경 시 Validation 및 실행 중 반영 로직(soft restart, 게인 스무딩)을 설계한다.

## Phase 5: 테스트, 문서화 및 인수

**목표**: VisualServoModule 성능을 포함한 전 구간 테스트를 수행하고 산출물을 Packee Main 일정과 맞춰 인수한다.

- **Step 5.1: 단위 테스트**
  - gtest + rclcpp로 서비스 입력 검증, 상태 전이, VisualServoModule 오차 계산과 게인 적용 로직을 테스트한다.
  - CNN 추론 모듈은 ONNXRuntime/TensorRT mock을 이용해 입력/출력 형식을 검증한다.
- **Step 5.2: 통합/시뮬레이션 테스트**
  - Packee Main Mock과 연동해 `SC_03_3`, `SC_03_4` 시퀀스, CNN 신뢰도 저하 시나리오를 재현한다.
  - Gazebo/Isaac Sim에서 eye-in-hand 카메라, 금속 블록 모델, 조명 변화를 적용해 수렴 스텝과 최종 오차를 측정한다.
- **Step 5.3: HIL 테스트 및 문서 인수**
  - YASKAWA MH5 하드웨어로 최소 20개 시나리오(초기 오프셋, 조명 변화)를 실행해 15스텝 내 수렴을 검증한다.
  - 테스트 리포트, 데이터 수집/라벨링 절차, 장애 대응 가이드를 업데이트한다.
  - 완료 기준(DoD): 서비스 성공/실패 재현 가능, CNN 신뢰도/수렴 스텝 모니터링, 예외 로그 확인, 문서 최신화.

## 공통 관리 항목
- **의존성**: Packee Main/Packee Vision 인터페이스 확정, 학습 데이터(목표/현재 이미지 페어) 수집 협조, 하드웨어 API 제공, 테스트 환경(Mock/시뮬레이터) 준비.
- **리스크 및 대응**  
  - 학습 데이터 부족/조명 불균일 → 데이터 수집 확대, glare 증강, 광원 제어 장치 도입  
  - CNN 추론 지연 → TensorRT 최적화, FP16/INT8 변환, GPU 리소스 모니터링  
  - 듀얼 암 충돌 → 시뮬레이션 검증, 충돌 영역 제한 로직  
  - ROS 통신 지연 → QoS/타임아웃 조정, 재시도 로직  
- **커뮤니케이션**: 주간 스탠드업(Arm SW 팀), 스프린트 리뷰(Arm/Main/Vision/HW 팀), QA 리뷰. 모든 문서/리포트는 `docs/DevelopmentPlan/PackeeArm/` 경로에 저장한다.
