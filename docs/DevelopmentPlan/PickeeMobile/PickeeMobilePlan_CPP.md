# Shopee Pickee Mobile 구현 계획 (Nav2 기반 C++ 버전)

이 문서는 `PickeeMobileDesign_CPP.md`에 기술된 **Nav2 기반 설계**를 바탕으로 Pickee Mobile의 Shopee 브리지 노드 및 Nav2 통합 컴포넌트를 구현하기 위한 단계별 계획을 정의합니다.

## Phase 1: Nav2 환경 설정 및 기본 구조 구현

**목표**: Nav2 스택과 연동하기 위한 기본 프로젝트 구조와 브리지 노드의 뼈대를 완성합니다.

- **Step 1.1: Nav2 기반 ROS2 패키지 설정**
  - `pickee_mobile_wonho` 패키지의 `package.xml`에 Nav2 의존성을 추가합니다.
  - `nav2_bringup`, `nav2_msgs`, `nav2_map_server`, `nav2_amcl`, `nav2_planner`, `nav2_controller`, `nav2_bt_navigator` 등
  - `CMakeLists.txt`에 Nav2 관련 패키지와 `rclcpp_action`, `rclcpp_components` 의존성을 설정합니다.
  - `shopee_interfaces`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `tf2_ros` 등 필수 의존성 포함
  - 기존 launch 파일들(`nav2_bringup_launch.xml`, `navigation_launch.xml`)과 설정 파일들(`nav2_params.yaml`) 검토 및 정리

- **Step 1.2: Shopee 브리지 노드 기본 구조 생성**
  - `include/pickee_mobile_wonho/pickee_mobile_bridge.hpp` 헤더 파일을 생성합니다.
  - `src/pickee_mobile_bridge.cpp` 파일을 생성하여 `rclcpp::Node`를 상속받는 `PickeeMobileBridge` 클래스를 구현합니다.
  - Nav2와 Shopee 시스템 간의 통신을 담당하는 중앙 허브 역할 설계
  - `src/main.cpp` 파일을 생성하여 브리지 노드를 초기화하고 실행하는 메인 함수를 작성합니다.
  - ROS2 파라미터 선언 및 로딩 기능 구현 (`robot_id`, `pose_publish_frequency`, `navigation_timeout` 등)

- **Step 1.3: 핵심 컴포넌트 클래스 골격 생성**
  - `NavigationBridge`: Nav2 Action 클라이언트 관리 (`include/pickee_mobile_wonho/navigation_bridge.hpp`)
  - `StateManager`: 네비게이션 상태 및 시스템 상태 관리 (`include/pickee_mobile_wonho/state_manager.hpp`)
  - `CostmapUpdater`: 동적 장애물 처리 및 costmap 업데이트 (`include/pickee_mobile_wonho/costmap_updater.hpp`)
  - `PoseReporter`: AMCL 위치를 Shopee 포맷으로 변환 및 발행 (`include/pickee_mobile_wonho/pose_reporter.hpp`)
  - 각 컴포넌트의 기본 생성자, 소멸자, 초기화 함수만 구현

- **Step 1.4: Nav2 launch 파일 통합 및 테스트**
  - 기존 `nav2_bringup_launch.xml` 검토 및 Shopee 시스템에 맞게 수정
  - `mobile_bringup.launch.xml` 생성하여 Nav2 스택과 브리지 노드를 함께 시작
  - Gazebo 시뮬레이션 환경에서 Nav2 기본 동작 확인 (AMCL, Map Server, Planner)
  - 기본 빌드 및 실행 테스트

## Phase 2: Nav2 Action 클라이언트 및 상태 관리 구현

**목표**: Nav2의 NavigateToPose Action을 호출하고 네비게이션 상태를 관리하는 핵심 기능을 구현합니다.

- **Step 2.1: NavigationBridge 구현**
  - `src/navigation_bridge.cpp` 파일을 생성하여 Nav2 Action 클라이언트를 구현합니다.
  - `rclcpp_action::Client<nav2_msgs::action::NavigateToPose>` 를 사용한 Nav2 Action 호출
  - `SendNavigationGoal()`: Shopee 좌표를 Nav2 goal로 변환하여 전송
  - `CancelNavigation()`: 현재 네비게이션 취소
  - `IsNavigationActive()`: 네비게이션 활성 상태 확인
  - `GetNavigationStatus()`: 현재 네비게이션 상태 반환 (PENDING, ACTIVE, SUCCEEDED, FAILED 등)
  - Action 피드백 콜백을 통한 실시간 상태 업데이트

- **Step 2.2: StateManager 구현**
  - `src/state_manager.cpp` 파일을 생성하여 로봇 상태를 관리합니다.
  - 상태 정의: `IDLE`, `NAVIGATING`, `ARRIVED`, `STOPPED`, `ERROR`, `CHARGING`
  - 상태 전환 로직 구현 (`TransitionTo()`, `GetCurrentState()`, `CanTransitionTo()`)
  - Nav2 Action 상태와 Shopee 상태 간의 매핑
  - 상태별 타임아웃 및 에러 처리 로직
  - `std::atomic` 및 `std::mutex`를 활용한 thread-safe 상태 관리

- **Step 2.3: PoseReporter 구현**  
  - Nav2 AMCL에서 발행하는 `/amcl_pose` 토픽을 구독
  - `geometry_msgs::msg::PoseWithCovarianceStamped` 메시지를 `shopee_interfaces::msg::PickeeMobilePose`로 변환
  - 배터리 정보, 속도 정보 등 추가 데이터와 함께 통합 위치 정보 발행
  - 10Hz 주기로 `/pickee/mobile/pose` 토픽 발행
  - 위치 정확도 및 신뢰도 평가 기능 추가
  - `std::unique_ptr<AStarPlanner>` 및 `std::unique_ptr<DWAPlanner>` 활용
  - Vision에서 제공되는 장애물 정보 기반 동적 경로 수정 로직
  - `nav_msgs::msg::Path`를 활용한 경로 데이터 관리
  - 실시간 성능을 위한 알고리즘 최적화 및 캐싱 전략

- **Step 2.3: MotionControlComponent 구현**
  - `include/pickee_mobile/components/motion_control_component.hpp` 및 `src/components/motion_control_component.cpp` 파일을 생성합니다.
  - 고빈도 제어 루프(50-100Hz)를 위한 최적화된 C++ PID 제어기 구현
  - `std::unique_ptr<PIDController>` 기반 선속도/각속도 제어
  - `std::atomic<bool> emergency_stop_`을 활용한 thread-safe 안전 제어
  - Real-time 제어를 위한 interrupt-driven 안전 로직
  - `geometry_msgs::msg::Twist`를 통한 `/cmd_vel` 발행
  - Exception safety 및 collision avoidance 로직

- **Step 2.4: SensorManager 구현**
  - 통합 센서 데이터 관리를 위한 `SensorManager` 클래스 구현
  - Lock-free 링 버퍼를 활용한 고성능 센서 데이터 처리
  - 센서별 데이터 전처리 및 필터링 로직
  - `std::optional`을 활용한 안전한 센서 데이터 처리

- **Step 2.5: BatteryManager 및 ArrivalDetector 구현**
  - `std::atomic<double>`을 활용한 thread-safe 배터리 모니터링
  - 배터리 예측 알고리즘 및 이력 추적(`std::deque` 활용)
  - 고정밀 도착 감지를 위한 Eigen 라이브러리 거리/각도 계산
  - 연산 최적화를 위한 거리 제곱 비교 및 조기 종료 로직

## Phase 3: 외부 통신 인터페이스 구현

**목표**: Pickee Main Controller와의 ROS2 통신 인터페이스를 고성능 C++로 구현합니다.

- **Step 3.1: CommunicationInterface 클래스 구현**
  - `include/pickee_mobile/communication_interface.hpp` 파일을 생성하여 통신 인터페이스를 캡슐화
  - Modern C++ 스타일의 콜백 함수 및 스마트 포인터 활용
  - Exception handling 및 통신 오류 복구 메커니즘 구현

- **Step 3.2: Publisher 구현**
  - `/pickee/mobile/pose` 토픽 발행
    - AMCL pose를 Shopee 포맷으로 변환하여 10Hz 주기로 발행
    - `PoseReporter` 클래스를 통한 좌표계 변환 및 메시지 구성
    - 배터리 잔량, 로봇 상태 등 추가 정보 포함
  - `/pickee/mobile/arrival` 토픽 발행
    - Nav2 goal 도달 시 자동 발행
    - 도착 시간, 최종 위치 오차 등 상세 정보 포함

- **Step 3.3: Service Server 구현**
  - `/pickee/mobile/move_to_location` 서비스 서버 구현
    - Shopee 좌표계를 Nav2 map frame으로 변환
    - `NavigationBridge`를 통해 Nav2 Action 호출
    - 비동기 응답 처리 및 에러 핸들링
  - `/pickee/mobile/update_global_path` 서비스 서버 구현 (선택사항)
    - Nav2 경로 재계획 요청
    - 동적 장애물 정보 반영

- **Step 3.4: Subscriber 구현**
  - `/pickee/mobile/speed_control` 토픽 구독
    - Shopee 속도 제어 명령 수신
    - Nav2 `/cmd_vel` 토픽에 속도 제한 적용
    - 비상 정지 명령 처리
  - Nav2 상태 토픽들 구독
    - `/amcl_pose`: 로봇 위치 정보
    - `/plan`: 전역 경로 정보
    - `/local_costmap/costmap`: 지역 costmap 정보

- **Step 3.5: 좌표계 변환 및 데이터 포맷 변환**
  - Shopee 좌표계 ↔ Nav2 map frame 변환
  - `tf2_ros::Buffer`를 활용한 동적 좌표 변환
  - 메시지 타입 간 변환 유틸리티 함수들
  - 시간 동기화 및 latency 보상

## Phase 4: Nav2 파라미터 설정 및 최적화

**목표**: Nav2 스택의 파라미터를 Shopee 환경에 맞게 최적화하고, 성능 튜닝을 수행합니다.

- **Step 4.1: Nav2 파라미터 파일 구성**
  - `config/nav2_params.yaml` 파일 생성 및 Shopee 환경 최적화
    - AMCL 파라미터: 로봇 크기, 센서 노이즈, localization 정확도
    - Planner 파라미터: A* 알고리즘 최적화, 경로 스무딩
    - Controller 파라미터: DWB controller 속도 제한, 장애물 회피
    - Costmap 파라미터: inflation radius, obstacle layer 설정
  - Recovery behavior 파라미터: backup, rotate, clear costmap 동작
  - BT Navigator XML 파일 구성: behavior tree 로직 커스터마이징

- **Step 4.2: 상태별 Nav2 연동 로직 구현**
  - **IdleState 구현**: 
    - Nav2 스택 유지하면서 목표 대기 상태
    - AMCL localization 지속적 업데이트
    - `/pickee/mobile/move_to_location` 서비스 요청 시 `MovingState` 전환
  - **MovingState 구현**:
    - Nav2 Action 실행 및 상태 모니터링
    - 실시간 AMCL pose 기반 위치 보고
    - 목적지 도달 또는 Nav2 실패 시 적절한 상태 전환
    - Vision 시스템 장애물 정보를 costmap에 반영
  - **StoppedState 구현**:
    - Nav2 Action cancel 및 cmd_vel 정지
    - Emergency stop 해제까지 대기
    - 재시작 시 이전 목표로 자동 복귀 기능
  - **ChargingState 구현**:
    - 충전소 위치로 자동 네비게이션
    - 도킹 완료 후 Nav2 스택 일시 중지
    - 충전 완료 시 원래 작업 위치로 복귀
  - **ErrorState 구현**:
    - Nav2 관련 오류 진단 및 복구
    - AMCL kidnapped robot 복구
    - 네비게이션 실패 시 recovery behavior 실행

- **Step 4.3: Nav2 성능 최적화**
  - **플래너 최적화**:
    - 경로 계획 주기 및 범위 최적화
    - 동적 장애물 대응 속도 개선
    - 메모리 사용량 및 CPU 부하 모니터링
  - **컨트롤러 최적화**:
    - DWB local planner 파라미터 튜닝
    - 속도 프로파일 및 가속도 제한 최적화
    - 장애물 회피 성능 향상
  - **Costmap 최적화**:
    - Layer 업데이트 주기 최적화
    - 메모리 효율적인 costmap 관리
    - 불필요한 계산 최소화

- **Step 4.4: 파라미터 관리 시스템 구현**
  - **Shopee 브리지 파라미터**:
    - `config/shopee_bridge_params.yaml` 파일 생성
    - 좌표계 변환, 통신 주기, 오류 처리 파라미터
    - Launch 파일에서 Nav2와 Shopee 파라미터 통합 관리
  - **동적 파라미터 업데이트**:
    - Nav2 파라미터 런타임 변경 지원
    - Shopee 시스템 요구사항에 따른 동적 조정
    - 파라미터 변경 시 안전한 적용 메커니즘

## Phase 5: 테스트 및 검증

**목표**: 구현된 기능의 신뢰성, 성능, 안전성을 체계적으로 검증합니다.

- **Step 5.1: 단위 테스트 (Unit Test)**
  - **Google Test 프레임워크 설정**:
    - `test/` 디렉터리에 `ament_cmake_gtest` 활용
    - `CMakeLists.txt`에 테스트 타겟 추가
  - **핵심 컴포넌트 테스트**:
    - `test_navigation_bridge.cpp`: Nav2 Action 클라이언트 및 상태 관리 테스트
    - `test_state_manager.cpp`: 상태 전환 로직 및 thread-safe 동작 테스트
    - `test_pose_reporter.cpp`: 좌표계 변환 및 메시지 포맷 정확도 테스트
    - `test_costmap_updater.cpp`: 동적 장애물 처리 및 costmap 업데이트 테스트
  - **Memory Leak 및 Performance 테스트**:
    - Valgrind, AddressSanitizer 호환성 검증
    - 실행 시간 및 메모리 사용량 벤치마크

- **Step 5.2: Nav2 통합 테스트**
  - **Nav2 스택 테스트**:
    - `test/integration/` 디렉터리에 Nav2 통합 테스트 구현
    - AMCL localization 정확도 검증
    - Planner와 Controller 연동 테스트
    - Recovery behavior 동작 검증
  - **Shopee-Nav2 브리지 테스트**:
    - 좌표계 변환 정확도 검증
    - 메시지 포맷 변환 및 통신 지연 측정
    - Action 클라이언트-서버 통신 안정성 테스트

- **Step 5.3: 시스템 테스트 (System Test)**
  - **Gazebo 시뮬레이션 테스트**:
    - Nav2 전체 스택과 Shopee 브리지 통합 시뮬레이션
    - 실제 환경과 유사한 navigation 시나리오 구성
    - 동적 장애물 및 costmap 업데이트 검증
  - **실시간 성능 테스트**:
    - Nav2 플래너-컨트롤러 주기 성능 검증
    - AMCL localization 지연시간 측정
    - Shopee 통신 인터페이스 응답 시간 평가
  - **안전성 테스트**:
    - Nav2 emergency stop 연동 검증
    - AMCL kidnapped robot 복구 테스트
    - 네비게이션 실패 시 recovery behavior 동작 확인

- **Step 5.4: 성능 벤치마크 및 최적화**
  - **Nav2 파라미터 최적화**: 실제 환경에서의 성능 데이터 기반 파라미터 튜닝
  - **메모리 및 CPU 사용량 최적화**: Nav2 스택과 브리지 노드의 리소스 효율성 개선
  - **네트워크 통신 최적화**: Shopee 인터페이스 통신 주기 및 QoS 최적화

## Phase 6: 프로덕션 배포 및 최적화

**목표**: Nav2 기반 시스템을 실제 운영 환경에 배포하고 성능을 최적화합니다.

- **Step 6.1: 배포 환경 구성**
  - **Launch 파일 통합**:
    - `launch/pickee_mobile_nav2.launch.xml`: Nav2와 Shopee 브리지 통합 launch
    - 환경별 설정 파일 분리 (개발/테스트/프로덕션)
    - 자동 실행 스크립트 및 systemd 서비스 구성
  - **Docker 컨테이너 최적화**:
    - Nav2 의존성을 포함한 최적화된 Docker 이미지
    - 멀티스테이지 빌드를 통한 이미지 크기 최소화
    - 런타임 최적화 및 리소스 제한 설정

- **Step 6.2: 모니터링 및 진단 시스템**
  - **성능 메트릭 수집**:
    - Nav2 스택 성능 지표 모니터링
    - AMCL localization 정확도 추적
    - 경로 계획 성공률 및 실행 시간 측정
  - **로그 시스템 구성**:
    - 구조화된 로그 포맷 (JSON) 적용
    - 로그 레벨별 분리 및 자동 순환
    - 원격 로그 수집 및 분석 시스템 연동
  - **Health Check 시스템**:
    - Nav2 노드 상태 자동 모니터링
    - 이상 상황 자동 감지 및 알림
    - 자동 복구 메커니즘 구현

- **Step 6.3: 최종 성능 튜닝**
  - **Nav2 파라미터 최종 최적화**:
    - 실제 환경에서의 성능 데이터 기반 튜닝
    - A* planner 및 DWB controller 파라미터 최적화
    - AMCL 파라미터 환경별 세부 조정
  - **메모리 및 CPU 최적화**:
    - Nav2 스택 메모리 사용량 최적화
    - Shopee 브리지 노드 성능 개선
    - 실시간 제약 조건 충족 검증
  - **네트워크 통신 최적화**:
    - QoS 설정 최적화로 통신 안정성 향상
    - 메시지 크기 및 주기 최적화
    - 대역폭 사용량 모니터링 및 제어
## Phase 7: 품질 보증 및 문서화

**목표**: Nav2 기반 시스템의 품질 보증, 문서화, 그리고 최종 검증을 완료합니다.

- **Step 7.1: 코딩 표준 준수 검증**
  - **자동화된 코드 검사 도구 설정**:
    - clang-format을 통한 코딩 스타일 자동 적용
    - clang-tidy를 통한 정적 분석 및 모범 사례 검증
    - 커밋 전 자동 코드 검사 Hook 설정
  - **ROS2 및 Nav2 표준 준수**:
    - Package Names: `snake_case` (`pickee_mobile`)
    - Node/Topic/Service/Action Names: `snake_case`
    - Nav2 파라미터 네이밍 규칙 준수
    - 타입 및 상수 이름 표준 준수
  - **C++ 표준 준수**:
    - Modern C++17 기능 및 Nav2 호환성 확보
    - RAII 패턴 및 스마트 포인터 활용
    - Thread-safe 코딩 및 const correctness

- **Step 7.2: 종합 문서화**
  - **Nav2 통합 가이드**: Nav2와 Shopee 시스템 연동 방법 상세 문서화
  - **설정 파일 가이드**: Nav2 파라미터 설정 및 튜닝 가이드
  - **개발자 가이드**: 아키텍처, 확장 방법, 트러블슈팅 가이드
  - **성능 벤치마크**: 실행 시간, 메모리 사용량, 처리량 데이터

- **Step 7.3: 최종 통합 테스트 및 배포 준비**
  - **CI/CD 파이프라인 구축**: GitHub Actions를 통한 자동 빌드/테스트
  - **Cross-platform 호환성**: Ubuntu 20.04/22.04 LTS 지원 확인
  - **배포 패키지 최종 검증**: .deb 패키지 및 Docker 이미지 테스트

## 7. 개발 환경 및 도구 (Development Environment)

- **필요 소프트웨어**:
  - ROS2 Jazzy/Humble
  - CMake 3.16+
  - **트러블슈팅 가이드**: Nav2 관련 일반적인 문제 및 해결 방법
  - **API 문서**: Doxygen을 활용한 자동 API 문서 생성

- **Step 7.3: 최종 검증 및 배포**
  - **통합 시나리오 테스트**: 실제 Shopee 환경에서의 완전한 워크플로우 검증
  - **성능 기준 충족 확인**: 모든 요구사항 대비 성능 지표 검증
  - **배포 패키지 준비**: 실제 로봇에 배포 가능한 최종 패키지 구성

## 8. 프로젝트 일정 (Project Timeline)

| Phase | 예상 기간 | 주요 마일스톤 |
|-------|-----------|---------------|
| Phase 1 | 1주 | Nav2 환경 구성 및 기본 구조 완성 |
| Phase 2 | 2주 | Shopee 브리지 노드 핵심 기능 구현 |
| Phase 3 | 1주 | Shopee 통신 인터페이스 완성 |
| Phase 4 | 1주 | Nav2 파라미터 최적화 및 상태 관리 |
| Phase 5 | 2주 | Nav2 통합 테스트 및 검증 |
| Phase 6 | 1주 | 프로덕션 배포 및 최적화 |
| Phase 7 | 1주 | 문서화 및 최종 검증 |
| **총 기간** | **9주** | **Nav2 기반 완전한 구현** |

## 9. 리스크 관리 (Risk Management)

- **기술적 리스크**:
  - Nav2 파라미터 최적화 복잡성: 시뮬레이션 환경에서 사전 튜닝 진행
  - Shopee-Nav2 좌표계 변환 오류: 철저한 단위 테스트 및 검증
  - AMCL localization 불안정성: 다양한 환경에서의 사전 테스트

- **일정 리스크**:
  - Nav2 스택 학습 곡선: 충분한 사전 학습 및 문서 검토
  - 실제 환경 튜닝 시간: 시뮬레이션 환경에서 최대한 사전 최적화

- **품질 리스크**:
  - 코드 품질 저하: 자동화된 코드 검사 및 코드 리뷰 프로세스
  - 테스트 부족: TDD 방법론 적용 및 커버리지 목표 설정 (90% 이상)

## 10. 추가 고려사항

### 10.1. 확장성 및 유지보수성
- **플러그인 아키텍처**: 경로 계획 및 제어 알고리즘의 플러그인화 고려
- **설정 관리**: 런타임 파라미터 변경 및 Hot-reload 지원
- **모니터링 인터페이스**: 실시간 상태 모니터링 대시보드 구현

### 10.2. 국제화 및 현지화
- **다국어 지원**: 로그 메시지 및 사용자 인터페이스 국제화 준비
- **지역별 최적화**: 다양한 환경(실내/실외, 지역별 규제)에 대한 설정 옵션

### 10.3. 보안 및 안전
- **통신 보안**: ROS2 DDS 보안 활성화 및 암호화 통신
- **기능 안전**: ISO 26262 등 안전 표준 준수 검토
- **사이버 보안**: 네트워크 공격에 대한 방어 메커니즘

## 11. 성공 기준 (Success Criteria)

### 11.1. 기능적 요구사항
- ✅ Pickee Main Controller와의 모든 인터페이스 정상 동작
- ✅ 목표 위치까지 ±5cm 정확도로 도착
- ✅ 장애물 감지 및 회피 성공률 99% 이상
- ✅ 배터리 관리 및 자동 충전 기능 완동

### 11.2. 성능 요구사항
- ✅ 제어 주기 100Hz 안정적 달성
- ✅ 센서 데이터 처리 지연시간 10ms 이하
- ✅ CPU 사용률 평상시 30% 이하, 피크 시 70% 이하
- ✅ 메모리 사용량 1GB 이하

### 11.3. 품질 요구사항
- ✅ 단위 테스트 커버리지 90% 이상
- ✅ 24시간 연속 운행 시 메모리 누수 없음
- ✅ 모든 코딩 표준 준수 및 정적 분석 통과
- ✅ 문서화 완성도 95% 이상

이 구현 계획을 통해 Pickee Mobile이 고성능, 고신뢰성을 갖춘 산업용 이동 로봇 플랫폼으로 완성될 것입니다.