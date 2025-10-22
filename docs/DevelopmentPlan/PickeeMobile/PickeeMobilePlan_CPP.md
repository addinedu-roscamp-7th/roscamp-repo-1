# Shopee Pickee Mobile 구현 계획 (C++ 버전)

이 문서는 `PickeeMobileDesign_CPP.md`에 기술된 설계를 바탕으로 Pickee Mobile의 ROS2 C++ 노드 및 컴포넌트를 구현하기 위한 단계별 계획을 정의합니다.

## Phase 1: 프로젝트 설정 및 핵심 구조 구현

**목표**: 코드 작성을 위한 기본적인 프로젝트 구조와 상태 기계의 뼈대를 완성합니다.

- **Step 1.1: ROS2 패키지 생성 및 설정**
  - `pickee_mobile` 패키지 폴더 내에 `package.xml`, `CMakeLists.txt` 등 ROS2 C++ 패키지 필수 파일들을 설정합니다.
  - 의존성 패키지(`rclcpp`, `shopee_interfaces`, `tf2_ros`, `nav2_common`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `Eigen3` 등)를 명시합니다.
  - `include/pickee_mobile/` 및 `src/` 디렉터리 구조를 생성합니다.
  - CMake에 Modern C++17 표준 설정 및 컴파일러 최적화 옵션(`-O3`, `-march=native`)을 추가합니다.

- **Step 1.2: 메인 노드 클래스 생성**
  - `include/pickee_mobile/mobile_controller.hpp` 헤더 파일을 생성하고 `#pragma once` 가드를 사용합니다.
  - `src/mobile_controller.cpp` 파일을 생성하여 `rclcpp::Node`를 상속받는 `PickeeMobileController` 클래스를 구현합니다.
  - 스마트 포인터를 활용한 컴포넌트 멤버 변수 선언: `std::shared_ptr<LocalizationComponent>`, `std::unique_ptr<StateMachine>` 등
  - `src/main.cpp` 파일을 생성하여 노드를 초기화하고 `rclcpp::spin`을 통해 실행하는 메인 함수를 작성합니다.

- **Step 1.3: 상태 기계(State Machine) 프레임워크 구현**
  - 상태의 기반이 될 추상 기본 클래스 `State`를 `include/pickee_mobile/states/state.hpp`에 정의합니다.
  - 순수 가상 함수: `OnEnter()`, `Execute()`, `OnExit()`, `GetType()` 포함
  - 현재 상태를 관리하고 상태 간의 전환을 처리하는 `StateMachine` 클래스를 `include/pickee_mobile/state_machine.hpp`에 구현합니다.
  - `std::unique_ptr<State>`을 활용한 효율적인 상태 전환 및 메모리 관리 구현
  - Exception safety 및 RAII 패턴 적용

- **Step 1.4: 모든 상태(State) 클래스 정의**
  - 설계 문서의 상태 관리에 명시된 모든 상태(`IdleState`, `MovingState`, `StoppedState`, `ChargingState`, `ErrorState`)에 대한 클래스 파일을 생성합니다.
  - `include/pickee_mobile/states/` 디렉터리에 각 상태별 헤더 파일을 생성
  - `src/states/` 디렉터리에 해당 구현 파일들을 생성합니다.
  - 각 클래스는 `State` 클래스를 상속받으며, 초기에는 간단한 로그 출력만 추가합니다.

## Phase 2: 핵심 컴포넌트 구현

**목표**: Pickee Mobile 내부의 주요 컴포넌트들을 고성능 C++로 구현하고 연동합니다.

- **Step 2.1: LocalizationComponent 구현**
  - `include/pickee_mobile/components/localization_component.hpp` 및 `src/components/localization_component.cpp` 파일을 생성합니다.
  - Eigen 라이브러리를 활용한 고성능 행렬 연산 및 SIMD 최적화를 적용합니다.
  - 센서 데이터(LiDAR, IMU, Encoder) 융합을 위한 EKF(Extended Kalman Filter) C++ 구현
  - `Eigen::Vector3d current_pose_`와 `Eigen::Matrix3d pose_covariance_`를 사용한 위치 추정
  - `rclcpp::TimerBase`를 사용하여 100ms 주기로 위치 정보를 발행
  - `tf2_ros::TransformBroadcaster`를 사용하여 Transform 발행
  - 메모리 효율적인 센서 데이터 처리 및 스마트 포인터 활용

- **Step 2.2: PathPlanningComponent 구현**
  - `include/pickee_mobile/components/path_planning_component.hpp` 및 `src/components/path_planning_component.cpp` 파일을 생성합니다.
  - A* 알고리즘의 고성능 C++ 구현으로 Grid-based Global Path Planning 수행
  - DWA(Dynamic Window Approach) 또는 TEB 알고리즘의 실시간 Local Path Planning 구현
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
  - `rclcpp::Publisher<shopee_interfaces::msg::PickeeMobilePose>::SharedPtr pose_publisher_` 구현
  - `rclcpp::TimerBase::SharedPtr pose_timer_`를 사용한 100ms 주기 위치 보고
  - `rclcpp::Publisher<shopee_interfaces::msg::PickeeMobileArrival>::SharedPtr arrival_publisher_` 구현
  - 메시지 발행 시 메모리 효율성 및 실시간 성능 고려
  - `rclcpp::QoS` 설정을 통한 통신 신뢰성 향상

- **Step 3.3: Service Server 구현**
  - `/pickee/mobile/move_to_location` 서비스 서버 구현
  - `/pickee/mobile/update_global_path` 서비스 서버 구현
  - 서비스 콜백에서 `std::bind`, 람다 함수, `std::function` 활용
  - 비동기 서비스 처리 및 exception safety 보장
  - 요청 검증 및 응답 생성 로직 구현

- **Step 3.4: Subscriber 구현**
  - `rclcpp::Subscription<shopee_interfaces::msg::PickeeMobileSpeedControl>::SharedPtr speed_control_subscription_` 구현
  - 콜백 함수에서 수신된 명령의 내부 컴포넌트 전달
  - 메시지 큐 관리 및 처리 우선순위 설정
  - Thread-safe 데이터 전달 보장

- **Step 3.5: Transform 관리 구현**
  - `std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_` 구현
  - `std::shared_ptr<tf2_ros::TransformListener> tf_listener_` 및 `std::unique_ptr<tf2_ros::Buffer> tf_buffer_` 구현
  - Transform 발행/수신 및 시간 동기화 처리

## Phase 4: 상태 로직 및 파라미터 시스템 구현

**목표**: 각 상태의 세부 동작 로직과 파라미터 시스템을 완성합니다.

- **Step 4.1: 상태별 세부 로직 구현**
  - **IdleState 구현**: 
    - Pickee Main Controller의 명령 대기 로직
    - `/pickee/mobile/move_to_location` 서비스 요청 수신 시 `MovingState`로 안전한 상태 전환
    - 상태 진입/이탈 시 리소스 정리 및 초기화
  - **MovingState 구현**:
    - 경로 계획 및 추종 로직 실행
    - 실시간 위치 추정 및 주기적 상태 보고
    - 목적지 도착 감지 시 `IdleState` 전환 또는 장애물 감지 시 `StoppedState` 전환
    - Vision 장애물 정보 기반 동적 경로 수정
  - **StoppedState 구현**:
    - 안전 정지 상태 유지 및 모터 제어 중단
    - `/pickee/mobile/speed_control` 토픽 모니터링 및 재시작 조건 대기
    - Emergency stop 해제 로직 구현
  - **ChargingState 구현**:
    - 배터리 충전 모니터링 및 충전소 도킹 제어
    - 배터리 임계값 달성 시 `IdleState` 복귀
  - **ErrorState 구현**:
    - 시스템 오류 진단 및 복구 시도
    - 로그 기록 및 오류 보고 메커니즘
    - 수동 복구 대기 또는 자동 재시작 로직

- **Step 4.2: 파라미터 관리 시스템 구현**
  - **파라미터 선언 및 로딩**:
    - `DeclareParameters()` 및 `LoadParameters()` 메서드 구현
    - 설계 문서의 모든 파라미터에 대한 타입 안전 처리
    - 런타임 파라미터 변경 감지 및 적용
  - **YAML 설정 파일**:
    - `config/pickee_mobile_params.yaml` 파일 생성
    - 모든 파라미터의 기본값 및 설명 포함
    - Launch 파일에서 파라미터 로딩 설정

- **Step 4.3: 오류 처리 및 로깅 시스템 구현**
  - **Exception Safety**: 
    - `SafeExecute()` 메서드를 통한 통합 예외 처리
    - RAII 패턴 및 스마트 포인터 활용
    - 오류 발생 시 `ErrorState` 전환 로직
  - **성능 프로파일링**:
    - `PerformanceTimer` 클래스를 통한 실행 시간 측정
    - 중요 경로의 성능 모니터링 및 로깅
    - Debug/Release 빌드별 로깅 레벨 조정

## Phase 5: 테스트 및 검증

**목표**: 구현된 기능의 신뢰성, 성능, 안전성을 체계적으로 검증합니다.

- **Step 5.1: 단위 테스트 (Unit Test)**
  - **Google Test 프레임워크 설정**:
    - `test/` 디렉터리에 `ament_cmake_gtest` 활용
    - `CMakeLists.txt`에 테스트 타겟 추가
  - **핵심 컴포넌트 테스트**:
    - `test_state_machine.cpp`: 상태 전환 로직 및 메모리 관리 테스트
    - `test_localization_component.cpp`: Eigen 기반 위치 추정 정확도 테스트
    - `test_path_planning_component.cpp`: A*/DWA 알고리즘 성능 테스트
    - `test_motion_control_component.cpp`: PID 제어기 안정성 테스트
    - `test_sensor_manager.cpp`: 센서 데이터 융합 및 필터링 테스트
  - **Memory Leak 및 Performance 테스트**:
    - Valgrind, AddressSanitizer 호환성 검증
    - 실행 시간 및 메모리 사용량 벤치마크

- **Step 5.2: 통합 테스트 (Integration Test)**
  - **Mock 시스템 구현**:
    - `test/mock/` 디렉터리에 C++ Mock 클래스들 구현
    - `MockPickeeMainController`: 외부 인터페이스 시뮬레이션
    - `MockSensorNodes`: 센서 데이터 시뮬레이션
  - **Launch 파일 테스트**:
    - `launch/test_integration.launch.xml`: 전체 시스템 통합 실행
    - 컴포넌트 간 통신 및 데이터 흐름 검증
    - ROS2 QoS 및 네트워크 지연 시뮬레이션

- **Step 5.3: 시스템 테스트 (System Test)**
  - **Gazebo 시뮬레이션 테스트**:
    - 실제 환경과 유사한 시뮬레이션 시나리오 구성
    - Navigation stack과의 완전한 통합 테스트
    - 장애물 회피 및 동적 경로 수정 시나리오 검증
  - **실시간 성능 테스트**:
    - 제어 주기(50-100Hz) 달성 여부 검증
    - 센서 데이터 처리 지연시간 측정
    - CPU/메모리 사용률 모니터링
  - **안전성 테스트**:
    - Emergency stop 반응 시간 측정
    - 통신 단절 시 안전 모드 전환 검증
    - 예외 상황에서의 시스템 복구 능력 테스트

- **Step 5.4: 스트레스 테스트 및 내구성 검증**
  - **장기간 운행 테스트**: 24시간 연속 운행을 통한 메모리 누수 및 성능 저하 검증
  - **고부하 시나리오**: 복잡한 환경에서의 다중 장애물 회피 성능 테스트
  - **네트워크 불안정 상황**: 통신 지연 및 패킷 손실 환경에서의 동작 검증

## Phase 6: 성능 최적화 및 프로덕션 준비

**목표**: C++ 고유의 성능 최적화를 통해 실제 운영 환경에서의 최적 성능을 달성합니다.

- **Step 6.1: 메모리 관리 최적화**
  - **스마트 포인터 최적화**:
    - `std::shared_ptr`, `std::unique_ptr`을 통한 자동 메모리 관리
    - 참조 카운트 오버헤드 최소화를 위한 `std::unique_ptr` 우선 사용
    - `std::make_shared`, `std::make_unique` 활용으로 메모리 단편화 방지
  - **메모리 풀링 시스템**:
    - 고빈도 메시지 처리를 위한 `ObjectPool` 템플릿 클래스 구현
    - 센서 데이터 및 경로 계획 객체의 재사용 최적화
    - RAII 패턴 적용으로 자동 리소스 관리
  - **메모리 지역성 최적화**:
    - 캐시 친화적인 데이터 구조 설계
    - 연속된 메모리 레이아웃을 위한 `std::vector` 활용

- **Step 6.2: 실시간 성능 최적화**
  - **Real-time Executor 구성**:
    - `rclcpp::executors::SingleThreadedExecutor` 최적화
    - CPU 친화성(CPU affinity) 설정으로 성능 향상
    - 우선순위 기반 스케줄링 적용
  - **Lock-free 프로그래밍**:
    - `LockFreeRingBuffer` 템플릿을 활용한 센서 데이터 처리
    - `std::atomic` 변수를 통한 thread-safe 상태 관리
    - Lock contention 최소화를 위한 설계
  - **알고리즘 최적화**:
    - 중요 경로의 시간 복잡도 개선
    - 캐싱 전략 구현 (경로 계획 결과, 센서 필터링 등)
    - SIMD 명령어 활용 (Eigen 라이브러리 최적화)

- **Step 6.3: 컴파일러 및 빌드 최적화**
  - **컴파일러 최적화**:
    - `CMakeLists.txt`에 `-O3 -march=native -flto` 옵션 추가
    - Profile-Guided Optimization (PGO) 적용 고려
    - 컴파일러별 최적화 옵션 세분화
  - **헤더 파일 최적화**:
    - `#pragma once` 사용 및 전방 선언 적극 활용
    - 불필요한 헤더 포함 최소화
    - 컴파일 시간 단축을 위한 모듈화
  - **템플릿 최적화**:
    - 템플릿 특수화를 통한 성능 향상
    - 컴파일 시간 최적화를 위한 SFINAE 패턴 활용
    - 인라인 함수 최적화

- **Step 6.4: 프로덕션 환경 준비**
  - **배포 패키지 구성**:
    - Debian 패키지 생성을 위한 설정
    - Docker 컨테이너 이미지 최적화
    - 설치 스크립트 및 시스템 서비스 등록
  - **모니터링 및 진단 도구**:
    - 실시간 성능 모니터링 대시보드
    - 로그 분석 및 원격 진단 기능
    - 자동 성능 리포트 생성
  - **보안 및 안정성**:
    - 코드 정적 분석 도구 통합 (cppcheck, clang-analyzer)
    - 메모리 안전성 검증 (AddressSanitizer, MemorySanitizer)
    - 네트워크 보안 및 암호화 적용

## Phase 7: 품질 보증 및 문서화

**목표**: 코딩 표준 준수, 문서화, 그리고 최종 품질 검증을 완료합니다.

- **Step 7.1: 코딩 표준 준수 검증**
  - **자동화된 코드 검사 도구 설정**:
    - clang-format을 통한 코딩 스타일 자동 적용
    - clang-tidy를 통한 정적 분석 및 모범 사례 검증
    - 커밋 전 자동 코드 검사 Hook 설정
  - **ROS2 표준 준수**:
    - Package Names: `snake_case` (`pickee_mobile`)
    - Node/Topic/Service/Action/Parameter Names: `snake_case`
    - Type Names: `PascalCase` (`PickeeMobileController`)
    - Type Field Names: `snake_case`
    - Type Constants Names: `SCREAMING_SNAKE_CASE`
  - **C++ 표준 준수**:
    - Modern C++17 기능 적극 활용
    - `std::` 접두사 명시적 사용 (using namespace std 금지)
    - const correctness 및 RAII 패턴 준수
    - 스마트 포인터 사용 및 메모리 안전성 확보

- **Step 7.2: 종합 문서화**
  - **API 문서**: Doxygen을 활용한 자동 API 문서 생성
  - **사용자 가이드**: 설치, 설정, 실행 방법 상세 문서화
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
  - GCC 9+ 또는 Clang 10+ (C++17 지원)
  - Eigen3 라이브러리
  - Google Test 프레임워크
  - Gazebo 시뮬레이터

- **권장 개발 도구**:
  - VS Code + ROS2 확장
  - clang-format, clang-tidy
  - Valgrind, AddressSanitizer
  - Git + GitLab/GitHub
  - Doxygen 문서 생성 도구

## 8. 프로젝트 일정 (Project Timeline)

| Phase | 예상 기간 | 주요 마일스톤 |
|-------|-----------|---------------|
| Phase 1 | 1주 | 프로젝트 구조 및 상태 기계 완성 |
| Phase 2 | 2주 | 핵심 컴포넌트 구현 완료 |
| Phase 3 | 1주 | 외부 통신 인터페이스 완성 |
| Phase 4 | 1주 | 상태 로직 및 파라미터 시스템 완료 |
| Phase 5 | 2주 | 전체 테스트 및 검증 완료 |
| Phase 6 | 1주 | 성능 최적화 및 프로덕션 준비 |
| Phase 7 | 1주 | 문서화 및 최종 배포 준비 |
| **총 기간** | **9주** | **완전한 제품 수준 구현** |

## 9. 리스크 관리 (Risk Management)

- **기술적 리스크**:
  - Nav2 통합 복잡성: 단계적 통합 및 Mock 테스트로 완화
  - 실시간 성능 요구사항: 조기 성능 테스트 및 최적화 우선순위 설정
  - 메모리 관리 복잡성: 스마트 포인터 및 RAII 패턴 엄격 적용

- **일정 리스크**:
  - 예상보다 긴 디버깅 시간: 충분한 단위 테스트 및 조기 통합 테스트
  - 성능 최적화 복잡성: 핵심 기능 우선 구현 후 점진적 최적화

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