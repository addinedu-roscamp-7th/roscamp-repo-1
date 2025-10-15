# Packee Arm Controller 개발 계획

## 1단계: 기반 설정 (Foundation)
- **목표**: 양팔 로봇 제어를 위한 ROS2 노드 및 하드웨어 연동
- **세부 작업**:
  1. **ROS2 노드 생성**: `packee_arm_controller` 노드 생성
  2. **하드웨어 드라이버 연동**: 양팔(Dual-Arm) 및 그리퍼의 ROS2 드라이버를 연동하고 개별 제어 테스트

## 2단계: 양팔 모션 플래닝 (Dual-Arm Motion Planning)
- **목표**: MoveIt 2를 이용한 양팔 협응 및 충돌 회피 기능 구현
- **세부 작업**:
  1. **MoveIt 설정**: 양팔 로봇의 URDF 파일을 기반으로 MoveIt 설정 파일 생성. 특히 양팔 간, 그리고 로봇 자신과의 충돌을 방지하는 설정이 중요.
  2. **협응 테스트**: 한 팔이 다른 팔의 작업 공간을 침범하지 않으면서, 동시에 작업을 수행하거나 순차적으로 작업을 수행하는 경로 계획 기능 테스트

## 3단계: 인터페이스 구현 (Interface)
- **목표**: `Pac_Main_vs_Pac_Arm.md` 명세에 따른 인터페이스 구현
- **세부 작업**:
  1. **서비스 서버 구현**: `/packee/arm/move_to_pose`, `/pick_product`, `/place_product` 서비스 요청을 받아, 지정된 팔(`arm_side`)이 동작을 수행하도록 MoveIt API를 호출하는 서버 구현
  2. **토픽 발행 구현**: 각 팔의 자세 변경, 픽업, 담기 등 동작의 진행 상태를 `/packee/arm/pose_status`, `/pick_status`, `/place_status` 토픽으로 발행

## 4단계: 포장 작업 고도화 (Fine-tuning)
- **목표**: 다양한 형태의 상품을 안정적으로 집고 놓는 기능 고도화
- **세부 작업**:
  1. **Grasp Planning**: Vision이 제공하는 상품 정보를 바탕으로, 상품의 형태와 무게 중심을 고려하여 최적의 파지(Grasping) 자세와 힘을 결정하는 로직 구현
  2. **Place Planning**: 포장 박스 내에 상품을 순서대로, 그리고 안정적으로 쌓기 위한 최적의 위치와 자세를 결정하는 로직 구현
# Packee Arm Controller 상세 설계 및 개발 계획

Packee Arm Controller는 Packee Main Controller 지시에 따라 양팔 로봇의 시각 서보 기반 제어를 수행하고, 작업 진행 상황을 실시간으로 보고하는 실행 계층이다. `Pac_Main_vs_Pac_Arm.md` 명세와 `SC_03_3`, `SC_03_4` 시퀀스 다이어그램을 기준으로 MoveIt2 대신 IBVS(Image-Based Visual Servoing)를 도입한 설계 내용을 정리한다.

## 1. 시스템 역할 요약
- **프리셋 포즈 정렬**: `/packee/arm/move_to_pose` 서비스 요청을 받아 `cart_view`, `standby` 등 미리 정의된 관절 세트로 양팔을 이동시키고 `/packee/arm/pose_status` 토픽으로 결과 보고.
- **IBVS 기반 픽업**: Vision이 제공한 2D 이미지 평면 좌표(픽셀 혹은 정규화 좌표)를 추적하면서 IBVS로 엔드이펙터를 정렬하고, 그리퍼 동작까지 완료하면 `/packee/arm/pick_status`로 단계별 상태 발행.
- **IBVS 기반 담기**: 포장 박스 내 투영 좌표를 목표로 IBVS 정렬을 수행한 뒤 상품을 내려놓고 `/packee/arm/place_status`에 완료/실패 상태 보고.
- **오류 감시 및 보고**: 서비스 응답으로 수락 여부를 즉시 회신하고, 제어 루프 중 발생한 비정상 상황(타깃 상실, 속도 제한 초과, 하드웨어 오류)을 상태 토픽의 `status=failed`, `message` 필드로 전달.

## 2. 노드 아키텍처
- **ROS2 노드**: `packee_arm_controller` (C++ 또는 Python). 3개의 서비스 서버와 3개의 상태 퍼블리셔 운영.
- **주요 컴포넌트**
  - `VisionTargetSubscriber`: Vision 노드가 발행하는 2D 목표 포인트/타깃 ID를 구독하고 최신 값을 공유 메모리에 보관.
  - `IBVSController`: 카메라 내·외부 파라미터, 추정 깊이, 현재 엔드이펙터 투영 좌표를 활용해 속도 트위스트(또는 조인트 속도) 명령을 계산.
  - `PresetPoseManager`: `move_to_pose` 요청 시 사용할 관절 목표를 로드하고 `ArmCommandRouter`에 전달.
  - `ArmCommandRouter`: 좌·우 팔 하드웨어 인터페이스 래퍼. 속도/위치 명령을 전송하고 안전 한계 속도를 적용.
  - `TaskCoordinator`: 서비스 요청 큐 관리, 팔 별 동시성 제어, 진행 단계 전환 로직 수행.
  - `StatusPublisher`: Pose/Pick/Place 상태 메시지 구성 및 발행.
  - `SafetyMonitor`: 포스/토크 센서, 비상정지, 목표 상실 이벤트를 감지하여 제어 중단 및 실패 리포트.

## 3. 데이터 및 상태 모델
- **타깃 표현**: `TargetPoint2D` 구조(타깃 ID, 2D 좌표, 추정 깊이/스케일, 타임스탬프). Vision이 좌표 갱신을 멈추면 타임아웃 처리.
- **IBVS 파라미터**: 카메라 내참(`fx`, `fy`, `cx`, `cy`), 로봇-카메라 외참(TF), 깊이 추정 상한/하한, 게인 행렬. YAML 파라미터로 관리.
- **진행 상태 모델**
  - Pose 상태: `status`는 `in_progress`/`completed`/`failed`, `progress`는 관절 궤적 완료 비율, `message`는 현재 단계 설명.
  - Pick/Place 상태: `current_phase`는 `acquiring_target`, `aligning`, `approaching`, `grasping`, `retreating`, `done` 중 하나. 실패 시 해당 단계 유지.
- **요청 큐 정책**
  - Pose 명령은 최신 요청 우선으로 덮어쓰기.
  - Pick/Place는 팔 별 큐를 운용하고, 동일 팔에 중첩 요청이 오면 `accepted=false` 또는 큐잉 정책 선택 후 문서화.
  - 좌/우 팔 동시 작업 허용 시 공통 자원(카메라, 협동 공간)에 대한 충돌 규칙 정의.

## 4. 주요 플로우 상세
### 4.1 Preset Pose Alignment
1. `/move_to_pose` 요청 수신 → 유효성 검증(존재하는 `pose_type`, 팔 사용 가능 여부).
2. `PresetPoseManager`가 관절 목표 불러오기 → `ArmCommandRouter`가 속도 프로파일을 적용해 실행.
3. `TaskCoordinator`가 진행률 추적 → `/pose_status` 발행.
4. 완료 시 `status=completed`, 실패/취소 시 `status=failed`로 보고.

### 4.2 Pick Product with IBVS
1. 요청 파싱 → `TaskCoordinator`가 해당 `arm_side` 슬롯 확보 → 서비스 응답 `accepted=true`.
2. `VisionTargetSubscriber`의 최신 타깃 좌표 확인. 없으면 `status=failed`, `message="target lost"`.
3. `IBVSController`가 현재 엔드이펙터 투영 좌표를 계산(카메라-TF + 포워드 키네매틱스)하고 속도 명령 생성.
4. 단계 전환
   - `acquiring_target`: 타깃 좌표 유효성 확보, 초기 정렬 시작.
   - `aligning`: IBVS 루프에서 오차(픽셀 차이)가 임계값 이하로 수렴할 때까지 반복.
   - `approaching`: 엔드이펙터를 목표 깊이로 전진(보간된 속도 명령).
   - `grasping`: 그리퍼 폐합, 힘 센서 체크.
   - `retreating`: 안전한 분리 거리까지 후퇴.
   - `done`: Pick 완료.
5. 각 단계별 `/pick_status` 업데이트. 타깃 상실, 힘 초과, 시간 초과 시 실패 처리.

### 4.3 Place Product with IBVS
1. Packee Main이 전달한 박스 투영 좌표 또는 내부 포장 템플릿에서 목표 포인트 로드.
2. `acquiring_target` 단계에서 투영 좌표 확인 후 `aligning`/`approaching` 절차 수행(상품 위치 대신 박스 투영 좌표 사용).
3. `grasping` 단계에서 그리퍼 오픈 및 하중 감소 확인 → `retreating`으로 전환하며 안전 거리 확보.
4. 상태 토픽에 `current_phase` 갱신, 완료 시 `status=completed`, Failure 시 원인 메시지 기록.

## 5. 예외 처리 및 복구 전략
- 타깃 상실/지연: Vision 타임아웃 또는 잔차 발산 시 즉시 속도 명령 정지, 실패 보고.
- 속도·힘 제한 초과: `SafetyMonitor`가 명령 클램핑 또는 E-stop 신호를 발신하고 실패 메시지에 원인 포함.
- 동시 작업 충돌: 카메라 시야 공유로 좌/우 팔 타깃이 충돌하면 우선순위 정책에 따라 한쪽을 Pause/Retry.
- 안전 포즈 복귀: 실패 후 자동으로 `standby` 포즈 명령 트리거 옵션 제공, 실패가 연속되면 Packee Main에 경보.

## 6. 연동 고려 사항
- Vision → Arm 좌표계 정합: TF2 트리에서 카메라 프레임과 각 팔의 엔드이펙터 프레임을 지속적으로 브로드캐스트.
- 깊이 추정: 기본적으로 2D 좌표 기반이나, 필요 시 Vision이 제공하는 추정 깊이/크기 정보를 사용하도록 필드 예약.
- 상태 토픽 지연: IBVS 루프 주기가 짧으므로 상태 업데이트는 10 Hz 정도로 제한하여 Packee Main에 과도한 로드 방지.
- 로깅 및 디버깅: 2D 타깃 좌표, 잔차, 속도 명령을 rosbag/CSV로 기록해 튜닝 및 회귀 테스트에 활용.

## 7. 테스트 전략
- **단위 테스트**: 서비스 핸들러, 상태 변환 로직, 잔차 계산 유틸리티를 모의 객체(Mock) 기반으로 검증.
- **HIL/시뮬레이션**: Gazebo 또는 하드웨어-인-더-루프 환경에서 가상 카메라 피드와 IBVS 제어 루프 검증.
- **통합 테스트**: Packee Main·Vision과 `SC_03_3`, `SC_03_4` 시나리오 자동화. 타깃 상실, 깊이 오차 등 경계 상황 포함.
- **장애 주입 테스트**: 노이즈가 많은 좌표, 갑작스러운 목표 전환, 그리퍼 실패 등 조건을 주입해 실패 보고 및 복구 흐름 검증.

## 8. 개발 계획

### 단계 0: 센서 및 좌표계 정합
- 카메라 캘리브레이션(내·외부 파라미터) 확보, TF 브로드캐스트 템플릿 작성.
- 인터페이스 메시지/서비스 빌드 확인(`shopee_interfaces` 의존성 점검).

### 단계 1: 노드 골격 및 인터페이스 스텁
- `packee_arm_controller` 노드 생성.
- 3개 서비스 서버·상태 퍼블리셔 뼈대 작성, 더미 응답으로 통신 검증.

### 단계 2: IBVS 제어 루프 기본 구현
- `VisionTargetSubscriber`, `IBVSController`, `ArmCommandRouter`간 데이터 파이프 구축.
- 잔차 계산, 속도 명령 생성, 속도 제한 적용 로직 개발.

### 단계 3: Pick 파이프라인 완성
- 단계별 상태 머신 구현, 그리퍼 제어 연동.
- 타깃 상실/시간 초과 등 실패 분기 처리.

### 단계 4: Place 파이프라인 완성
- 박스 포인트 정렬 로직 및 상품 릴리즈 제어.
- Pick → Place 연속 시나리오 자동화(`SC_03_4`).

### 단계 5: 프리셋 포즈 및 스케줄링
- `PresetPoseManager` 구현, `/move_to_pose` 서비스 연결.
- `TaskCoordinator` 동시성 정책 확정(팔 별 큐, 우선순위).

### 단계 6: 통합 및 안정화
- Vision/Packee Main과 합동 테스트, 상태 토픽 기반 모니터링 검증.
- 제어 게인 튜닝, 노이즈 대응 필터(저역통과 등) 적용.

### 단계 7: 검증 및 릴리즈 준비
- 자동화 테스트 스위트 정리, rosbag 재생 기반 회귀 테스트.
- 운영 환경 하드웨어 리허설, 장애 대응 시나리오 문서화.

## 9. 산출물 체크리스트
- ROS2 패키지 코드 및 설정 파일(IBVS 파라미터 YAML, TF 설정, 프리셋 포즈 정의).
- 상세 API 문서(서비스/토픽 사용법, 상태 코드, 실패 케이스 정의).
- 테스트 리포트: 시뮬레이션, HIL, 장애 주입 결과.
- 운영 매뉴얼: 초기화 절차, 센서 캘리브레이션, 복구 절차, 주요 로그 위치.
