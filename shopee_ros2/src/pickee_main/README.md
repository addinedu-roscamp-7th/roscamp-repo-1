# Pickee Main Controller

Pickee 로봇의 전체 워크플로우를 관장하는 메인 컨트롤러 패키지

## 주행 알고리즘 집중화 설계

### 새로운 아키텍처
- **PickeeMain**: 고수준 상태 관리 + 정보 전달자 역할
- **PickeeMobile**: 모든 주행 알고리즘 집중 처리

### 핵심 변경 사항
1. **Global Path 계획 제거**: Main에서 경로 계획 로직 제거
2. **장애물 정보 전달**: Vision → Main → Mobile (정보만 전달)
3. **Mobile 자율성 확대**: Mobile이 Global Path 자동 생성 및 동적 수정

### 장점
- 연산 효율성: 중복 계산 제거
- 실시간성: Mobile에서 직접 센서 데이터 처리
- 안전성: 통합된 장애물 처리 알고리즘

## 테스트 실행 방법

### Phase 5 통합 테스트

0. **기본**:
```bash
export PYTHONPATH=/home/wonho/venv/ros_venv/lib/python3.12/site-packages:$PYTHONPATH

# 종속성 설치
rosdep install --from-paths src --ignore-src -r -y
cd ./ros2_ws
colcon build --packages-select shopee_interfaces
colcon build --packages-select pickee_main
source install/setup.bash

2. **개별 컴포넌트 테스트**:
```bash
# 대쉬보드
ros2 run pickee_main dashboard
# Main Controller 실행
ros2 run pickee_main main_controller

# Mock 노드 개별 실행 : 테스트용
ros2 run pickee_main mock_shopee_main

ros2 run pickee_main mock_mobile_node
ros2 run pickee_main mock_arm_node  
ros2 run pickee_main mock_vision_node

```

```

## 프로젝트 구조

```
pickee_main/
├── pickee_main/
│   ├── main_controller.py      # 메인 컨트롤러
│   ├── state_machine.py        # 상태 기계
│   └── states/                 # 상태 구현
├── test/
│   ├── test_state_machine.py   # 단위 테스트
│   ├── mock_nodes/             # Mock 노드들 (실제 하드웨어 없어도 테스트 하기 위한 가짜 컴포넌트)
│   │   ├── mock_arm_node.py        # 실제 로봇 팔 대신 동작 (물건 집기 / 놓기 / 작업 완료 상태)
│   │   ├── mock_mobile_node.py     # 실제 모바일 베이스 대신 동작 (위치 이동 명령, 로봇 현재 위치, 배터리)
│   │   └── mock_vision_node.py     # 실제 카메라/비전 시스템 대신 동작 (제품 감지, 장애물 인식)
│   └── integration/            # 통합 테스트
├── launch/
│   └── integration_test.launch.py  # Launch 파일
└── run_tests.sh               # 테스트 실행 스크립트
```
