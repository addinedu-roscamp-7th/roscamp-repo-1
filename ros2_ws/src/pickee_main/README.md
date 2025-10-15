# Pickee Main Controller

Pickee 로봇의 전체 워크플로우를 관장하는 메인 컨트롤러 패키지

## 테스트 실행 방법

### Phase 5 통합 테스트

0. **기본**:
```bash
# 종속성 설치
rosdep install --from-paths src --ignore-src -r -y
```
1. **전체 테스트 실행**:
```bash
./run_tests.sh
```

1. **단위 테스트만 실행**:
```bash
python3 -m pytest test/test_state_machine.py -v
```

1. **통합 테스트 실행**:
```bash
# 터미널 1: Mock 노드들과 Main Controller 실행
cd ~/tech_research/Shopee/ros2_ws
colcon build --packages-select shopee_interfaces
colcon build --packages-select pickee_main
source install/setup.bash
ros2 launch pickee_main integration_test.launch.py

# 터미널 2: 테스트 클라이언트 실행  
ros2 run pickee_main integration_test_client
```

4. **개별 컴포넌트 테스트**:
```bash
# Mock 노드 개별 실행
ros2 run pickee_main mock_mobile_node
ros2 run pickee_main mock_arm_node  
ros2 run pickee_main mock_vision_node

# Main Controller 실행
ros2 run pickee_main main_controller
```

5. **수동 테스트**:
```bash
# 작업 시작 테스트
ros2 service call /pickee/workflow/start_task shopee_interfaces/srv/PickeeWorkflowStartTask '{robot_id: 1, order_id: 1001, product_list: [{product_id: 1001, location_id: 1001, quantity: 1}]}'

# 상태 모니터링
ros2 topic echo /pickee/robot_status
ros2 topic echo /pickee/mobile/arrival
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
