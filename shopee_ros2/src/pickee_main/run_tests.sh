#!/bin/bash

# Shopee Pickee Main Controller 테스트 실행 스크립트

echo "=============================================="
echo "  Shopee Pickee Main Controller 테스트 실행  "
echo "=============================================="

# 색깔 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ROS2 환경 설정 확인
echo -e "${BLUE}1. ROS2 환경 설정 확인 중...${NC}"
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}ROS2 환경이 설정되지 않았습니다. 자동으로 설정합니다.${NC}"
    source /opt/ros/humble/setup.bash
fi

echo "ROS_DISTRO: $ROS_DISTRO"

# 워크스페이스 경로 설정
WORKSPACE_ROOT="/home/wonho/tech_research/Shopee/ros2_ws"
PKG_PATH="$WORKSPACE_ROOT/src/pickee_main"

echo -e "${BLUE}2. 워크스페이스 경로: ${WORKSPACE_ROOT}${NC}"

# 현재 디렉토리를 패키지 경로로 변경
cd "$PKG_PATH"

echo -e "${BLUE}3. 단위 테스트 실행 중...${NC}"
echo "----------------------------------------"

# 단위 테스트 실행 - pytest 사용
if command -v pytest &> /dev/null; then
    echo "pytest로 단위 테스트 실행 중..."
    python3 -m pytest test/test_state_machine.py -v
    UNIT_TEST_RESULT=$?
else
    echo "pytest가 설치되지 않음. unittest로 실행 중..."
    python3 -m unittest test.test_state_machine -v
    UNIT_TEST_RESULT=$?
fi

if [ $UNIT_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ 단위 테스트 통과${NC}"
else
    echo -e "${RED}✗ 단위 테스트 실패${NC}"
fi

echo
echo -e "${BLUE}4. 패키지 빌드 중...${NC}"
echo "----------------------------------------"

# 워크스페이스로 이동
cd "$WORKSPACE_ROOT"

# 종속성 확인 및 설치
echo "rosdep을 사용하여 종속성 확인 중..."
rosdep install --from-paths src --ignore-src -r -y

# 패키지 빌드
echo "colcon을 사용하여 패키지 빌드 중..."
colcon build --packages-select pickee_main

BUILD_RESULT=$?

if [ $BUILD_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ 패키지 빌드 성공${NC}"
    
    # 환경 설정 소싱
    source install/setup.bash
    
    echo
    echo -e "${BLUE}5. 통합 테스트 준비 완료${NC}"
    echo "----------------------------------------"
    echo -e "${YELLOW}통합 테스트를 시작하려면 다음 명령어들을 사용하세요:${NC}"
    echo
    echo "새 터미널에서:"
    echo -e "${GREEN}  cd $WORKSPACE_ROOT${NC}"
    echo -e "${GREEN}  source install/setup.bash${NC}"
    echo -e "${GREEN}  ros2 launch pickee_main integration_test.launch.py${NC}"
    echo
    echo "다른 터미널에서 테스트 클라이언트 실행:"
    echo -e "${GREEN}  ros2 run pickee_main integration_test_client${NC}"
    echo
    echo "개별 Mock 노드 실행 (디버깅용):"
    echo -e "${GREEN}  ros2 run pickee_main mock_mobile_node${NC}"
    echo -e "${GREEN}  ros2 run pickee_main mock_arm_node${NC}"
    echo -e "${GREEN}  ros2 run pickee_main mock_vision_node${NC}"
    echo
    echo "토픽 모니터링:"
    echo -e "${GREEN}  ros2 topic list${NC}"
    echo -e "${GREEN}  ros2 topic echo /pickee/robot_status${NC}"
    echo -e "${GREEN}  ros2 topic echo /pickee/mobile/arrival${NC}"
    echo
    echo "서비스 테스트:"
    echo -e "${GREEN}  ros2 service call /pickee/workflow/start_task shopee_interfaces/srv/PickeeWorkflowStartTask '{robot_id: 1, order_id: 1001, product_list: [{product_id: \"P001\", location_id: \"L001\", quantity: 1}]}'${NC}"
    
else
    echo -e "${RED}✗ 패키지 빌드 실패${NC}"
    echo "빌드 로그를 확인하여 오류를 해결하세요."
fi

echo
echo "=============================================="
echo "              테스트 스크립트 완료            "
echo "=============================================="

# 결과 요약
echo -e "${BLUE}테스트 결과 요약:${NC}"
if [ $UNIT_TEST_RESULT -eq 0 ]; then
    echo -e "  단위 테스트: ${GREEN}통과${NC}"
else
    echo -e "  단위 테스트: ${RED}실패${NC}"
fi

if [ $BUILD_RESULT -eq 0 ]; then
    echo -e "  패키지 빌드: ${GREEN}성공${NC}"
else
    echo -e "  패키지 빌드: ${RED}실패${NC}"
fi