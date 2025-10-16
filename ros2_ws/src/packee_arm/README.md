# Shopee Packee Arm Controller

Packee 로봇의 양팔(Dual-Arm) 작업을 담당하는 ROS 2 패키지입니다.  
서비스 기반 API를 통해 자세 이동, 상품 픽업, 상품 담기 명령을 처리하며,
각 단계의 상태를 토픽으로 브로드캐스트합니다.

## 패키지 구성
- `src/packee_arm_controller.cpp`  
  실제 로봇 제어를 추상화한 메인 노드로, 서비스 요청을 받아 작업을 실행하고
  진행 상황을 토픽으로 퍼블리시합니다.
- `src/mock_packee_main.cpp`  
  Packee 메인 서비스가 없는 환경에서 Arm 컨트롤러 인터페이스를 시험할 수 있는
  모의(Node) 클라이언트입니다.
- `resource/`  
  ROS 2 실행 파일 등록을 위한 `packee_arm_controller`, `mock_packee_main` 리소스 파일.

## 퍼블리시/구독 인터페이스
- `/packee/arm/pose_status` (`shopee_interfaces/msg/ArmPoseStatus`)  
  자세 이동 명령 진행 상황을 전송합니다.
- `/packee/arm/pick_status` (`shopee_interfaces/msg/PackeeArmTaskStatus`)  
  상품 픽업 작업 상태를 브로드캐스트합니다.
- `/packee/arm/place_status` (`shopee_interfaces/msg/PackeeArmTaskStatus`)  
  상품 담기 작업 상태를 브로드캐스트합니다.

## 서비스 인터페이스
- `/packee/arm/move_to_pose` (`shopee_interfaces/srv/PackeeArmMoveToPose`)  
  `pose_type`(예: `cart_view`, `standby`)에 따라 로봇 팔 자세를 변경합니다.
- `/packee/arm/pick_product` (`shopee_interfaces/srv/PackeeArmPickProduct`)  
  지정한 `robot_id`, `order_id`, `product_id`, `arm_side`로 픽업 작업을 수행합니다.
- `/packee/arm/place_product` (`shopee_interfaces/srv/PackeeArmPlaceProduct`)  
  픽업한 상품을 포장 영역으로 이동해 담는 과정을 처리합니다.

각 서비스 요청은 입력 값 검증을 거친 뒤, 진행 상황을 `status`, `current_phase`,
`progress`, `message` 필드로 요약해 관련 토픽에 퍼블리시합니다.

## 빌드 방법
```bash
cd <workspace>  # 예: roscamp-repo-1/ros2_ws
colcon build --packages-select packee_arm
source install/setup.bash
```

## 실행 예시
1. Arm 컨트롤러 노드 실행
   ```bash
   ros2 run packee_arm packee_arm_controller
   ```
2. (선택) 모의 메인 노드 실행  
   Arm 서비스를 시험하기 위한 상태 머신 기반 클라이언트입니다.
   ```bash
   ros2 run packee_arm mock_packee_main \
     --ros-args \
       -p robot_id:=1 \
       -p order_id:=100 \
       -p product_id:=501 \
       -p arm_side:=left \
       -p arm_sides:=left,right
   ```

## 파라미터 (Mock Node)
- `robot_id` (기본: `1`)  
  테스트할 로봇 ID.
- `order_id` (기본: `100`)  
  테스트 주문 ID.
- `product_id` (기본: `501`)  
  시작 상품 ID. 양팔 테스트 시 인덱스만큼 증가합니다.
- `arm_side` (기본: `left`)  
  단일 팔 테스트 시 사용되는 팔 구분.
- `arm_sides` (기본: `left,right`)  
  CSV 형식의 팔 목록. 예: `left,right` 또는 `left`.

## 개발 참고 사항
- 현재 구현은 하드웨어 의존 로직 없이 상태 업데이트를 시뮬레이션하는 형태입니다.
- 새로운 포즈 타입이나 작업 단계를 도입할 경우, 유효성 검증 집합(`valid_pose_types_`,
  `valid_arm_sides_`)과 상태 퍼블리시 로직을 함께 업데이트해야 합니다.
