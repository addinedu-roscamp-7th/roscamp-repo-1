
# Shopee Packee Main Controller 상세 설계

## 1. 개요 (Overview)

`packee_main_controller`는 Packee 로봇의 두뇌 역할을 하는 중앙 제어 노드입니다. 전체 시스템 아키텍처에서 Shopee Main Service로부터 태스크를 할당받아 Packee의 하위 컴포넌트(Arm, Vision)를 총괄 제어하여 주어진 임무를 수행하고, 로봇의 상태를 지속적으로 Main Service에 보고합니다.

**주요 책임:**
- Packee 로봇의 상태를 `StateDiagram_Packee.md`에 정의된 상태 기계(State Machine)에 따라 관리
- Shopee Main Service와의 ROS2 인터페이스를 통한 명령 수신 및 상태 보고
- Packee Arm, Vision 컴포넌트에 대한 작업 지시 및 결과 수신

## 2. 노드 아키텍처 (Node Architecture)

- **노드 이름**: `packee_main_controller`
- **실행 파일**: `shopee_packee_main/main_controller.py` (가칭)
- **실행 방식**: `ros2 run shopee_packee_main main_controller`

## 3. 상태 관리 (State Management)

`StateDiagram_Pickee.md`에 정의된 상태를 기반으로 한 상태 기계를 구현합니다. 각 상태는 클래스로 정의되며, 상태 진입(on_enter), 실행(execute), 이탈(on_exit) 로직을 포함합니다.

| 상태 (State) | 영문명 | 주요 역할 | 다음 상태 전이 조건 |
|---|---|---|---|
| 초기화중 | `INITIALIZING` | 노드 초기화, 파라미터 로드, 각 컴포넌트 연결 확인 | 초기화 완료 |
| 작업대기중 | `STANDBY` | 장바구니 도착 및 포장 요청 대기중 | `/packee/packing/start` 서비스 수신 |
| 장바구니확인중 | `CHECKING_CART` | 장바구니 유무 확인 및 자세 변경 중 | `/packee/vision/check_cart_presence` 서비스 수신 및 `/packee/arm/pose_status` 토픽 전송 |
| 상품인식중 | `DETECTING_PRODUCTS` | Vision을 통해 장바구니 내 상품 위치 인식 중 | `/packee/vision/detect_products_in_cart` 서비스 수신 |
| 작업계획중 | `PLANNING_TASK` | Task Allocation, Scheduling, Collision Avoidance 계획 중 | `/packee/arm/pick_product` 서비스 수신 |
| 상품담기중 | `PACKING_PRODUCTS` | 양팔을 이용하여 상품을 픽업하고 포장 박스에 담는 중 (Pick & Place) | `packee/arm/place_product` 서비스 수신 |

## 4. 인터페이스 상세 (Interface Specification)

### 4.1. 외부 인터페이스 (vs. Main Service)

#### Publishers
- `/packee/packing_complete` (`PackeePackingComplete.msg`): 포장 완료 시 발행.
- `/packee/robot_status` (`PackeeRobotStatus.msg`): 현재 로봇의 상태, 위치, 배터리 등을 주기적으로 발행.
- `/packee/availability_result` (`PackeeTaskChecking.msg`): 패키 작업 가능 여부 확인 후 가능 여부와 함께 발행.

#### Service Clients
- `/packee/packing/check_availability` (`MainGetTaskChecking.srv`): (필요시) 작업 가능 여부를 Main Service에 질의.
- `/packee/packing/start` (`MainGetPackingStart.srv`): 포장 시작 명령을 받고 시작 했다고 알림.

### 4.2. 내부 인터페이스 (vs. Arm, Vision)

#### Service Clients.
- **vs. Arm**
  - `/packee/arm/move_to_pose` (`PackeeArmMoveToPose.srv`): 'checking_cart', 'standby' 등 특정 자세로 변경을 명령.
  - `/packee/arm/pick_product` (`PackeeArmPickProduct.srv`): '좌측 팔' 또는 '우측 팔'에 특정 위치의 상품을 피킹하도록 명령.
  - `/packee/arm/place_product` (`PackeeArmPlaceProduct.srv`): 좌측 팔' 또는 '우측 팔'에피킹한 상품을 장바구니에 담도록 명령.
- **vs. Vision**
  - `/packee/vision/check_cart_presence` (`PackeeVisionDetectCart.srv`): 장바구니 유무를 확인.
  - `/packee/vision/detect_products_in_cart` (`PackeeVisionSetProduct.srv`): 장바구니 내 상품의 위치를 확인.
  - `/packee/vision/verify_packing_complete` (`PackeeVisionPackingComplete.srv`): 포장 완료 여부를 확인.

#### Subscribers
- **from Arm**
  - `/packee/arm/pose_status` (`PackeeArmPoseChangeStatus.msg`): 'in_progress', 'completed', 'failed' 자세 변경 상태 확인
  - `/packee/arm/pick_status` (`PackeeArmPickupStatus.msg`): 좌측 팔 또는 우측 팔에 'in_progress', 'completed', , 'failed', 'planning', 'approaching', 'grasping', 'lifting', 'done' 픽업 상태를 확인
  - `/packee/arm/place_status` (`PackeeArmStackingStatus.msg`): 좌측 팔 또는 우측 팔에 'in_progress', 'completed', , 'failed', 'planning', 'approaching', 'grasping', 'lifting', 'done'  담기 상태를 확인.

## 5. 주요 기능 로직 (Key Logic)

### 5.1. 원격 쇼핑 시나리오 워크플로우
1.  **[STANDBY]** 상태에서 `/packee/packing/start` 서비스 호출 대기.
2.   서비스 호출 시, 상태를 **[CHECKING_CART]** 로 변경.
3.  `/packee/arm/move_to_pose` 토픽으로 장바구니 확인 자세 변경 명령.
4.  자세 변경 완료 시 `/packee/vision/check_cart_presence` 서비스로 장바구니 유무 확인.
5.  장바구니 유무 확인 시 상태를 **[DETECTING_PRODUCTS]**  로 변경..
6.  `/packee/vision/detect_products_in_cart` 서비스로 장바구니 내 상품 위치를 확인.
7.  장바구니 내 상품 위치 확인 시 상태를 **[PLANNING_TASK]**  로 변경..
8.  작업 분배, 스케줄링, 충돌 회피를 계획
9.  작업 계획 완료 시, 상태를 **[PACKING_PRODUCTS]** 로 변경.
10. `/packee/arm/pick_product`  서비스로 작업 계획을 기반 상품 픽업을 명령.
11. `/packee/arm/place_product` 픽업한 상품을 박스에 적재.
12. `/packee/vision/verify_packing_complete` 서비스로 포장 완료 여부 확인.
13. 포장 완료 시 `/packee/packing_complete` 서비스로 포장 완료 알림.
14. `/packee/arm/move_to_pose` 토픽으로 대기 자세 변경 명령.
15. 자세 변경 완료 시 **[STANDBY]** 로 변경.

## 6. 파라미터 (ROS2 Parameters)

- `robot_id` (int): 로봇의 고유 ID.
- `battery_threshold_available` (float): 작업 가능으로 판단하는 최소 배터리 잔량 (e.g., 30.0).
- `battery_threshold_unavailable` (float): 즉시 충전이 필요한 배터리 잔량 (e.g., 10.0).
- `main_service_timeout` (float): Main Service 응답 대기 시간.
- `component_service_timeout` (float): 내부 컴포넌트 응답 대기 시간.
