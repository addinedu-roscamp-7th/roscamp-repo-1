
# Shopee Pickee Main Controller 상세 설계

## 1. 개요 (Overview)

`pickee_main_controller`는 Pickee 로봇의 두뇌 역할을 하는 중앙 제어 노드입니다. 전체 시스템 아키텍처에서 Shopee Main Service로부터 태스크를 할당받아 Pickee의 하위 컴포넌트(Mobile, Arm, Vision)를 총괄 제어하여 주어진 임무를 수행하고, 로봇의 상태를 지속적으로 Main Service에 보고합니다.

**주요 책임:**
- Pickee 로봇의 상태를 `StateDiagram_Pickee.md`에 정의된 상태 기계(State Machine)에 따라 관리
- Shopee Main Service와의 ROS2 인터페이스를 통한 명령 수신 및 상태 보고
- Pickee Mobile, Arm, Vision 컴포넌트에 대한 작업 지시 및 결과 수신
- 원격 쇼핑, 재고 보충 등 주요 시나리오 워크플로우 실행
- LLM Service와의 연동을 통한 음성 명령 처리 (재고 보충 시나리오)

## 2. 노드 아키텍처 (Node Architecture)

- **노드 이름**: `pickee_main_controller`
- **실행 파일**: `shopee_pickee_main/main_controller.py` (가칭)
- **실행 방식**: `ros2 run shopee_pickee_main main_controller`

## 3. 상태 관리 (State Management)

`StateDiagram_Pickee.md`에 정의된 상태를 기반으로 한 상태 기계를 구현합니다. 각 상태는 클래스로 정의되며, 상태 진입(on_enter), 실행(execute), 이탈(on_exit) 로직을 포함합니다.

| 상태 (State) | 영문명 | 주요 역할 | 다음 상태 전이 조건 |
|---|---|---|---|
| 초기화중 | `INITIALIZING` | 노드 초기화, 파라미터 로드, 각 컴포넌트 연결 확인 | 초기화 완료 |
| 충전중 (작업불가) | `CHARGING_UNAVAILABLE` | 배터리 충전 (임계값 이하) | 배터리 임계값(e.g., 30%) 이상 충전 |
| 충전중 (작업가능) | `CHARGING_AVAILABLE` | 작업 대기 | `/start_task` 서비스 수신 |
| 상품위치이동중 | `MOVING_TO_SHELF` | 지정된 매대로 이동 | `/pickee/mobile/arrival` 토픽 수신 |
| 상품인식중 | `DETECTING_PRODUCT` | 매대 상품 인식 | `/pickee/vision/detection_result` 토픽 수신 |
| 상품선택대기중 | `WAITING_SELECTION` | 사용자(고객)의 상품 선택 대기 | `/pickee/product/process_selection` 서비스 수신 |
| 상품담기중 | `PICKING_PRODUCT` | 로봇팔을 이용해 상품을 피킹하여 장바구니에 담기 | `/pickee/arm/place_status` (completed) 토픽 수신 |
| 포장대이동중 | `MOVING_TO_PACKING` | 포장대로 이동 | `/pickee/mobile/arrival` 토픽 수신 |
| 장바구니전달대기중 | `WAITING_HANDOVER` | Packee 로봇에게 장바구니 전달 대기 | `/pickee/cart_handover_complete` 토픽 수신 |
| 대기장소이동중 | `MOVING_TO_STANDBY` | 지정된 대기 장소로 복귀 | `/pickee/mobile/arrival` 토픽 수신 |
| 직원등록중 | `REGISTERING_STAFF` | 재고 보충 직원 등록 | `/pickee/vision/register_staff_result` (success) 토픽 수신 |
| 직원추종중 | `FOLLOWING_STAFF` | 직원을 따라 이동 | 음성 명령(e.g., "창고로 가자") 인식 |
| 창고이동중 | `MOVING_TO_WAREHOUSE` | 재고 보충을 위해 창고로 이동 | `/pickee/mobile/arrival` 토픽 수신 |
| 적재대기중 | `WAITING_LOADING` | 직원이 상품을 장바구니에 싣는 것을 대기 | 음성 명령(e.g., "매대로 가자") 인식 |
| 매대이동중 | `MOVING_TO_SHELF` (재사용) | 상품을 보충할 매대로 이동 | `/pickee/mobile/arrival` 토픽 수신 |
| 하차대기중 | `WAITING_UNLOADING` | 직원이 상품을 매대에 진열하는 것을 대기 | 음성 명령(e.g., "다 끝났어") 인식 |

## 4. 인터페이스 상세 (Interface Specification)

### 4.1. 외부 인터페이스 (vs. Main Service)

#### Service Servers
- `/pickee/workflow/start_task` (`PickeeWorkflowStartTask.srv`): 원격 쇼핑 작업을 시작. 상품 목록 및 위치 정보를 받음.
- `/pickee/workflow/move_to_section` (`PickeeWorkflowMoveToSection.srv`): 특정 섹션으로 이동을 명령 받음.
- `/pickee/product/detect` (`PickeeProductDetect.srv`): 특정 상품을 인식하도록 명령 받음.
- `/pickee/product/process_selection` (`PickeeProductProcessSelection.srv`): 사용자가 선택한 상품을 집도록 명령 받음.
- `/pickee/workflow/end_shopping` (`PickeeWorkflowEndShopping.srv`): 쇼핑을 종료하고 포장대로 이동하도록 명령 받음.
- 기타 `Main_vs_Pic_Main.md`에 명시된 모든 서비스

#### Publishers
- `/pickee/robot_status` (`PickeeRobotStatus.msg`): 현재 로봇의 상태, 위치, 배터리 등을 주기적으로 발행.
- `/pickee/arrival_notice` (`PickeeArrival.msg`): 목적지 도착 시 발행.
- `/pickee/product_detected` (`PickeeProductDetection.msg`): 상품 인식 완료 시 BBox 정보와 함께 발행.
- 기타 `Main_vs_Pic_Main.md`에 명시된 모든 토픽

#### Service Clients
- `/main/get_product_location` (`MainGetProductLocation.srv`): (필요시) 특정 상품의 위치 정보를 Main Service에 질의.

### 4.2. 내부 인터페이스 (vs. Mobile, Arm, Vision)

#### Service Clients
- **vs. Mobile**
  - `/pickee/mobile/move_to_location` (`PickeeMobileMoveToLocation.srv`): Mobile 컴포넌트에 목적지 이동을 명령.
- **vs. Arm**
  - `/pickee/arm/move_to_pose` (`PickeeArmMoveToPose.srv`): 'shelf_view', 'standby' 등 특정 자세로 변경을 명령.
  - `/pickee/arm/pick_product` (`PickeeArmPickProduct.srv`): 특정 위치의 상품을 피킹하도록 명령.
  - `/pickee/arm/place_product` (`PickeeArmPlaceProduct.srv`): 피킹한 상품을 장바구니에 담도록 명령.
- **vs. Vision**
  - `/pickee/vision/detect_products` (`PickeeVisionDetectProducts.srv`): 매대의 상품을 인식하도록 명령.
  - `/pickee/vision/set_mode` (`PickeeVisionSetMode.srv`): Vision 컴포넌트의 동작 모드(e.g., `navigation`, `detect_products`)를 설정.
  - `/pickee/vision/track_staff` (`PickeeVisionTrackStaff.srv`): 직원 추종을 시작/중지하도록 명령.

#### Subscribers
- **from Mobile**
  - `/pickee/mobile/arrival` (`PickeeMobileArrival.msg`): 목적지 도착 알림을 구독하여 상태 전이에 사용.
  - `/pickee/mobile/pose` (`PickeeMobilePose.msg`): 현재 위치 및 상태를 수신하여 `/pickee/robot_status` 토픽에 반영.
- **from Arm**
  - `/pickee/arm/pick_status` (`PickeeArmTaskStatus.msg`): 상품 피킹 작업의 진행 상태(성공/실패/진행중)를 구독.
  - `/pickee/arm/place_status` (`PickeeArmTaskStatus.msg`): 상품 놓기 작업의 진행 상태를 구독.
- **from Vision**
  - `/pickee/vision/detection_result` (`PickeeVisionDetection.msg`): 상품 인식 결과를 구독.
  - `/pickee/vision/obstacle_detected` (`PickeeVisionObstacles.msg`): 장애물 감지 정보를 구독하여 Mobile 제어에 반영.

## 5. 주요 기능 로직 (Key Logic)

### 5.1. 원격 쇼핑 시나리오 워크플로우
1.  **[CHARGING_AVAILABLE]** 상태에서 `/pickee/workflow/start_task` 서비스 호출 대기.
2.  서비스 호출 시, 요청된 상품 목록(`product_list`)을 내부 변수에 저장하고 첫 번째 목적지로 이동 시작. 상태를 **[MOVING_TO_SHELF]** 로 변경.
3.  `/pickee/mobile/move_to_location` 서비스 호출.
4.  `/pickee/mobile/arrival` 토픽 수신 시, 상태를 **[DETECTING_PRODUCT]** 로 변경.
5.  `/pickee/arm/move_to_pose` (shelf_view) 호출 및 `/pickee/vision/set_mode` (detect_products) 호출.
6.  `/pickee/vision/detect_products` 서비스 호출.
7.  `/pickee/vision/detection_result` 토픽 수신 후, 인식 결과를 `/pickee/product_detected` 토픽으로 Main Service에 전송.
8.  사용자 선택 방식에 따라 분기:
    - **자동 선택**: 가장 신뢰도 높은 상품을 선택하고 상태를 **[PICKING_PRODUCT]** 로 변경.
    - **수동 선택**: 상태를 **[WAITING_SELECTION]** 으로 변경하고 `/pickee/product/process_selection` 서비스 호출 대기.
9.  **[PICKING_PRODUCT]** 상태에서 `/pickee/arm/pick_product` 및 `/pickee/arm/place_product` 서비스를 순차적으로 호출.
10. `/pickee/arm/place_status` (completed) 토픽 수신 후, `/pickee/product/selection_result` 토픽으로 Main Service에 결과 보고.
11. 남은 상품이 있으면 다음 목적지로 이동 (2번부터 반복). 모든 상품을 담았으면 `/pickee/workflow/end_shopping` 서비스 호출 대기.
12. `/pickee/workflow/end_shopping` 수신 시, 상태를 **[MOVING_TO_PACKING]** 으로 변경하고 포장대로 이동.
13. 포장대 도착 후 **[WAITING_HANDOVER]** 상태에서 장바구니 전달 로직 수행.
14. 완료 후 **[MOVING_TO_STANDBY]** 상태로 변경하고 대기 장소로 복귀.

## 6. 파라미터 (ROS2 Parameters)

- `robot_id` (int): 로봇의 고유 ID.
- `battery_threshold_available` (float): 작업 가능으로 판단하는 최소 배터리 잔량 (e.g., 30.0).
- `battery_threshold_unavailable` (float): 즉시 충전이 필요한 배터리 잔량 (e.g., 10.0).
- `default_linear_speed` (float): 기본 주행 선속도.
- `default_angular_speed` (float): 기본 주행 각속도.
- `main_service_timeout` (float): Main Service 응답 대기 시간.
- `component_service_timeout` (float): 내부 컴포넌트 응답 대기 시간.
