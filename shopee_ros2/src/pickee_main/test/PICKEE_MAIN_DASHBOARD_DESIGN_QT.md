# 테스트를 위한 대쉬보드 제작 설명
## 실행시킨 터미널 창 총 4개
  - ros2 run pickee_main main_controller
  - ros2 run pickee_main mock_mobile_node
  - ros2 run pickee_main mock_arm_node
  - ros2 run pickee_main mock_vision_node 

## 테스터 상태
  - 테스터는 위 4개의 명령어를 사용하여 pickee_main main_controller를 중심으로 통신 테스트중임
  - 테스트할 서비스와 토픽 목록은 아래 참고
  - 테스터가 느끼는 불편함은 아래의 서비스, 토픽 목록을 테스트하는데, 터미널 로그가 너무 길어서 보기 힘듬
  - GUI를 통해 어떤 노드에 어떤 로그가 생겼는지 확인하고 싶음.

## 서비스, 토픽 목록
### 서비스 리스트 : ros2 service list -t
```bash
client /main/get_location_pose [shopee_interfaces/srv/MainGetLocationPose]
client /main/get_product_location [shopee_interfaces/srv/MainGetProductLocation]
client /main/get_section_pose [shopee_interfaces/srv/MainGetSectionPose]
client /main/get_warehouse_pose [shopee_interfaces/srv/MainGetWarehousePose]
client /pickee/arm/move_to_pose [shopee_interfaces/srv/PickeeArmMoveToPose]
client /pickee/arm/pick_product [shopee_interfaces/srv/PickeeArmPickProduct]
client /pickee/arm/place_product [shopee_interfaces/srv/PickeeArmPlaceProduct]
client /pickee/mobile/move_to_location [shopee_interfaces/srv/PickeeMobileMoveToLocation]
client /pickee/mobile/update_global_path [shopee_interfaces/srv/PickeeMobileUpdateGlobalPath]
service /pickee/product/detect [shopee_interfaces/srv/PickeeProductDetect]
service /pickee/product/process_selection [shopee_interfaces/srv/PickeeProductProcessSelection]
service /pickee/tts_request [shopee_interfaces/srv/PickeeTtsRequest]
service /pickee/video_stream/start [shopee_interfaces/srv/PickeeMainVideoStreamStart]
service /pickee/video_stream/stop [shopee_interfaces/srv/PickeeMainVideoStreamStop]
client /pickee/vision/check_cart_presence [shopee_interfaces/srv/PickeeVisionCheckCartPresence]
client /pickee/vision/check_product_in_cart [shopee_interfaces/srv/PickeeVisionCheckProductInCart]
client /pickee/vision/detect_products [shopee_interfaces/srv/PickeeVisionDetectProducts]
client /pickee/vision/register_staff [shopee_interfaces/srv/PickeeVisionRegisterStaff]
client /pickee/vision/set_mode [shopee_interfaces/srv/PickeeVisionSetMode]
client /pickee/vision/track_staff [shopee_interfaces/srv/PickeeVisionTrackStaff]
client /pickee/vision/video_stream_start [shopee_interfaces/srv/PickeeVisionVideoStreamStart]
client /pickee/vision/video_stream_stop [shopee_interfaces/srv/PickeeVisionVideoStreamStop]
service /pickee/workflow/end_shopping [shopee_interfaces/srv/PickeeWorkflowEndShopping]
service /pickee/workflow/move_to_packaging [shopee_interfaces/srv/PickeeWorkflowMoveToPackaging]
service /pickee/workflow/move_to_section [shopee_interfaces/srv/PickeeWorkflowMoveToSection]
service /pickee/workflow/return_to_base [shopee_interfaces/srv/PickeeWorkflowReturnToBase]
service /pickee/workflow/return_to_staff [shopee_interfaces/srv/PickeeWorkflowReturnToStaff]
service /pickee/workflow/start_task [shopee_interfaces/srv/PickeeWorkflowStartTask]
```

### 토픽 리스트 : ros2 topic list -t
```bash
sub /pickee/arm/pick_status [shopee_interfaces/msg/PickeeArmTaskStatus]
sub /pickee/arm/place_status [shopee_interfaces/msg/PickeeArmTaskStatus]
sub /pickee/arm/pose_status [shopee_interfaces/msg/ArmPoseStatus]
pub /pickee/arrival_notice [shopee_interfaces/msg/PickeeArrival]
pub /pickee/cart_handover_complete [shopee_interfaces/msg/PickeeCartHandover]
sub /pickee/mobile/arrival [shopee_interfaces/msg/PickeeMobileArrival]
sub /pickee/mobile/pose [shopee_interfaces/msg/PickeeMobilePose]
pub /pickee/mobile/speed_control [shopee_interfaces/msg/PickeeMobileSpeedControl]
pub /pickee/moving_status [shopee_interfaces/msg/PickeeMoveStatus]
pub /pickee/product/loaded [shopee_interfaces/msg/PickeeProductLoaded]
pub /pickee/product/selection_result [shopee_interfaces/msg/PickeeProductSelection]
pub /pickee/product_detected [shopee_interfaces/msg/PickeeProductDetection]
pub /pickee/robot_status [shopee_interfaces/msg/PickeeRobotStatus]
sub /pickee/vision/cart_check_result [shopee_interfaces/msg/PickeeVisionCartCheck]
sub /pickee/vision/detection_result [shopee_interfaces/msg/PickeeVisionDetection]
sub /pickee/vision/obstacle_detected [shopee_interfaces/msg/PickeeVisionObstacles]
sub /pickee/vision/register_staff_result [shopee_interfaces/msg/PickeeVisionStaffRegister]
sub /pickee/vision/staff_location [shopee_interfaces/msg/PickeeVisionStaffLocation]
```

---

# Pickee Main 대시보드 상세 설계 (수정)

## 1. 개요

### 1.1. 목적
`pickee_main`, `shopee_main_service` 및 관련 mock 노드들(`mock_mobile_node`, `mock_arm_node`, `mock_vision_node`) 간의 ROS2 통신(서비스, 토픽)을 시각적으로 모니터링하고, 각 노드에서 발생하는 로그를 실시간으로 확인하여 테스트 및 디버깅의 효율성을 높이는 것을 목적으로 한다. 기존의 터미널 기반 모니터링 방식의 불편함을 해소하고 사용자 친화적인 GUI를 제공한다.

### 1.2. 주요 기능
-   실행 중인 ROS2 노드 목록 및 상태 표시
-   노드별/전체 로그 메시지 실시간 출력 및 필터링 (로그 레벨 기준)
-   활성화된 서비스 및 토픽 목록과 상세 정보(타입) 표시
-   특정 토픽의 메시지를 실시간으로 구독하고 내용 확인
-   GUI에서 직접 서비스 호출 및 파라미터 입력 기능 (선택 사항)

## 2. UI/UX 설계

### 2.1. 레이아웃
PySide6를 사용하여 메인 윈도우를 다음과 같이 4개의 주요 패널로 구성한다.

-   **A. 노드 목록 패널 (Node List Panel):**
    -   위치: 좌측 상단
    -   위젯: `QListWidget`
    -   기능: 현재 실행 중인 노드(`main_controller`, `robot_coordinator`, `mock_mobile_node` 등)의 목록을 표시한다. 주기적으로 노드 목록을 갱신하여 최신 상태를 유지한다.

-   **B. 서비스/토픽 패널 (Service/Topic Panel):**
    -   위치: 좌측 하단
    -   위젯: `QTreeWidget`
    -   기능: 시스템의 모든 서비스와 토픽 목록을 트리 형태로 보여준다. 최상위 레벨은 "Services"와 "Topics"로 구분한다.
        -   **Services:** `pickee_main`이 호출하는 서비스(예: `/pickee/workflow/start_task`)와 `shopee_main_service`가 제공하는 서비스(예: `/main/get_available_robots`)를 모두 표시한다.
        -   **Topics:** `pickee_main`이 발행하는 토픽(예: `/pickee/robot_status`)과 `shopee_main_service`가 구독하는 토픽(예: `/pickee/moving_status`)을 모두 표시한다.

-   **C. 로그 출력 패널 (Log Output Panel):**
    -   위치: 우측
    -   위젯: `QTextEdit` 또는 `QTableView`
    -   기능: ROS2 시스템의 `/rosout` 토픽을 구독하여 모든 로그를 실시간으로 표시한다. 로그 레벨(INFO, WARN, ERROR, DEBUG)에 따라 텍스트 색상을 다르게 하여 가독성을 높인다. 노드 목록 패널에서 특정 노드를 선택하면 해당 노드의 로그만 필터링하여 보여준다.

-   **D. 상세 정보 패널 (Detailed Info Panel):**
    -   위치: (초기에는 숨김) 서비스/토픽 패널에서 항목 더블클릭 시 별도 팝업 또는 하단에 표시
    -   위젯: `QDialog` 또는 `QTextEdit`
    -   기능: 선택된 토픽의 메시지 내용을 실시간으로 표시하거나, 선택된 서비스를 호출할 수 있는 입력 폼을 제공한다.

### 2.2. 사용자 인터랙션
-   **노드 선택:** 노드 목록 패널에서 특정 노드를 클릭하면, 로그 출력 패널의 내용이 해당 노드의 로그로 필터링된다.
-   **로그 필터링:** 로그 출력 패널 상단에 체크박스(INFO, WARN, ERROR)를 두어 원하는 레벨의 로그만 선택적으로 볼 수 있다.
-   **정보 조회:** 서비스/토픽 패널에서 항목을 더블클릭하면 상세 정보 패널이 나타나 해당 항목의 상세 내용을 보여준다.
-   **검색:** 각 패널 상단에 `QLineEdit`을 배치하여 노드, 서비스, 토픽, 로그 내용을 검색할 수 있는 기능을 제공한다.

## 3. 아키텍처 및 데이터 흐름

### 3.1. 기술 스택
-   **언어:** Python
-   **GUI 프레임워크:** PySide6
-   **ROS2 연동:** `rclpy` (ROS2 Python Client Library)

### 3.2. 아키텍처
대시보드 애플리케이션 자체가 하나의 ROS2 노드(`pickee_dashboard_node`)로 동작한다. 이 노드는 GUI 스레드와 ROS2 통신 스레드로 구성된다.

-   **GUI 스레드 (Main Thread):**
    -   PySide6 애플리케이션의 메인 루프를 실행하고 UI 렌더링 및 사용자 입력을 처리한다.
    -   ROS2 스레드에서 전달받은 데이터를 UI 위젯에 업데이트한다.

-   **ROS2 통신 스레드 (Background Thread):**
    -   `rclpy.spin()`을 실행하여 ROS2 메시지 펌프를 계속 돌린다. 이를 통해 GUI가 블로킹되는 것을 방지한다.
    -   `/rosout` 토픽 구독자, 주기적인 노드/서비스/토픽 목록 폴러(Poller)를 관리한다.
    -   수신한 데이터나 시스템 변경 사항을 Qt의 시그널(Signal)을 통해 GUI 스레드로 안전하게 전달한다.

### 3.3. 데이터 흐름
1.  **초기화:** 대시보드 실행 시, ROS2 통신 스레드는 `rclpy.init()`을 호출하고 `pickee_dashboard_node`를 생성한다.
2.  **로그 수신:** `/rosout` 토픽 구독자가 새로운 로그 메시지(`Log` 타입)를 수신하면, 콜백 함수가 호출된다. 콜백 함수는 메시지에서 노드 이름, 로그 레벨, 내용 등을 파싱하여 시그널을 발생시킨다.
3.  **UI 업데이트:** GUI 스레드의 슬롯(Slot) 함수가 시그널을 받아 로그 출력 패널에 해당 로그를 추가한다.
4.  **시스템 정보 갱신:** ROS2 통신 스레드는 `QTimer` 등을 이용해 주기적으로(`ros2node.get_node_names()`, `ros2service.get_service_names_and_types()`, `ros2topic.get_topic_names_and_types()`)를 호출하여 시스템 정보를 가져온다. 이 정보에는 `pickee_main`과 `shopee_main_service`가 호출하거나 제공하는 모든 서비스와 토픽이 포함된다.
5.  **목록 갱신:** 이전 정보와 비교하여 변경된 내용이 있으면, 시그널을 통해 GUI 스레드에 알리고 노드 및 서비스/토픽 패널을 업데이트한다.

## 4. 구현 계획

### 1단계: 기본 UI 및 레이아웃 구현
-   `QMainWindow` 기반의 메인 윈도우 생성
-   `QSplitter`와 `QDockWidget`을 사용하여 4개의 패널 레이아웃 구성
-   각 패널에 필요한 위젯(`QListWidget`, `QTreeWidget`, `QTextEdit`) 배치

### 2단계: ROS2 연동 및 로그 표시
-   `rclpy`를 사용하는 별도의 스레드 클래스 작성
-   `/rosout` 구독 로직 및 수신 데이터를 GUI로 전달하기 위한 시그널/슬롯 구현
-   GUI 스레드에서 로그 메시지를 받아 `QTextEdit`에 표시하는 기능 구현

### 3단계: 노드, 서비스, 토픽 목록 표시
-   주기적으로 ROS2 시스템 정보를 폴링하는 로직 구현
-   폴링한 데이터를 파싱하여 `QListWidget`과 `QTreeWidget`에 표시하는 기능 구현
-   노드 선택 시 로그 필터링 기능 구현

### 4단계: 상세 기능 고도화
-   로그 레벨별 필터링 체크박스 기능 추가
-   각 패널에 검색 기능 추가
-   토픽 메시지 실시간 조회 및 서비스 호출을 위한 다이얼로그 창 구현