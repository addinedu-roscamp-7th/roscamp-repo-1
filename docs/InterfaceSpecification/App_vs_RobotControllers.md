Shopee App (Admin Dashboard) = 쇼피 관리자 앱
Pickee Main Controller = 피키 메인 컨트롤러
Packee Main Controller = 패키 메인 컨트롤러

## 통신 개요
- **프로토콜**: ROS 2 Topic (rosbridge WebSocket 구독)
- **전송 방향**: 로봇 컨트롤러 → Shopee App (읽기 전용)
- **목적**: 관리자 대시보드에서 실시간 로봇 상태 / 위치 / 작업 상황을 표시
- **갱신 주기**: 기본 30초 폴링 + 이벤트 발생 시 즉시 Push (토픽 발행 시점 기준)

## Topic 인터페이스

### Pickee 로봇 상태 스트림
- **Topic**: `/pickee/robot_status`
- **ROS 타입**: `shopee_interfaces/msg/PickeeRobotStatus`
- **From**: Pickee Main Controller
- **To**: Shopee App (Admin)
- **필드**
  | 필드 | 타입 | 설명 |
  |---|---|---|
  | `robot_id` | `int32` | 로봇 고유 ID |
  | `state` | `string` | 피키 상태 코드 (예: `PK_S10`) |
  | `battery_level` | `float32` | 배터리 잔량 (%) |
  | `current_order_id` | `int32` | 수행 중인 주문 ID (없으면 0) |
  | `position_x` | `float32` | X 좌표 (m) |
  | `position_y` | `float32` | Y 좌표 (m) |
  | `orientation_z` | `float32` | 요(heading) 방향 (rad) |
- **샘플 JSON (rosbridge)**:
  ```json
  {
    "op": "publish",
    "topic": "/pickee/robot_status",
    "msg": {
      "robot_id": 1,
      "state": "PK_S20",
      "battery_level": 78.5,
      "current_order_id": 12,
      "position_x": 5.3,
      "position_y": 2.1,
      "orientation_z": 0.48
    }
  }
  ```
- **활용 시나리오**: Dashboard 실시간 카드/지도 표시 (`SC_05_1_1`, `SC_05_1_2`, `SC_05_1_4`)

### Packee 로봇 상태 스트림
- **Topic**: `/packee/robot_status`
- **ROS 타입**: `shopee_interfaces/msg/PackeeRobotStatus`
- **From**: Packee Main Controller
- **To**: Shopee App (Admin)
- **필드**
  | 필드 | 타입 | 설명 |
  |---|---|---|
  | `robot_id` | `int32` | 로봇 고유 ID |
  | `state` | `string` | 패키 상태 코드 (예: `PC_S10`) |
  | `current_order_id` | `int32` | 수행 중인 주문 ID |
  | `items_in_cart` | `int32` | 현재 카트 내 아이템 수 |
- **샘플 JSON (rosbridge)**:
  ```json
  {
    "op": "publish",
    "topic": "/packee/robot_status",
    "msg": {
      "robot_id": 2,
      "state": "PC_PACKING",
      "current_order_id": 23,
      "items_in_cart": 5
    }
  }
  ```
- **활용 시나리오**: 관리자 대시보드 로봇 테이블 및 포장 진행률 표시 (`SC_05_1_1`, `SC_05_1_4`)

### Pickee 위치 확인 (선택)
- **Topic**: `/pickee/mobile/pose`
- **ROS 타입**: `shopee_interfaces/msg/PickeeMobilePose`
- **From**: Pickee Mobile Controller
- **To**: Shopee App (Admin)
- **필드**
  | 필드 | 타입 | 설명 |
  |---|---|---|
  | `robot_id` | `int32` | 로봇 ID |
  | `order_id` | `int32` | 연관 주문 ID |
  | `current_pose` | `shopee_interfaces/msg/Pose2D` | x, y, theta |
  | `linear_velocity` | `float32` | 선속도 (m/s) |
  | `angular_velocity` | `float32` | 각속도 (rad/s) |
  | `battery_level` | `float32` | 배터리 잔량 (%) |
  | `status` | `string` | 이동 상태 (`moving`, `waiting` 등) |
- **비고**: 위치 지도 표시가 필요한 경우에만 구독 (`SC_05_1_2`)

### 이벤트 트리거 정리
| 시퀀스 ID | 사용 토픽 | 설명 |
|---|---|---|
| `SC_05_1_1` | `/pickee/robot_status`, `/packee/robot_status` | 대시보드 메인 카드/리스트 갱신 |
| `SC_05_1_2` | `/pickee/robot_status`, `/pickee/mobile/pose` | 위치 지도 갱신 |
| `SC_05_1_4` | `/pickee/robot_status`, `/packee/robot_status` | 로봇 상태 패널 |

## 보안 및 운영 고려사항
- rosbridge WebSocket 연결은 관리자 인증 이후에만 활성화.
- ROS Topic은 읽기 전용(subscribe)으로 제한; 관리자 앱에서 Publish 금지.
- 네트워크 대역폭 확보를 위해 1초 이하 간격의 연속 발행 시 샘플링(Throttle) 적용 권장.
- 필요 시 Shopee Main Service가 필터링/캐싱한 후 앱에 재전송하도록 중간 게이트웨이 구성 가능.
