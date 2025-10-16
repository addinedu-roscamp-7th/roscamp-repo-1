Pic Main = Pickee Main Controller

Pic Mobile = Pickee Mobile Service

### `/pickee/mobile/pose`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeMobilePose.msg`

### `/pickee/mobile/arrival`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeMobileArrival.msg`

### `/pickee/mobile/speed_control`
> **ROS2 Interface:** `shopee_interfaces/msg/PickeeMobileSpeedControl.msg`

### `/pickee/mobile/move_to_location`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeMobileMoveToLocation.srv`

### `/pickee/mobile/update_global_path`
> **ROS2 Interface:** `shopee_interfaces/srv/PickeeMobileUpdateGlobalPath.srv`

**구조체 매핑**
- `Pose2D` → `shopee_interfaces/msg/Pose2D`
- `Obstacle` → `shopee_interfaces/msg/Obstacle`




## Topic 인터페이스

| 구분 | 메시지명 | 토픽 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **위치 업데이트** | `/pickee/mobile/pose` | Topic | Pic Mobile | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`Pose2D current_pose`<br>`float32 linear_velocity`<br>`float32 angular_velocity`<br>`float32 battery_level`<br>`string status` | `robot_id: 1`<br>`order_id: 3`<br>`current_pose: {x: 5.3, y: 2.1, theta: 0.5}`<br>`linear_velocity: 0.8`<br>`angular_velocity: 0.0`<br>`battery_level: 75.5`<br>`status: "moving"` |
| **도착 알림** | `/pickee/mobile/arrival` | Topic | Pic Mobile | Pic Main | `int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`Pose2D final_pose`<br>`float32 position_error`<br>`float32 travel_time`<br>`string message` | `robot_id: 1`<br>`order_id: 3`<br>`location_id: 3`<br>`final_pose: {x: 10.52, y: 5.18, theta: 1.56}`<br>`position_error: 0.03`<br>`travel_time: 43.5`<br>`message: "Arrived at LOC_A1"` |
| **속도 제어** | `/pickee/mobile/speed_control` | Topic | Pic Main | Pic Mobile | `int32 robot_id`<br>`int32 order_id`<br>`string speed_mode`<br>`float32 target_speed`<br>`Obstacle[] obstacles`<br>`string reason`<br><br>**speed_mode**<br>`"normal"`, `"decelerate"`, `"stop"` | `robot_id: 1`<br>`order_id: 44`<br>`speed_mode: "decelerate"`<br>`target_speed: 0.3`<br>`obstacles:`<br>`- obstacle_type: "person"`<br>`  distance: 1.5`<br>`  velocity: 0.8`<br>`reason: "dynamic_obstacle_near"`<br><br>`speed_mode: "stop"`<br>`target_speed: 0.0`<br>`reason: "collision_risk"` |

## Service 인터페이스

| 구분 | 서비스명 | 서비스 | From | To | 메시지 구조 | 예시 |
|---|---|---|---|---|---|---|
| **목적지 이동 명령** | `/pickee/mobile/move_to_location` | Service | Pic Main | Pic Mobile | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`Pose2D target_pose`<br>`Pose2D[] global_path`<br>`string navigation_mode`<br><br>**Response**<br>`bool success`<br>`string message` | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`location_id: 3`<br>`target_pose: {x: 10.5, y: 5.2, theta: 1.57}`<br>`global_path:`<br>`- {x: 0.0, y: 0.0, theta: 0.0}`<br>`- {x: 2.5, y: 1.2, theta: 0.5}`<br>`- {x: 5.0, y: 2.5, theta: 0.8}`<br>`- {x: 7.5, y: 4.0, theta: 1.2}`<br>`- {x: 10.5, y: 5.2, theta: 1.57}`<br>`navigation_mode: "normal"`<br><br>**Response**<br>`success: true`<br>`message: "Navigation started"` |
| **Global Path 업데이트** | `/pickee/mobile/update_global_path` | Service | Pic Main | Pic Mobile | **Request**<br>`int32 robot_id`<br>`int32 order_id`<br>`int32 location_id`<br>`Pose2D[] global_path`<br><br>**Response**<br>`bool success`<br>`string message`<br><br>*Main이 A* 알고리즘으로 생성한 경로* | **Request**<br>`robot_id: 1`<br>`order_id: 3`<br>`location_id: 3`<br>`global_path:`<br>`- {x: 5.0, y: 2.5, theta: 0.8}`<br>`- {x: 4.0, y: 3.5, theta: 1.2}`<br>`- {x: 6.0, y: 4.0, theta: 0.5}`<br>`- {x: 10.5, y: 5.2, theta: 1.57}`<br><br>**Response**<br>`success: true`<br>`message: "Global path updated"` |
