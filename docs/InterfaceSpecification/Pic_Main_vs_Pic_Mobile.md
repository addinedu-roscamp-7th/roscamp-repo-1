Pic Main = Pickee Main Controller

Pic Mobile = Pickee Mobile Service

## ROS 인터페이스 매핑

| 분류 | 토픽/서비스 | ROS 타입 |
|---|---|---|
| Topic | `/pickee/mobile/pose` | `shopee_interfaces/msg/PickeeMobilePose` |
| Topic | `/pickee/mobile/arrival` | `shopee_interfaces/msg/PickeeMobileArrival` |
| Topic | `/pickee/mobile/speed_control` | `shopee_interfaces/msg/PickeeMobileSpeedControl` |
| Service | `/pickee/mobile/move_to_location` | `shopee_interfaces/srv/PickeeMobileMoveToLocation` |
| Service | `/pickee/mobile/update_global_path` | `shopee_interfaces/srv/PickeeMobileUpdateGlobalPath` |

**구조체 매핑**
- `Pose2D` → `shopee_interfaces/msg/Pose2D`
- `Obstacle` → `shopee_interfaces/msg/Obstacle`




From

To

Message

예시

Topic











위치 업데이트

/pickee/mobile/pose

Pic Mobile

Pic Main

int32 robot_id
int32 order_id
Pose2D current_pose
float32 linear_velocity
float32 angular_velocity
float32 battery_level
string status

robot_id: 1
order_id: 3
current_pose: {x: 5.3, y: 2.1, theta: 0.5}
linear_velocity: 0.8
angular_velocity: 0.0
battery_level: 75.5
status: "moving"

도착 알림

/pickee/mobile/arrival

Pic Mobile

Pic Main

int32 robot_id
int32 order_id
int32 location_id
Pose2D final_pose
float32 position_error
float32 travel_time
string message

robot_id: 1
order_id: 3
location_id: 3
final_pose: {x: 10.52, y: 5.18, theta: 1.56}
position_error: 0.03
travel_time: 43.5
message: "Arrived at LOC_A1"

속도 제어

/pickee/mobile/speed_control

Pic Main

Pic Mobile

int32 robot_id
int32 order_id
string speed_mode ("normal", "decelerate", "stop")
float32 target_speed
Obstacle[] obstacles
string reason

robot_id: 1
order_id: 44
speed_mode: "decelerate"
target_speed: 0.3
obstacles:
  - obstacle_type: "person"
    distance: 1.5
    velocity: 0.8
reason: "dynamic_obstacle_near"

speed_mode: "stop"
target_speed: 0.0
reason: "collision_risk"

Service











목적지 이동 명령

/pickee/mobile/move_to_location

Pic Main

Pic Mobile

# Request
int32 robot_id
int32 order_id
int32 location_id
Pose2D target_pose
Pose2D[] global_path
string navigation_mode

---
# Response
bool success
string message

Request:
  robot_id: 1
  order_id: 3
  location_id: 3
  target_pose: {x: 10.5, y: 5.2, theta: 1.57}
  global_path:
    - {x: 0.0, y: 0.0, theta: 0.0}
    - {x: 2.5, y: 1.2, theta: 0.5}
    - {x: 5.0, y: 2.5, theta: 0.8}
    - {x: 7.5, y: 4.0, theta: 1.2}
    - {x: 10.5, y: 5.2, theta: 1.57}
  navigation_mode: "normal"

Response:
  success: true
  message: "Navigation started"

Global Path 업데이트

/pickee/mobile/update_global_path

Pic Main

Pic Mobile

# Request
int32 robot_id
int32 order_id
int32 location_id
Pose2D[] global_path  # Main이 A* 알고리즘으로 생성한 경로

---
# Response
bool success
string message

Request:
  robot_id: 1
  order_id: 3
  location_id: 3
  global_path:
    - {x: 5.0, y: 2.5, theta: 0.8}
    - {x: 4.0, y: 3.5, theta: 1.2}
    - {x: 6.0, y: 4.0, theta: 0.5}
    - {x: 10.5, y: 5.2, theta: 1.57}

Response:
  success: true
  message: "Global path updated"
