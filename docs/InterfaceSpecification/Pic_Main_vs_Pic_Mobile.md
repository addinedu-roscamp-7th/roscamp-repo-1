# Pic Main ↔ Pic Mobile

**Pic Main** = Pickee Main Controller

**Pic Mobile** = Pickee Mobile Service

## Topic

### 위치 업데이트
- **Topic:** /pickee/mobile/pose
- **From:** Pic Mobile
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- Pose2D current_pose
- float32 linear_velocity
- float32 angular_velocity
- float32 battery_level
- string status

#### 예시
    robot_id: 1
    order_id: "ORDER_001"
    current_pose: {x: 5.3, y: 2.1, theta: 0.5}
    linear_velocity: 0.8
    angular_velocity: 0.0
    battery_level: 75.5
    status: "moving"

---

### 도착 알림
- **Topic:** /pickee/mobile/arrival
- **From:** Pic Mobile
- **To:** Pic Main

#### Message
- int32 robot_id
- string order_id
- string location_id
- Pose2D final_pose
- float32 position_error
- float32 travel_time
- string message

#### 예시
    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"
    final_pose: {x: 10.52, y: 5.18, theta: 1.56}
    position_error: 0.03
    travel_time: 43.5
    message: "Arrived at LOC_A1"

---

## Service

### 목적지 이동 명령
- **Service:** /pickee/mobile/move_to_location
- **From:** Pic Main
- **To:** Pic Mobile

> **Note:** Service+Topic 말고 Service 단독으로 변경 예정

#### Request
- int32 robot_id
- string order_id
- string location_id
- Pose2D target_pose
- Pose2D[] global_path (Main이 A* 알고리즘으로 생성한 경로)
- string navigation_mode

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"
    target_pose: {x: 10.5, y: 5.2, theta: 1.57}
    global_path:
      - {x: 0.0, y: 0.0, theta: 0.0}
      - {x: 2.5, y: 1.2, theta: 0.5}
      - {x: 5.0, y: 2.5, theta: 0.8}
      - {x: 7.5, y: 4.0, theta: 1.2}
      - {x: 10.5, y: 5.2, theta: 1.57}
    navigation_mode: "normal"

**Response:**

    success: true
    message: "Navigation started"

---

### Global Path 업데이트
- **Service:** /pickee/mobile/update_global_path
- **From:** Pic Main
- **To:** Pic Mobile

> **Note:** Service+Topic 말고 Service 단독으로 변경 예정

#### Request
- int32 robot_id
- string order_id
- string location_id
- Pose2D[] global_path (Main이 A* 알고리즘으로 생성한 경로)

#### Response
- bool success
- string message

#### 예시
**Request:**

    robot_id: 1
    order_id: "ORDER_001"
    location_id: "LOC_A1"
    global_path:
      - {x: 5.0, y: 2.5, theta: 0.8}
      - {x: 4.0, y: 3.5, theta: 1.2}
      - {x: 6.0, y: 4.0, theta: 0.5}
      - {x: 10.5, y: 5.2, theta: 1.57}

**Response:**

    success: true
    message: "Global path updated"

---

### 속도 제어
- **Service:** /pickee/mobile/speed_control
- **From:** Pic Main
- **To:** Pic Mobile

> **Note:** Service+Topic 말고 Service 단독으로 변경 예정

#### Request
- int32 robot_id
- string order_id
- string speed_mode ("normal", "decelerate", "stop")
- float32 target_speed
- Obstacle[] obstacles
- string reason

#### Response
- bool success
- string current_speed
- string message

#### 예시
**감속:**

Request:

    robot_id: 1
    order_id: "ORDER_001"
    speed_mode: "decelerate"
    target_speed: 0.3
    obstacles:
      - obstacle_type: "person"
        distance: 1.5
        velocity: 0.8
    reason: "dynamic_obstacle_near"

Response:

    success: true
    current_speed: "0.3"
    message: "Speed reduced"

**정지:**

Request:

    speed_mode: "stop"
    target_speed: 0.0
    reason: "collision_risk"

---

### 긴급 정지
- **Service:** /pickee/mobile/emergency_stop
- **From:** Pic Main
- **To:** Pic Mobile

#### Request
- int32 robot_id
- string reason

#### Response
- bool success
- Pose2D stopped_pose
- string message

#### 예시
**Request:**

    robot_id: 1
    reason: "emergency"

**Response:**

    success: true
    stopped_pose: {x: 7.3, y: 3.8, theta: 0.9}
    message: "Emergency stop activated"