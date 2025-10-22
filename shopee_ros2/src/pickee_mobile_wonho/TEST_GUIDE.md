# Pickee Mobile Wonho - 테스트 가이드

## Service & Topic List

### 서비스 리스트 : ros2 service list -t
``` bash
service /pickee/mobile/move_to_location [shopee_interfaces/srv/PickeeMobileMoveToLocation]
service /pickee/mobile/update_global_path [shopee_interfaces/srv/PickeeMobileUpdateGlobalPath]
```

### 토픽 리스트 : ros2 topic list -t
``` bash
pub /pickee/mobile/arrival [shopee_interfaces/msg/PickeeMobileArrival]
pub /pickee/mobile/pose [shopee_interfaces/msg/PickeeMobilePose]
sub /pickee/mobile/speed_control [shopee_interfaces/msg/PickeeMobileSpeedControl]
```

## 명령어
``` bash
# service /move_to_location
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{
  robot_id: 1,
  order_id: 123,
  location_id: 5,
  target_pose: {x: 10.5, y: 5.2, theta: 1.57},
}"

# service /update_global_path
ros2 service call /pickee/mobile/update_global_path shopee_interfaces/srv/PickeeMobileUpdateGlobalPath "{
  robot_id: 1,
  order_id: 123,
  location_id: 3,
  target_pose: {x: 10.5, y: 5.2, theta: 1.57},
}"

# pub /pickee/mobile/arrival
# ros2 topic pub --once /pickee/mobile/arrival shopee_interfaces/msg/PickeeMobileArrival "{
#   robot_id: 1,
#   order_id: 1,
#   location_id: 1,
#   final_pose: {
#     x: 1.0,
#     y: 1.0,
#     theta: 1.0
#   },
#   position_error: 1.0,
#   travel_time: 1.0,
#   message: 'complete'
# }"

# pub /pickee/mobile/pose
# status : 'idle', 'moving', 'stopped', 'charging', 'error'
# ros2 topic pub --once /pickee/mobile/pose shopee_interfaces/msg/PickeeMobilePose "{
#   robot_id: 1,
#   order_id: 1,
#   current_pose: {
#     x: 1.0,
#     y: 1.0,
#     theta: 1.0
#   },
#   linear_velocity: 1.0,
#   angular_velocity: 1.0,
#   battery_level: 70.0,
#   status: 'moving'
# }"

# ['cart', 'shelf', 'wall'] : 정적
# ['person', 'small_object'] : 동적
# sub /pickee/mobile/speed_control
ros2 topic pub --once /pickee/mobile/speed_control shopee_interfaces/msg/PickeeMobileSpeedControl "{
  robot_id: 1
  order_id: 44
  speed_mode: "decelerate"
  target_speed: 0.3
  obstacles: {
        obstacle_type: 'person',
        position: {x: 1.0, y: 1.0},
        distance: 1.0,
        velocity: 1.0,
        direction: {vx: 1.0, vy: 1.0},
        bbox: {
            x1: 1,
            y1: 1,
            x2: 1,
            y2: 1
        },
        confidence: 0.98
    }
  reason: "dynamic_obstacle_near"
}"
```