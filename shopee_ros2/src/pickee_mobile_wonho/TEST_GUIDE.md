# Pickee Mobile Wonho - 테스트 가이드


## 실행 명령어
``` bash
# 통신 테스트
## pickee_mobile_wonho 
ros2 run pickee_mobile_wonho pickee_mobile_wonho_node  

## pick main 대쉬보드 : 통신 로그 시간순 모니터링
## pickee_main 노드와 mock_shopee_main 노드 실행 : shopee_main <-> pickee_main <-> pickee_mobile_wonho 간 통신을 위함
ros2 run pickee_main dashboard 
ros2 run pickee_main main_controller 
ros2 run pickee_main mock_shopee_main 

## 간단 실행 명령어 (SC_02_1 매대 이동 테스트용)
ros2 service call /pickee/workflow/move_to_section shopee_interfaces/srv/PickeeWorkflowMoveToSection "{
  robot_id: 1,
  order_id: 123,
  location_id: 1,
  section_id: 1
}"

## 여기까지 실행하면 통신 테스트 됩니다. 주행 시뮬레이션이 없다면 마지막에 에러가 뜰 것입니다.

# 주행 시뮬레이션 or 실제 주행
## Gazebo 실행 / bringup 실행 / rviz 실행
ros2 launch pickee_mobile_wonho_robot gazebo_bringup.launch.xml 
ros2 launch pickee_mobile_wonho nav2_bringup_launch.xml use_sim_time:=True
ros2 launch pickee_mobile_wonho nav2_view.launch.xml 
## 로봇 bringup 실행 / 네비 bringup 실행 / rviz 실행
ros2 launch pickee_mobile_wonho_robot bringup.launch.xml 
ros2 launch pickee_mobile_wonho nav2_bringup_launch.xml
ros2 launch pickee_mobile_wonho nav2_view.launch.xml

## 간단 실행 명령어를 주행 시뮬레이션을 띄워놓은 상태에서 실행해봐도 됩니다. (물론 노드들 다 실행시켜 놓아야 됩니다.)

## 기타 명렁어
### 속도 명령 확인
ros2 topic echo /cmd_vel

### 코스트맵 상태 확인  
ros2 topic echo /local_costmap/costmap_updates

### 플래너 상태 확인
ros2 topic echo /plan
```

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