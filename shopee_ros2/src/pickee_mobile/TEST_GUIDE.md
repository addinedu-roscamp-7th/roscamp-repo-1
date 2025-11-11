# /pickee/mobile/arrival publish
#도착알림
ros2 topic pub -1 /pickee/mobile/arrival shopee_interfaces/msg/PickeeMobileArrival "
{
  robot_id: 1,
  order_id: 1,
  location_id: 3,
  final_pose: {
    x: 1.2,
    y: 3.4,
    theta: 1.57
  },
  position_error: {
    x: 0.01,
    y: -0.02,
    theta: 0.005
  },
  travel_time: 12.5,
  message: \"Arrived at location\"
}"

#bool true
ros2 topic pub -1 /test std_msgs/Bool "{data: true}"


# /pickee/mobile/move_to_location service_client
# 좌매대 우
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "
{
  robot_id: 1,
  order_id: 1,
  location_id: 2,
  target_pose: {
    x: 0.2,
    y: 1.47,
    theta: 3.14
  }
}"

# 중매대 하
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "
{
  robot_id: 1,
  order_id: 1,
  location_id: 1,
  target_pose: {
    x: 0.79,
    y: -0.20,
    theta: 1.57
  }
}"

# 중매대 우
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "
{
  robot_id: 1,
  order_id: 1,
  location_id: 2,
  target_pose: {
    x: 1.73,
    y: 0.85,
    theta: 3.14
  }
}"

# 우매대 하
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "
{
  robot_id: 1,
  order_id: 1,
  location_id: 3,
  target_pose: {
    x: 3.04,
    y: 0.491,
    theta: 1.57
  }
}"

# 우매대 좌
ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "
{
  robot_id: 1,
  order_id: 1,
  location_id: 2,
  target_pose: {
    x: 2.01,
    y: 1.47,
    theta: 0.0
  }
}"

pose:
    position:
      x: 3.3088653922559184
      y: 1.1099082400657065
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.6973268683075913
      w: 0.7167532621037224


#직선주행
ros2 service call /pickee/mobile/go_straight shopee_interfaces/srv/PickeeMobileGoStraight "
{
  distance: 0.47
}"

#회전
ros2 service call /pickee/mobile/rotate shopee_interfaces/srv/PickeeMobileRotate "
{
  angle: 1.57
}"



