# /pickee/mobile/arrival publish

ros2 topic pub /pickee/mobile/arrival shopee_interfaces/msg/PickeeMobileArrival "
{
  robot_id: 1,
  order_id: 123,
  location_id: 5,
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



# /pickee/mobile/move_to_location service_client
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
