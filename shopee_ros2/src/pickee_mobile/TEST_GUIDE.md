# /pickee/mobile/arrival publish

ros2 topic pub /pickee/mobile/arrival shopee_interfaces/msg/PickeeMobileArrival "{robot_id: 1, order_id: 123, location_id: 5, final_pose: {x: 1.2, y: 3.4, theta: 1.57}, position_error: {x: 0.01, y: -0.02, theta: 0.005}, travel_time: 12.5, message: \"Arrived at location\"}"


# /pickee/mobile/move_to_location service_client

ros2 service call /pickee/mobile/move_to_location shopee_interfaces/srv/PickeeMobileMoveToLocation "{robot_id: 1, order_id: 100, location_id: 5, target_pose: {x: 1.2, y: 2.3, theta: 1.57}}"
