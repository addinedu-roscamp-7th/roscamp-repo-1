import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

from shopee_interfaces.srv import PickeeMobileMoveToLocation
from shopee_interfaces.msg import PickeeMobileArrival, Pose2D, PickeeMobilePose



class PickeeMobileController(Node):
    def __init__(self):
        super().__init__('navigate_to_pose_client')
        self.get_logger().info('🚀 PickeeMobileController 노드 시작 🚀')

        self._action_client = ActionClient(self, 
                                           NavigateToPose, 
                                           '/navigate_to_pose')

        self.create_service(PickeeMobileMoveToLocation, 
                            '/pickee/mobile/move_to_location', 
                            self.pickee_move_to_location_callback)
        
        self.arrival_publisher = self.create_publisher(PickeeMobileArrival, 
                                                       '/pickee/mobile/arrival',
                                                         10)
        
        self.pose_publisher = self.create_publisher(PickeeMobilePose,
                                                    '/pickee/mobile/pose',
                                                    10)
        
        self.vel_subscriber = self.create_subscription(Twist,
                                                       '/cmd_vel_modified',
                                                       self.vel_calculate_callback,
                                                       10)
        
        #초기 설정
        self.status = 'idle'
        self.working = 0 # 대기중
        




    def pickee_move_to_location_callback(self, request, response):
        # 명령 관련 정보 저장
        self.robor_id = request.robot_id
        self.order_id = request.order_id
        self.location_id = request.location_id

        self.get_logger().info("===== Move To Location Service Called =====")
        self.get_logger().info(f"robot_id       : {request.robot_id}")
        self.get_logger().info(f"order_id       : {request.order_id}")
        self.get_logger().info(f"location_id    : {request.location_id}")
        target = request.target_pose
        self.get_logger().info(f"target_pose    : (x={target.x}, y={target.y}, theta={target.theta})")

        # 목적지 이동 액션 실행
        try:
            self.send_goal(target.x, target.y, math.degrees(target.theta)) 
            response.success = True
            response.message = "Successfully received goal."
        except Exception as e:
            self.get_logger().error(f"Error sending goal: {e}")
            response.success = False
            response.message = f"Failed to receive goal: {e}"

        return response
    
    def send_goal(self, x, y, yaw_deg):
        self.working = 1  # 작업 중 상태 설정
        self.start_time = time.time()  # 목표 전송 시각 기록
        self.get_logger().info(f'⏳ Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # 목표 좌표
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # 목표 방향 (쿼터니언 변환)
        yaw = math.radians(yaw_deg)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'🎯 Sending goal to ({x}, {y}), yaw={yaw_deg}°')
        self.goal = [x, y, yaw_deg]# 목적지 좌표 저장 이후 도착 위치와 비교

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ Goal rejected!')
            return

        self.get_logger().info(' Goal accepted!')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pose = feedback.current_pose.pose
        self.get_logger().info(
            f'🔄 Feedback: current position: x={pose.position.x:.2f}, y={pose.position.y:.2f}'
        )

        self.currnet_x = pose.position.x
        self.currnet_y = pose.position.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        self.current_theta = math.atan2(2.0 * qz * qw, 1.0 - 2.0 * (qz ** 2))

    def get_result_callback(self, future):
        working = 0  # 작업 완료 상태 설정
        status = future.result().status
        result = future.result().result

        
        if status == GoalStatus.STATUS_SUCCEEDED:
            # 도착 위치와 목표 위치 비교
            position_error = Pose2D()
            position_error.x = abs(self.goal[0] - self.currnet_x)
            position_error.y = abs(self.goal[1] - self.currnet_y)
            position_error.theta = abs(self.goal[2] - self.current_theta)

            self.end_time = time.time()  # 도착 시각 기록
            travel_time = self.end_time - self.start_time  # 이동 시간 계산


            self.get_logger().info("✅ Goal reached successfully!")
            self.get_logger().info(f"Total travel time: {travel_time:.2f} seconds")
            self.get_logger().info(f"Position error: x={position_error.x:.3f}, y={position_error.y:.3f}, theta={position_error.theta:.3f}")

            arrival_msg = PickeeMobileArrival()
            arrival_msg.robot_id = self.robor_id
            arrival_msg.order_id = self.order_id
            arrival_msg.location_id = self.location_id
            final_pose = Pose2D()
            final_pose.x = self.currnet_x
            final_pose.y = self.currnet_y
            final_pose.theta = self.current_theta
            arrival_msg.final_pose = final_pose
            arrival_msg.position_error = position_error
            arrival_msg.travel_time = travel_time
            arrival_msg.message = "Success."

            self.arrival_publisher.publish(arrival_msg)

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info("❌ Goal aborted.")

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("⚠️ Goal canceled.")

        else:
            self.get_logger().info(f"Unknown status: {status}")


        self.get_logger().info('status')
    
    def pose_publisher_timer_callback(self):
        pose_msg = PickeeMobilePose()
        pose_msg.robot_id = self.robor_id
        pose_msg.pose.x = self.currnet_x
        pose_msg.pose.y = self.currnet_y
        pose_msg.pose.theta = self.current_theta
        pose_msg.linear_velocity = self.linear_velocity
        pose_msg.angular_velocity = self.angular_velocity

        self.pose_publisher.publish(pose_msg)

    def vel_calculate_callback(self, msg: Twist):
        self.linear_velocity = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.angular_velocity = msg.angular.z

        if abs(self.linear_velocity) > 0 or abs(self.angular_velocity) > 0:
            self.status = 'moving'
        
        elif abs(self.linear_velocity) == 0 and abs(self.angular_velocity) == 0 and self.working == 0:
            self.status = 'idle'
        
        elif abs(self.linear_velocity) == 0 and abs(self.angular_velocity) == 0 and self.working == 1:
            self.status = 'stopped'
        
def main(args=None):
    rclpy.init(args=args)
    node = PickeeMobileController()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
