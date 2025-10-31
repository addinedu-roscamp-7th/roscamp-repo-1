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

# Pickee 전용 이동 함수 (직선 이동, 회전)
from pickee_mobile.module.module_go_strait import run
from pickee_mobile.module.module_rotate import rotate


class PickeeMobileController(Node):
    def __init__(self):
        # ================= 노드 초기화 =================
        super().__init__('navigate_to_pose_client')
        self.get_logger().info('🚀 PickeeMobileController 노드 시작 🚀')

        # Nav2 action client 생성 (NavigateToPose 사용)
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # 서비스 서버 등록 (물류 시스템에서 목적지 요청)
        self.create_service(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location',
            self.pickee_move_to_location_callback
        )
        
        # 도착 알림 publisher
        self.arrival_publisher = self.create_publisher(
            PickeeMobileArrival, '/pickee/mobile/arrival', 10
        )
        
        # 현재 로봇 pose 상태 publisher
        self.pose_publisher = self.create_publisher(
            PickeeMobilePose, '/pickee/mobile/pose', 10
        )
        
        # 수정된 속도 topic 구독 → 로봇 상태 계산 (moving/idle 등)
        self.vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel_modified', self.vel_calculate_callback, 10
        )

        # ================= 변수 초기화 =================
        self.status = 'idle'
        self.working = 0  # (0=대기, 1=작업중)
        self.pose_timer = self.create_timer(0.2, self.pose_publisher_timer_callback)

        self.robor_id = 1
        self.order_id = 0
        self.location_id = 0
        self.old_location_id = 0

        self.currnet_x = 0.0
        self.currnet_y = 0.0
        self.current_radian = 0.0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.current_battery_level = 100.0  # TODO: 실제 배터리 값 연동 필요


    # ================= 서비스 요청 처리 함수 =================
    def pickee_move_to_location_callback(self, request, response):
        # 요청 정보 저장
        self.robor_id = request.robot_id
        self.order_id = request.order_id
        self.location_id = request.location_id

        self.get_logger().info("===== Move To Location Service Called =====")
        self.get_logger().info(f"robot_id       : {request.robot_id}")
        self.get_logger().info(f"order_id       : {request.order_id}")
        self.get_logger().info(f"location_id    : {request.location_id}")

        target = request.target_pose
        self.get_logger().info(f"target_pose    : (x={target.x}, y={target.y}, theta={target.theta})")

        if self.old_location_id > 0:
            run(self, -0.2)

        # Nav2 goal 전송
        try:
            self.send_goal(target.x, target.y, target.theta)
            response.success = True
            response.message = "Successfully received goal."
        except Exception as e:
            self.get_logger().error(f"Error sending goal: {e}")
            response.success = False
            response.message = f"Failed to receive goal: {e}"

        return response
    

    # ================= Nav2 Goal 전송 =================
    def send_goal(self, x, y, yaw_radian):

        

        self.working = 1  # 작업 중 표시
        self.start_time = time.time()  # 이동 시간 측정 시작

        self.get_logger().info(f'⏳ Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # 목표 좌표 설정
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # yaw(rad) → quaternion 변환
        goal_msg.pose.pose.orientation.z = math.sin(yaw_radian / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw_radian / 2.0)

        

        self.get_logger().info(f'🎯 Sending goal to ({x}, {y}), yaw={yaw_radian} rad')

        # 완료 시 오차 계산을 위해 목표 저장
        self.goal = [x, y, yaw_radian]

        # async 방식으로 goal 전송
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    # ================= Nav2 Goal 응답 =================
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ Goal rejected!')
            return

        self.get_logger().info('✅ Goal accepted!')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)


    # ================= 주행 중 Feedback 처리 =================
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pose = feedback.current_pose.pose

        # 현재 위치 업데이트 (실시간)
        self.currnet_x = pose.position.x
        self.currnet_y = pose.position.y
        
        # quaternion → yaw 변환
        qz = pose.orientation.z
        qw = pose.orientation.w
        self.current_radian = math.atan2(2.0 * qz * qw, 1.0 - 2.0 * (qz ** 2))

        self.get_logger().info(
            f'🔄 Feedback: x={self.currnet_x:.2f}, y={self.currnet_y:.2f}'
        )


    # ================= Nav2 Goal 완료 처리 =================
    def get_result_callback(self, future):
        self.working = 0  # 작업 완료
        status = future.result().status
        
        # 성공했을 때만 위치 오차 계산
        if status == GoalStatus.STATUS_SUCCEEDED:
            # 위치 오차 계산
            position_error = Pose2D()
            position_error.x = self.goal[0] - self.currnet_x
            position_error.y = self.goal[1] - self.currnet_y
            position_error.theta = self.goal[2] - self.current_radian

            # 이동 시간 계산
            travel_time = time.time() - self.start_time

            self.get_logger().info("✅ Goal reached successfully!")
            self.get_logger().info(f"⏱️ Travel time: {travel_time:.2f} sec")
            self.get_logger().info(
                f"📍 Error: x={position_error.x:.3f}, y={position_error.y:.3f}, θ={position_error.theta:.3f}"
            )

            # 도착 메시지 publish (백엔드/DB로 전송 가능)
            arrival_msg = PickeeMobileArrival()
            arrival_msg.robot_id = self.robor_id
            arrival_msg.order_id = self.order_id
            arrival_msg.location_id = self.location_id

            final_pose = Pose2D()
            final_pose.x = self.currnet_x
            final_pose.y = self.currnet_y
            final_pose.theta = self.current_radian

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

        self.old_location_id = self.location_id

    # ================= Pose 정보 Publish (0.2초마다) =================
    def pose_publisher_timer_callback(self):
        pose_msg = PickeeMobilePose()
        pose_msg.robot_id = self.robor_id
        pose_msg.current_pose.x = self.currnet_x
        pose_msg.current_pose.y = self.currnet_y
        pose_msg.current_pose.theta = self.current_radian
        pose_msg.linear_velocity = self.linear_velocity
        pose_msg.angular_velocity = self.angular_velocity
        pose_msg.battery_level = self.current_battery_level
        pose_msg.status = self.status

        self.pose_publisher.publish(pose_msg)


    # ================= 속도 callback → 로봇 상태 구분 =================
    def vel_calculate_callback(self, msg: Twist):
        # 선속도 magnitude 계산 (x,y 합성)
        self.linear_velocity = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.angular_velocity = msg.angular.z

        # 속도 기반 상태 머신
        if abs(self.linear_velocity) > 0 or abs(self.angular_velocity) > 0:
            self.status = 'moving'
        elif self.working == 0:
            self.status = 'idle'
        elif self.working == 1:
            self.status = 'stopped'
        else:
            self.status = 'error'


# ================= main 함수 =================
def main(args=None):
    rclpy.init(args=args)
    node = PickeeMobileController()

    # Multi-thread executor → 서비스 + action 동시에 처리 가능
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
