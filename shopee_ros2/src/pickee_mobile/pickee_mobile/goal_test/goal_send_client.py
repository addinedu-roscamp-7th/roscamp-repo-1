import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import math

class NavigateClient(Node):
    def __init__(self):
        super().__init__('navigate_to_pose_client')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def send_goal(self, x, y, yaw_deg):
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

        self.get_logger().info('✅ Goal accepted!')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pose = feedback.current_pose.pose
        self.get_logger().info(
            f'🔄 Feedback: current position: x={pose.position.x:.2f}, y={pose.position.y:.2f}'
        )

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('🎉 Goal reached!')

def main(args=None):
    rclpy.init(args=args)
    node = NavigateClient()

    # 목표 좌표(x, y)와 회전(yaw) 설정
    node.send_goal(x=-0.0383292734622955, y=-2.0135283470153809, yaw_deg=0.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
