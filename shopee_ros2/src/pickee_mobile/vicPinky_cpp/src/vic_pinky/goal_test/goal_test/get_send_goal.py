import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import math

class GetSendGoal(Node):

    def __init__(self):
        super().__init__('get_pose')
        self.goal_x_old = 0.0
        self.goal_y_old = 0.0

        self.subs_get_goal = self.create_subscription(
            PointStamped,
            'clicked_point',
            self.get_goal_callback,
            10)
        
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.subs_get_goal  # prevent unused variable warning

    def get_goal_callback(self, msg):
        self.get_logger().info('Goal position: x=%.2f, y=%.2f' % (msg.point.x, msg.point.y))
        self.goal_x = msg.point.x
        self.goal_y = msg.point.y
        self.send_goal(self.goal_x, self.goal_y)

    def send_goal(self, x, y):
        self.get_logger().info(f'⏳ Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # 목표 좌표
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # 목표 방향 각도 계산
        dx = x - self.goal_x_old
        dy = y - self.goal_y_old

        yaw = math.atan2(dy, dx)  # 라디안
        yaw_deg = math.degrees(yaw)

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

        self.goal_x_old = x
        self.goal_y_old = y

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

    node = GetSendGoal()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


#clicked_point.point.x
#
#
