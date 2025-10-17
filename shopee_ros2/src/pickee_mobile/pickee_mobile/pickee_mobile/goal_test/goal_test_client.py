import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.timer = self.create_timer(1.0, self.publish_goal)
        self.sent = False

    def publish_goal(self):
        if self.sent:
            return

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()

        # ✅ 목표 좌표 설정
        goal.pose.position.x = 0.31
        goal.pose.position.y = 0.13
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = -0.719458520846076
        goal.pose.orientation.w = 0.6945354107473402

        # goal.pose.position.x = 0.504417292657324
        # goal.pose.position.y = 0.047986208575005636
        # goal.pose.position.z = 0.0

        # goal.pose.orientation.x = 0.0
        # goal.pose.orientation.y = 0.0
        # goal.pose.orientation.z = -0.3113985056872772
        # goal.pose.orientation.w = 0.9502794171483094



        self.publisher_.publish(goal)
        self.sent = True


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
