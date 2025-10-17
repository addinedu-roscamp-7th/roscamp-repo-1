import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('get_pose')

        self.subs_get_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.get_pose_callback,
            10)

        # self.subs_get_goal = self.create_subscription(
        #     PointStamped,
        #     'clicked_point',
        #     self.get_goal_callback,
        #     10)
        
        
        
        # self.subs_get_goal  # prevent unused variable warning
        self.subs_get_pose

    # def get_goal_callback(self, msg):
    #     self.get_logger().info('Goal position: x=%.2f, y=%.2f' % (msg.point.x, msg.point.y))
    
    def get_pose_callback(self, msg):
        self.get_logger().info('Current position: x=%.2f, y=%.2f' % (msg.pose.pose.position.x, msg.pose.pose.position.y))


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


#clicked_point.point.x
#
#
