import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
import numpy as np

class ObstacleSlowdown(Node):
    def __init__(self):
        super().__init__('obstacle_slowdown')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safety', 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        self.latest_cmd = Twist()
        self.slowdown_factor = 0.4  # 장애물 가까우면 속도 x 0.4

    def cmd_callback(self, msg):
        self.latest_cmd = msg

    def costmap_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))

        # 로봇 주변 앞 영역만 검사
        forward_area = data[int(height/2):, int(width/3):int(2*width/3)]

        # 장애물 근처 여부 판단
        if np.any(forward_area >= 200):  # high cost zone
            cmd = Twist()
            cmd.linear.x = self.latest_cmd.linear.x * self.slowdown_factor
            cmd.angular.z = self.latest_cmd.angular.z
        else:
            cmd = self.latest_cmd

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = ObstacleSlowdown()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
