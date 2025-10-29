import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import ArucoPose
from geometry_msgs.msg import Twist
import math
import time

from pickee_mobile.module.module_go_strait import run   # run(node, dist)
from pickee_mobile.module.module_rotate import rotate   # rotate(node, deg)
from std_msgs.msg import Bool

class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking")

        self.state = "SEARCHING"
        self.aruco_detected = False
        self.pose = None
        self.cmd_vel = Twist()

        self.cmd_pub = self.create_publisher(
            Twist, 
            "/cmd_vel_modified", 
            10)
        self.docking_in_progress_pub = self.create_publisher(
            bool, 
            "/pickee/mobile/docking_in_progress", 
            10)


        self.sub = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_callback,
            10
        )

        self.get_logger().info("🤖 ArUco Docking FSM Started")

    def aruco_callback(self, msg: ArucoPose):

        x = msg.x      # left-right mm
        z = msg.z      # forward mm
        yaw = msg.pitch  # deg

        if x < 0 and yaw > 0:
            self.cmd_vel.angular.z = -0.2
        elif x > 0 and yaw > 0:
            self.cmd_vel.angular.z = -0.1
        elif x > 0 and yaw < 0:
            self.cmd_vel.angular.z = 0.2
        elif x < 0 and yaw < 0:
            self.cmd_vel.angular.z = 0.1

        if z > 200:
            self.cmd_vel.linear.x = 0.2
        else:
            self.cmd_vel.linear.x = 0.0
            self.publish_stop()
            self.get_logger().info("도킹 완료!")
            self.docking_in_progress_pub.publish(False)
            return
        
        

    def state_searching(self):
        self.get_logger().info("🔎 Searching marker...")
        cmd = Twist()
        cmd.angular.z = 0.2
        # self.cmd_pub.publish(cmd)

    def state_align_yaw(self):
        self.get_logger().info("🎯 Align yaw...")
        self.state = "ALIGN_YAW"

    def publish_stop(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = ArucoDocking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
