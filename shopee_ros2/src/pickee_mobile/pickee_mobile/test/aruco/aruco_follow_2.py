# Aruco marker로 접근 후 마커가 카메라 벗어나면 재정렬 후 작업, z랑 yaw로 도착 결정
# Python Standard Library
import math
import time

# ROS2 Core Libraries
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# ROS2 Message Types
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from shopee_interfaces.msg import ArucoPose

# Project Modules
from pickee_mobile.module.module_go_strait import run   # run(node, dist)
from pickee_mobile.module.module_rotate import rotate   # rotate(node, deg)



class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking")

        self.state = "SEARCHING"
        self.aruco_detected = False
        self.pose = None
        self.cmd_vel = Twist()
        self.old_x = 0.0
        self.old_yaw = 0.0
        self.aruco_lost_count = 0
        self.docked = False


        self.cmd_pub = self.create_publisher(
            Twist, 
            "/cmd_vel_modified", 
            10)
        self.docking_in_progress_pub = self.create_publisher(
            Bool, 
            "/pickee/mobile/docking_in_progress", 
            10)


        self.sub = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_callback,
            1

        )

        self.get_logger().info("🤖 ArUco Docking FSM Started")

    def aruco_callback(self, msg: ArucoPose):

        x = msg.x      # left-right mm
        z = msg.z      # forward mm
        yaw = msg.pitch  # deg
        
        self.get_logger().info(f"📍 ArUco Detected - x: {x} mm, z: {z} mm, yaw: {yaw} deg")

        # x = 카메라 중심 기준 마커가 오른쪽에 있는 정도
        # yaw = 양수면 마커가 왼쪽에, 음수면 오른쪽에 있음
        if abs(x) > 10:
            self.aruco_lost_count = 0
            if x < 0 and yaw > 0:
                self.cmd_vel.angular.z = 0.1
            elif x > 0 and yaw > 0:
                self.cmd_vel.angular.z = -0.1
            elif x > 0 and yaw < 0:
                self.cmd_vel.angular.z = -0.1
            elif x < 0 and yaw < 0:
                self.cmd_vel.angular.z = 0.1
        else:
            self.cmd_vel.angular.z = 0.0

        if z > 190:
            self.aruco_lost_count = 0
            scale = max(min((z - 200) / 1000, 0.2), 0.05)
            self.cmd_vel.linear.x = scale
        
        elif z == 0.0 and x == 0.0 and yaw == 0.0:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.publish_stop()
            self.get_logger().info("❌ ArUco marker lost. Stopping.")
            self.aruco_lost_count += 1
            if self.aruco_lost_count <= 3:
                self.Realign()

            else:
                self.get_logger().info("⚠️ ArUco marker lost for too long. Stopping docking.")
                self.docking_in_progress_pub.publish(Bool(data=False))
                self.publish_stop()
            return
            

        elif z <= 190 and abs(yaw) > 10:
            self.Realign()

        else:
            
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            msg.x = 0
            msg.z = 0
            msg.yaw = 0
            self.publish_stop()
            self.get_logger().info("✅ Docking complete!")
            self.docking_in_progress_pub.publish(Bool(data=False))
            run(self, 0.05)
            time.sleep(5)
            return
        
        self.cmd_pub.publish(self.cmd_vel)
        self.old_x = x
        self.old_yaw = yaw

    def Realign(self):
        self.get_logger().info("🔄 Realigning to find ArUco marker...")
        if self.old_yaw > 0 and self.old_x > 0:
            old_yaw_diff = 15

        elif self.old_yaw > 0 and self.old_x < 0:
            old_yaw_diff = 30

        elif self.old_yaw < 0 and self.old_x > 0:
            old_yaw_diff = -30
        
        elif self.old_yaw < 0 and self.old_x < 0:
            old_yaw_diff = -15

        rotate(self, -self.old_yaw - old_yaw_diff)
        time.sleep(0.5)
        run(self, -0.1)
        time.sleep(0.5)
        rotate(self, old_yaw_diff)
        time.sleep(2)
        
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
