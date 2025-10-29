import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import ArucoPose
from geometry_msgs.msg import Twist
import math
import time

from pickee_mobile.module.module_go_strait import run   # run(node, dist)
from pickee_mobile.module.module_rotate import rotate   # rotate(node, deg)

class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking")

        self.state = "SEARCHING"
        self.aruco_detected = False
        self.pose = None

        self.sub = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_modified", 10)

        self.timer = self.create_timer(0.2, self.fsm_step)

        self.get_logger().info("🤖 ArUco Docking FSM Started")

    def aruco_callback(self, msg: ArucoPose):
        self.aruco_detected = True
        self.pose = msg

    def fsm_step(self):
        if not self.aruco_detected:
            self.state_searching()
            return

        x = self.pose.x      # left-right mm
        y = self.pose.y      # up-down mm (unused)
        z = self.pose.z      # forward mm
        yaw = self.pose.pitch  # deg

        self.get_logger().info(
            f"[FSM={self.state}] x={x:.1f}mm y={y:.1f}mm z={z:.1f}mm yaw={yaw:.1f}°"
        )

        if self.state == "SEARCHING":
            self.state_align_yaw()
        
        elif self.state == "ALIGN_YAW":
            
            rotate(self, yaw)
            self.get_logger().info('첫 번째 회전 완료.')

            distance = math.sqrt(x**2 + z**2) / (6 * math.cos(math.radians(yaw)))
            self.get_logger().info(f'이동 거리: {distance:.3f} mm')

            run(self, 0.5)
            self.get_logger().info('직진 주행 완료.')

            # Step 3: rotate(-2 * theta_pitch)
            rotate(self, -2 * yaw)
            self.get_logger().info('두 번째 회전 완료.')

        # elif self.state == "ALIGN_YAW":
        #     if abs(yaw) > 5:
        #         rotate(self, yaw)  # correct yaw
        #     else:
        #         self.state = "ALIGN_X"

        # elif self.state == "ALIGN_X":
        #     if abs(x) > 30:
        #         offset = x / 1000.0  # mm → m
        #         run(self, offset)     # move sideways by turning wheels
        #     else:
        #         self.state = "APPROACH"

        # elif self.state == "APPROACH":
        #     if z > 60:
        #         forward = (z - 60) / 1000.0  # leave 60mm
        #         run(self, forward)
        #     else:
        #         self.state = "DOCKED"

        # elif self.state == "DOCKED":
        #     self.get_logger().info("✅ DOCKED SUCCESSFULLY!")
        #     self.publish_stop()
        #     return

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
