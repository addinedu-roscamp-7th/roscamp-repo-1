import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def euler_yaw_from_quaternion(q) -> float:
    """Quaternion (x,y,z,w) → yaw(rad). roll/pitch는 무시."""
    x, y, z, w = q.x, q.y, q.z, q.w
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

class Rotate(Node):
    def __init__(self):
        super().__init__('rotate_odom')
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_modified', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, odom_qos)

        self.x = self.y = self.yaw = None

    def odom_cb(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_yaw_from_quaternion(msg.pose.pose.orientation)

    def rotate(self, angle_rad: float):

        while self.yaw is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

        Kp, Kd   = 2.2, 0.18
        max_w    = 0.3
        alpha_max= 3.0
        latency  = 0.10
        done_deg = 2.0     
        angle_scale = 0.95

        angle_rad *= angle_scale
        target_yaw = normalize_angle(float(self.yaw) + angle_rad)
        done_rad = math.radians(done_deg)

        cmd = Twist()
        prev_err = None
        t_prev = time.monotonic()

        self.get_logger().info(f"🌀 Rotating {math.degrees(angle_rad):.2f}° (PD)")


        while rclpy.ok():
            rclpy.spin_once(self)  # ✅ 반드시 추가

            err = normalize_angle(target_yaw - float(self.yaw))
            if abs(err) < done_rad:
                break

            now = time.monotonic()
            dt  = max(now - t_prev, 1e-3)
            derr= 0.0 if prev_err is None else (err - prev_err)/dt
            w_cmd = Kp*err + Kd*derr
            prev_err, t_prev = err, now

            w_cmd = max(min(w_cmd, max_w), -max_w)

            wmag = abs(w_cmd)
            stop_angle = wmag*latency + (wmag*wmag)/(2.0*max(alpha_max, 1e-6))
            if abs(err) < stop_angle:
                w_cmd = math.copysign(max(0.03, wmag*0.5), err)

            cmd.linear.x = 0.0
            cmd.angular.z = float(w_cmd)
            # print(f'')
            self.cmd_pub.publish(cmd)
            # self.get_logger().info(f"w_cmd {w_cmd}")
            # time.sleep(0.02)
            rclpy.spin_once(self, timeout_sec=0.02)
        
        self.stop()

    def stop(self) -> None:
        self.cmd_pub.publish(Twist())
        time.sleep(0.05)
        self.get_logger().info("🛑 STOP")
