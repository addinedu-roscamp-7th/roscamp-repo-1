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

class GoStraight(Node):
    def __init__(self):
        super().__init__('go_straight_odom')
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

    def go_straight(self, distance) -> None:

        while self.yaw is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

        max_speed = 0.2
        Kp = 1.4
        Kd = 0.2
        a_max = 0.8
        latency = 0.12

        # 크기 보정은 유지, 부호는 보존
        distance = math.copysign(abs(distance) * 0.962, distance)
        dir_sign = 1.0 if distance >= 0 else -1.0 # 전진, 후진 판별

        # 오돔/헤딩 준비
        while (self.x is None) or (self.yaw is None):
            rclpy.spin_once(self)

        # 시작 위치(오돔) + 시작 헤딩
        sx, sy = float(self.x), float(self.y) # 출발시점 odom 값
        theta0 = float(self.yaw) # 출발시점 odom 값

        cmd = Twist()
        prev_err = None
        t_prev = time.monotonic() # 출발 시간
        target = abs(distance)  # 목적지까지 절대거리

        while rclpy.ok():
            rclpy.spin_once(self) 

            cx, cy = float(self.x), float(self.y)

            # 시작 헤딩으로 투영한 이동량(부호 포함)
            dx, dy = (cx - sx), (cy - sy)
            moved_signed = dx * math.cos(theta0) + dy * math.sin(theta0)
            moved_toward = dir_sign * moved_signed

            # 오차(양수)
            error = target - moved_toward
            if error < 0.005:  # 5mm
                break

            # PD
            now = time.monotonic()
            dt = max(now - t_prev, 1e-3)
            d_err = 0.0 if prev_err is None else (error - prev_err) / dt
            base = Kp * error + Kd * d_err
            prev_err, t_prev = error, now

            # 속도 명령 (부호 적용 + 제한)
            speed_cmd = dir_sign * base
            speed_cmd = max(min(speed_cmd, max_speed), -max_speed)

            # 정지거리 보정
            v = abs(speed_cmd)
            d_stop = v * latency + (v * v) / (2.0 * max(a_max, 1e-3))
            if error < d_stop:
                speed_cmd = dir_sign * max(0.05, v * 0.5)

            # 마찰 극복 최소속도
            if error > 0.02 and abs(speed_cmd) < 0.05:
                speed_cmd = dir_sign * 0.05

            cmd.linear.x = float(speed_cmd)
            cmd.angular.z = 0.0  # 필요하면 헤딩 홀드 PD 추가 가능
            self.cmd_pub.publish(cmd)
            time.sleep(0.02)

        self.stop()

    def stop(self) -> None:
        self.cmd_pub.publish(Twist())
        time.sleep(0.05)
        self.get_logger().info("🛑 STOP")
