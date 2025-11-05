#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
오도메트리 정보를 받아서 회전 각도를 계산하고,
지정된 각도만큼 회전하는 코드 (NumPy 2.x 호환 버전)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def euler_from_quaternion(q):
    """
    quaternion (x, y, z, w) → (roll, pitch, yaw)
    tf_transformations 없이 math 기반 계산
    """
    x, y, z, w = q

    # roll (x축 회전)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y축 회전)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z축 회전)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class OdomRotate(Node):
    """ 오도메트리를 기반으로 지정된 각도만큼 회전하는 노드 """

    def __init__(self, target_angle_deg: float):
        super().__init__('odom_rotate')

        # ---- 파라미터 선언 ----
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('vel_topic', '/cmd_vel_modified')
        self.declare_parameter('speed_angular', 0.2)  # [rad/s]
        self.declare_parameter('tolerance_deg', 2.0)  # 허용 오차 (deg)
        self.declare_parameter('rate_hz', 20)

        # ---- 파라미터 로드 ----
        self.odom_topic = self.get_parameter('odom_topic').value
        self.vel_topic = self.get_parameter('vel_topic').value
        self.wz_mag = abs(float(self.get_parameter('speed_angular').value))
        self.tol_deg = float(self.get_parameter('tolerance_deg').value)
        self.rate_hz = max(5, int(self.get_parameter('rate_hz').value))

        # 목표 회전 각도(rad)
        self.target_angle = math.radians(target_angle_deg)
        self.direction = 1.0 if self.target_angle >= 0 else -1.0

        # ---- 오도메트리 ----
        self.current_yaw = None
        self.start_yaw = None
        self.last_odom_time = time.time()
        self.start_time = None
        self.done = False

        # timeout 설정 (예상 회전시간의 3배)
        self.timeout = max(5.0, abs(self.target_angle) / max(self.wz_mag, 0.01) * 3.0)

        # ---- QOS 설정 ----
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        vel_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)
        self.pub = self.create_publisher(Twist, self.vel_topic, vel_qos)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._step)

        # self.get_logger().info(
        #     f"목표 회전각={target_angle_deg:.1f}°, 각속도={self.wz_mag:.2f}rad/s, 오차={self.tol_deg:.1f}°, timeout≈{self.timeout:.1f}s"
        # )

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.current_yaw = yaw
        self.last_odom_time = time.time()

        if self.start_yaw is None:
            self.start_yaw = yaw
            self.start_time = self.last_odom_time
            # self.get_logger().info(f"시작 각도={math.degrees(yaw):.1f}°")

    def _step(self):
        if self.done:
            return

        now = time.time()

        if self.current_yaw is None:
            if now - self.last_odom_time > 1000.0:
                # self.get_logger().warn('waiting /odom ...')
                pass
            return

        # 오도메트리 수신 중단 시 종료
        if now - self.last_odom_time > 1.0:
            # self.get_logger().error('odom lost → stop')
            self._finish()
            return

        turned = self._angle_diff(self.current_yaw, self.start_yaw)
        remain = abs(self.target_angle) - abs(turned)

        if math.degrees(remain) <= self.tol_deg:
            # self.get_logger().info(
            #     f"회전 완료: {math.degrees(turned):.1f}°, 목표={math.degrees(self.target_angle):.1f}°"
            # )
            self._finish()
            return

        if now - self.start_time > self.timeout:
            # self.get_logger().warn(f"timeout {self.timeout:.1f}s")
            self._finish()
            return

        cmd = Twist()
        cmd.angular.z = float(self.direction * self.wz_mag)
        self.pub.publish(cmd)

    def _finish(self):
        stop = Twist()
        self.pub.publish(stop)
        self.pub.publish(stop)
        # self.get_logger().info('STOP 회전 종료')
        self.done = True

    @staticmethod
    def _angle_diff(a, b):
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff


# def rotate(angle_deg: float):
#     rclpy.init()
#     node = OdomRotate(angle_deg)
#     try:
#         while rclpy.ok() and not node.done:
#             rclpy.spin_once(node, timeout_sec=0.1)
#     except KeyboardInterrupt:
#         node._finish()
#         rclpy.shutdown()
#     finally:
#         try:
#             node.destroy_node()
#         finally:
#             if rclpy.ok():
#                 rclpy.shutdown()

def rotate_standalone(angle_deg: float):
    rclpy.init()
    node = OdomRotate(angle_deg)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def rotate(node: Node, angle_rad: float):
    """
    이미 실행 중인 ROS 노드 내부에서 회전할 때 사용.
    rclpy.init() / shutdown() 없이 주어진 노드의 publisher 사용.
    """
    wz_mag = 0.2  # [rad/s]
    tol_deg = 2.0
    rate_hz = 20
    angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    direction = 1.0 if angle_rad >= 0 else -1.0
    target_angle = abs(angle_rad)
    duration = target_angle / wz_mag

    pub = node.create_publisher(Twist, '/cmd_vel_modified', 10)
    cmd = Twist()
    cmd.angular.z = float(direction * wz_mag)

    end_time = time.time() + duration
    while time.time() < end_time:
        pub.publish(cmd)
        time.sleep(1.0 / rate_hz)

    pub.publish(Twist())
    # node.get_logger().info(f"✅ 내부 회전 완료: {angle_rad:.1f}°")


def main():
    # +90도 회전
    rotate(90.0)
    # -90도 회전
    rotate(-90.0)
    # 180도 회전
    rotate(180.0)

if __name__ == '__main__':
    main()
