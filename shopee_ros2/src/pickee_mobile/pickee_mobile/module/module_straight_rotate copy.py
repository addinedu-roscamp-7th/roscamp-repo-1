#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from shopee_interfaces.srv import PickeeMobileGoStraight, PickeeMobileRotate


def euler_yaw_from_quaternion(q) -> float:
    """Quaternion (x,y,z,w) → yaw(rad). roll/pitch는 무시."""
    x, y, z, w = q.x, q.y, q.z, q.w
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class SimpleMotion(Node):
    """오도메트리를 구독해 직선 주행/회전을 수행하는 노드."""

    def __init__(self) -> None:
        super().__init__('straight_rotate_Node')

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_modified', 10)

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Service Server
        self.create_service(PickeeMobileGoStraight, '/go_straight', self.go_straight)

        self.x = None
        self.y = None
        self.yaw = None

        self.get_logger().info("✅ SimpleMotion initialized")

    # ---------------------------
    # Callbacks
    # ---------------------------
    def odom_cb(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_yaw_from_quaternion(msg.pose.pose.orientation)

    # ---------------------------
    # Utils
    # ---------------------------
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def stop(self) -> None:
        self.cmd_pub.publish(Twist())
        time.sleep(0.05)
        self.get_logger().info("🛑 STOP")

    # ---------------------------
    # Straight motion (distance PD)
    # ---------------------------
    def go_straight(
        self,
        distance: float,
        max_speed: float = 0.2,
        Kp: float = 1.4,
        Kd: float = 0.2,
        a_max: float = 0.8,
        latency: float = 0.12,
    ) -> None:

        # 크기 보정은 유지, 부호는 보존
        distance = math.copysign(abs(distance) * 0.962, distance)
        dir_sign = 1.0 if distance >= 0 else -1.0

        # 오돔/헤딩 준비
        while (self.x is None) or (self.yaw is None):
            rclpy.spin_once(self)

        # 시작 위치(오돔) + 시작 헤딩
        sx, sy = float(self.x), float(self.y)
        theta0 = float(self.yaw)

        cmd = Twist()
        prev_err = None
        t_prev = time.monotonic()

        target = abs(distance)  # 남은 거리는 양수로

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


    # ---------------------------
    # Rotation (shortest-path PD)
    # ---------------------------
    def rotate(
        self,
        angle_rad: float,
        Kp: float = 2.2,
        Kd: float = 0.18,
        max_w: float = 0.3,
        alpha_max: float = 3.0,
        latency: float = 0.10,
        done_deg: float = 1.0,
    ) -> None:
        """라디안(+CCW, -CW) 기준 최단경로 회전 + PD + 정지각 보정."""
        angle_rad *= 0.95  # 소규모 오버슈트 보정 스케일

        # yaw 준비
        while self.yaw is None:
            rclpy.spin_once(self)

        target_yaw = self.normalize_angle(float(self.yaw) + float(angle_rad))
        cmd = Twist()

        prev_err: Optional[float] = None
        t_prev = time.monotonic()
        done_rad = math.radians(done_deg)

        self.get_logger().info(f"🌀 Rotating {math.degrees(angle_rad):.2f}° (shortest+PD)")

        while rclpy.ok():
            rclpy.spin_once(self)

            # 최단 경로 오차
            err = self.normalize_angle(target_yaw - float(self.yaw))
            if abs(err) < done_rad:
                break

            # PD
            now = time.monotonic()
            dt = max(now - t_prev, 1e-3)
            derr = 0.0 if prev_err is None else (err - prev_err) / dt
            w_cmd = Kp * err + Kd * derr
            prev_err, t_prev = err, now

            # 속도 제한
            w_cmd = max(min(w_cmd, max_w), -max_w)

            # 정지각 보정 (관성+지연)
            w = abs(w_cmd)
            stop_angle = w * latency + (w * w) / (2.0 * max(alpha_max, 1e-6))
            if abs(err) < stop_angle:
                w_cmd = math.copysign(max(0.05, w * 0.5), err)

            cmd.linear.x = 0.0
            cmd.angular.z = float(w_cmd)
            self.cmd_pub.publish(cmd)
            time.sleep(0.02)

        self.stop()


def main() -> None:
    rclpy.init()
    robot = SimpleMotion()

    # 예시: -90° 회전 두 번
    time.sleep(1.0)  # odom 대기
    robot.rotate(math.radians(-90))
    time.sleep(3.0)
    robot.rotate(math.radians(-90))

    robot.get_logger().info("🎉 Motion complete")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
