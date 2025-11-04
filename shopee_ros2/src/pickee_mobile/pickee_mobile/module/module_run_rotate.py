#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math, time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped


def euler_from_quaternion(q):
    """
    ROS2 기본 Quaternion -> Yaw 변환
    roll, pitch 필요 없어서 계산 제외
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class SimpleMotion(Node):
    def __init__(self):
        super().__init__('simple_motion')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_modified', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        self.x = self.y = self.yaw = None
        self.mx = self.my = None  # map 기준

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = euler_from_quaternion(q)

    def amcl_cb(self, msg):
        self.mx = msg.pose.pose.position.x
        self.my = msg.pose.pose.position.y


    def move_straight(self, distance, max_speed=0.2, Kp=1.4, Kd=0.2,
                    a_max=0.8, latency=0.12, frame='odom'):
        # 보정은 크기에만 적용하고, 부호는 유지
        distance = math.copysign(abs(distance)*0.962, distance)
        dir_sign = 1.0 if distance >= 0 else -1.0

        # 시작 위치 준비 (+ 시작 헤딩)
        while (self.x is None) or (frame == 'map' and self.mx is None) or (self.yaw is None):
            rclpy.spin_once(self)

        if frame == 'map':
            sx, sy = self.mx, self.my
            get_xy = lambda: (self.mx, self.my)
        else:
            sx, sy = self.x, self.y
            get_xy = lambda: (self.x, self.y)

        theta0 = self.yaw  # 시작 헤딩

        cmd = Twist()
        prev_err = None
        t_prev = time.time()

        # 남은 거리를 '양의 값'으로 다루고, 속도에만 부호(dir_sign)를 적용
        target = abs(distance)

        while rclpy.ok():
            rclpy.spin_once(self)
            cx, cy = get_xy()

            # 시작 헤딩으로 투영한 '부호 있는' 이동거리
            dx, dy = (cx - sx), (cy - sy)
            moved_signed = dx*math.cos(theta0) + dy*math.sin(theta0)
            # 목표 방향으로 간 성분만 취함
            moved_toward = dir_sign * moved_signed

            # 남은 거리(항상 양수)
            error = target - moved_toward
            if error < 0.005:  # 5mm
                break

            # PD
            now = time.time()
            dt = max(now - t_prev, 1e-3)
            d_err = 0.0 if prev_err is None else (error - prev_err) / dt
            base = Kp*error + Kd*d_err
            prev_err, t_prev = error, now

            # 속도 명령: 방향 부호 적용
            speed_cmd = dir_sign * base
            speed_cmd = max(min(speed_cmd, max_speed), -max_speed)

            # 정지거리 보정
            v = abs(speed_cmd)
            d_stop = v*latency + (v*v)/(2.0*max(a_max, 1e-3))
            if error < d_stop:
                speed_cmd = dir_sign * max(0.05, v*0.5)

            # 마찰 극복용 최소속도
            if error > 0.02 and abs(speed_cmd) < 0.05:
                speed_cmd = dir_sign * 0.05

            cmd.linear.x = speed_cmd
            self.cmd_pub.publish(cmd)
            time.sleep(0.02)

        self.stop()



    def rotate(self, angle_deg, Kp=2.2, Kd=0.18, max_w=0.7,
            alpha_max=3.0, latency=0.10, done_deg=1.0):

        while self.yaw is None:
            rclpy.spin_once(self)

        target_yaw = self.normalize_angle(self.yaw + math.radians(angle_deg))
        cmd = Twist()

        prev_err = None
        t_prev = time.time()
        done_rad = math.radians(done_deg)

        self.get_logger().info(f"🌀 Rotating {angle_deg:.2f}° (shortest+PD)")

        while rclpy.ok():
            rclpy.spin_once(self)

            # 최단 경로 오차(라디안, -pi~pi)
            err = self.normalize_angle(target_yaw - self.yaw)
            if abs(err) < done_rad:
                break

            # --- PD 제어 ---
            now = time.time()
            dt = max(now - t_prev, 1e-3)
            derr = 0.0 if prev_err is None else (err - prev_err) / dt
            w_cmd = Kp * err + Kd * derr
            prev_err, t_prev = err, now

            # 속도 제한
            w_cmd = max(min(w_cmd, max_w), -max_w)

            # 부드러운 멈춤(정지각 보정: 관성+지연 고려)
            # 예상 정지각 ≈ |w|*latency + w^2/(2*alpha_max)
            w = abs(w_cmd)
            stop_angle = w * latency + (w * w) / (2.0 * max(alpha_max, 1e-6))
            if abs(err) < stop_angle:
                # 남은 각도가 작다면 조금 더 천천히
                w_cmd = math.copysign(max(0.05, w * 0.5), err)

            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)
            time.sleep(0.02)

        self.stop()

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def stop(self):
        self.cmd_pub.publish(Twist())
        time.sleep(0.05)
        self.get_logger().info("✅ STOP")


def main():
    rclpy.init()
    robot = SimpleMotion()

    time.sleep(1.0)   # wait for odom
    robot.move_straight(0.47)  # 1m forward

    time.sleep(10.0)   # wait for odom
    robot.move_straight(-0.47)  # 1m forward

    time.sleep(10.0)   # wait for odom
    robot.move_straight(0.47)  # 1m forward

    robot.get_logger().info("🎉 Motion complete")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
