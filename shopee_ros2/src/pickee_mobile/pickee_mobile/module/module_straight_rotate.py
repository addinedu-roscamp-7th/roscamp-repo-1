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
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


# PickeeMobileGoStraight
# float32 distance
# ---
# bool success
# PickeeMobileRotate
# float32 angle
# ---
# bool success

def euler_yaw_from_quaternion(q) -> float:
    """Quaternion (x,y,z,w) → yaw(rad). roll/pitch는 무시."""
    x, y, z, w = q.x, q.y, q.z, q.w
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class SimpleMotion(Node):
    def __init__(self) -> None:
        super().__init__('straight_rotate_Node')

        # 1) Reentrant 그룹
        self.cb = ReentrantCallbackGroup()

        # 2) QoS: odom은 보통 BEST_EFFORT
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Publisher (필요시 /cmd_vel 로)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_modified', 10)

        # Subscribers (reentrant 그룹으로)
        self.create_subscription(Odometry, '/odom', self.odom_cb, odom_qos,
                                 callback_group=self.cb)

        # Services (reentrant 그룹으로)
        self.create_service(PickeeMobileGoStraight, '/pickee/mobile/go_straight',
                            self.go_straight_cb, callback_group=self.cb)
        self.create_service(PickeeMobileRotate, '/pickee/mobile/rotate',
                            self.rotate_rad_cb, callback_group=self.cb)

        self.x = self.y = self.yaw = None

    # ---------------------------
    # Callbacks
    # ---------------------------
    def odom_cb(self, msg: Odometry) -> None:
        # 현재 odom값 업데이트
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
    def go_straight_cb(self,
                    req: PickeeMobileGoStraight.Request,
                    res: PickeeMobileGoStraight.Response):
        
        
        # 1) 요청 파싱
        distance = float(req.distance)
        self.get_logger().info(f'Start distance = {distance}')
        # 2) 제어 파라미터
        max_speed = 0.2
        Kp, Kd = 1.4, 0.2
        a_max, latency = 0.8, 0.12

        try:
            distance = math.copysign(abs(distance) * 0.962, distance)
            dir_sign = 1.0 if distance >= 0 else -1.0 # 전진, 후진 판별

            while (self.x is None) or (self.yaw is None):
                time.sleep(0.01)


            sx, sy = float(self.x), float(self.y) # 출발시점 odom 값
            theta0 = float(self.yaw) # 출발시점 odom 값

            cmd = Twist()
            prev_err = None
            t_prev = time.monotonic() # 출발 시간
            target = abs(distance) # 목적지까지 절대거리

            while rclpy.ok():
                
                #전진 거리 계산
                cx, cy = float(self.x), float(self.y) # 현재 odom 값
                dx, dy = (cx - sx), (cy - sy)
                moved_signed = dx * math.cos(theta0) + dy * math.sin(theta0)
                moved_toward = dir_sign * moved_signed # 이동 거리

                error = target - moved_toward
                if error < 0.005:
                    self.get_logger().info(f'Success!!!')
                    break

                now = time.monotonic() # 현재시간
                dt = max(now - t_prev, 1e-3) # 단위시간
                d_err = 0.0 if prev_err is None else (error - prev_err) / dt # 단위에러
                base = Kp * error + Kd * d_err
                prev_err, t_prev = error, now
                speed_cmd = dir_sign * base # 방향 결정
                speed_cmd = max(min(speed_cmd, max_speed), -max_speed)

                v = abs(speed_cmd)
                d_stop = v * latency + (v * v) / (2.0 * max(a_max, 1e-3))
                if error < d_stop:
                    speed_cmd = dir_sign * max(0.05, v * 0.5)

                if error > 0.02 and abs(speed_cmd) < 0.05:
                    speed_cmd = dir_sign * 0.05

                cmd.linear.x = float(speed_cmd)
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(0.02)
                self.get_logger().info(f'Loop End')

            self.get_logger().info(f'END')
            self.stop()
            # 3) 응답 채우기
            res.success = True
        except Exception as e:
            self.stop()
            res.success = False
        return res   # ⬅ 반드시 반환!



    # ---------------------------
    # Rotation (shortest-path PD)
    # ---------------------------
    def rotate_rad_cb(self, req, res):
        angle_rad = float(req.angle)

        Kp, Kd   = 2.2, 0.18
        max_w    = 0.3
        alpha_max= 3.0
        latency  = 0.10
        done_deg = 2.0          # 1.0 → 2.0 (끝조건 완화)
        angle_scale = 1.0       # 우선 고정 보정 제거

        try:
            # yaw 준비 (executor가 스핀하므로 sleep만)
            t0 = time.monotonic()
            while self.yaw is None and rclpy.ok():
                if time.monotonic() - t0 > 3.0:
                    res.success = False
                    return res
                time.sleep(0.01)

            angle_rad *= angle_scale
            target_yaw = self.normalize_angle(float(self.yaw) + angle_rad)
            done_rad = math.radians(done_deg)

            cmd = Twist()
            prev_err = None
            t_prev = time.monotonic()

            # 시간 상한
            timeout  = min(10.0, max(5.0, abs(angle_rad)/max(1e-3, max_w)*2.0))
            deadline = time.monotonic() + timeout

            # ⬇️ 명령 적분 세이프티 캡
            cmd_angle = 0.0

            self.get_logger().info(f"🌀 Rotating {math.degrees(angle_rad):.2f}° (PD)")

            while rclpy.ok():
                err = self.normalize_angle(target_yaw - float(self.yaw))
                if abs(err) < done_rad:
                    break

                now = time.monotonic()
                dt  = max(now - t_prev, 1e-3)
                derr= 0.0 if prev_err is None else (err - prev_err)/dt
                w_cmd = Kp*err + Kd*derr
                prev_err, t_prev = err, now

                # 제한
                w_cmd = max(min(w_cmd, max_w), -max_w)

                # 정지각 보정
                wmag = abs(w_cmd)
                stop_angle = wmag*latency + (wmag*wmag)/(2.0*max(alpha_max, 1e-6))
                if abs(err) < stop_angle:
                    w_cmd = math.copysign(max(0.03, wmag*0.5), err)  # 0.05→0.03

                # 퍼블리시
                cmd.linear.x = 0.0
                cmd.angular.z = float(w_cmd)
                self.cmd_pub.publish(cmd)

                # ⬇️ 내가 보낸 각속도 적분으로 세이프티 캡
                cmd_angle += w_cmd * dt
                if abs(cmd_angle) > 1.3 * abs(angle_rad):
                    self.get_logger().warn("Safety stop: commanded angle >130% target")
                    res.success = False
                    self.stop()
                    return res

                if now > deadline:
                    res.success = False
                    self.stop()
                    return res

                time.sleep(0.02)

            self.stop()
            res.success = True
            return res

        except Exception:
            self.stop()
            res.success = False
            return res



def main() -> None:
    rclpy.init(args=None)
    node = SimpleMotion()

    # Multi-thread executor → 서비스 + action 동시에 처리 가능
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
