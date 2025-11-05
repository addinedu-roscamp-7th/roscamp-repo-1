# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Callable, Optional, Tuple

from geometry_msgs.msg import Twist


class StraightRotateMixin:

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))


    def move_straight(
        self,
        distance: float,
        *,
        frame: str = "odom",           # 'odom' | 'map'
        Kp: float = 1.4,
        Kd: float = 0.2,
        max_speed: float = 0.2,         # [m/s]
        a_max: float = 0.8,             # [m/s^2] decel capability
        latency: float = 0.12,          # [s] total system latency
        Kyaw: float = 1.0,              # heading hold P
        Dyaw: float = 0.05,             # heading hold D
        done_tol: float = 0.005,        # [m] finish tolerance
        min_speed: float = 0.05,        # [m/s] to overcome stiction
        sample_dt: float = 0.02,        # [s] publish period
        use_projection: bool = True,
    ) -> None:
        """Drive straight for a signed distance with heading hold.

        Positive distance moves forward, negative backward. Uses distance PD with
        braking based on latency/a_max. Optionally holds initial heading via yaw PD.
        """
        # Guard: required state present
        while getattr(self, "x", None) is None or getattr(self, "yaw", None) is None or (
            frame == "map" and getattr(self, "mx", None) is None
        ):
            # Busy-wait until odom (/map) available
            try:
                import rclpy  # type: ignore
                rclpy.spin_once(self)
            except Exception:
                pass

        # Frame selector
        if frame == "map":
            sx, sy = float(self.mx), float(self.my)
            get_xy: Callable[[], Tuple[float, float]] = lambda: (float(self.mx), float(self.my))
        else:
            sx, sy = float(self.x), float(self.y)
            get_xy = lambda: (float(self.x), float(self.y))

        theta0 = float(self.yaw)
        dir_sign = 1.0 if distance >= 0.0 else -1.0
        target = abs(distance)

        cmd = Twist()
        prev_err: Optional[float] = None
        prev_yaw_err: Optional[float] = None
        t_prev = time.monotonic()

        while True:
            # spin callbacks, then read pose
            try:
                import rclpy  # type: ignore
                rclpy.spin_once(self)
            except Exception:
                pass

            cx, cy = get_xy()

            if use_projection:
                # signed progress along initial heading
                dx, dy = (cx - sx), (cy - sy)
                moved_signed = dx * math.cos(theta0) + dy * math.sin(theta0)
                moved_toward = dir_sign * moved_signed
            else:
                # straight-line Euclidean (unsigned) — not recommended for reverse
                dx, dy = (cx - sx), (cy - sy)
                moved_toward = abs(math.hypot(dx, dy))

            error = target - moved_toward
            if error < done_tol:
                break

            now = time.monotonic()
            dt = max(now - t_prev, 1e-3)

            # Distance PD
            d_err = 0.0 if prev_err is None else (error - prev_err) / dt
            base_v = Kp * error + Kd * d_err
            prev_err, t_prev = error, now

            # Clamp and keep direction
            v_cmd = max(min(base_v, max_speed), -max_speed)
            v_cmd *= dir_sign

            # Braking distance estimate
            vmag = abs(v_cmd)
            d_stop = vmag * latency + (vmag * vmag) / (2.0 * max(a_max, 1e-6))
            if error < d_stop:
                v_cmd = dir_sign * max(min_speed, vmag * 0.5)

            # Minimum speed to overcome stiction (only if not almost done)
            if error > 0.03 and abs(v_cmd) < min_speed:
                v_cmd = dir_sign * min_speed

            # Heading hold around initial heading
            yaw_err = self._normalize_angle(theta0 - float(self.yaw))
            dyaw = 0.0 if prev_yaw_err is None else (yaw_err - prev_yaw_err) / dt
            prev_yaw_err = yaw_err
            w_cmd = Kyaw * yaw_err + Dyaw * dyaw

            # Publish
            cmd.linear.x = float(v_cmd)
            cmd.angular.z = float(w_cmd)
            self.cmd_pub.publish(cmd)
            time.sleep(sample_dt)

        # Stop
        self._stop_twist()

    # -------------------------------
    # Shortest-path PD rotate with braking
    # -------------------------------
    def rotate_pd(
        self,
        angle_rad: float,
        *,
        Kp: float = 2.2,
        Kd: float = 0.18,
        max_w: float = 0.3,           # [rad/s]
        alpha_max: float = 3.0,       # [rad/s^2]
        latency: float = 0.10,        # [s]
        done_deg: float = 1.0,        # [deg]
        sample_dt: float = 0.02,
        angle_scale: float = 1.0,     # e.g., 0.95 for small overshoot compensation
    ) -> None:
        """Rotate by `angle_rad` along the shortest path with PD and braking.
        Positive is CCW, negative CW.
        """
        # Ensure yaw available
        while getattr(self, "yaw", None) is None:
            try:
                import rclpy  # type: ignore
                rclpy.spin_once(self)
            except Exception:
                pass

        delta = float(angle_rad) * float(angle_scale)
        target_yaw = self._normalize_angle(float(self.yaw) + delta)
        done_rad = math.radians(done_deg)

        cmd = Twist()
        prev_err: Optional[float] = None
        t_prev = time.monotonic()

        while True:
            try:
                import rclpy  # type: ignore
                rclpy.spin_once(self)
            except Exception:
                pass

            err = self._normalize_angle(target_yaw - float(self.yaw))
            if abs(err) < done_rad:
                break

            now = time.monotonic()
            dt = max(now - t_prev, 1e-3)
            derr = 0.0 if prev_err is None else (err - prev_err) / dt
            w_cmd = Kp * err + Kd * derr
            prev_err, t_prev = err, now

            # Clamp
            w_cmd = max(min(w_cmd, max_w), -max_w)

            # Braking (expected stop angle)
            wmag = abs(w_cmd)
            stop_angle = wmag * latency + (wmag * wmag) / (2.0 * max(alpha_max, 1e-6))
            if abs(err) < stop_angle:
                w_cmd = math.copysign(max(0.05, wmag * 0.5), err)

            cmd.linear.x = 0.0
            cmd.angular.z = float(w_cmd)
            self.cmd_pub.publish(cmd)
            time.sleep(sample_dt)

        # Stop
        self._stop_twist()

    # -------------------------------
    # Publish zero twist
    # -------------------------------
    def _stop_twist(self) -> None:
        t = Twist()
        self.cmd_pub.publish(t)
        # small settle delay to ensure last command is delivered
        time.sleep(0.05)
