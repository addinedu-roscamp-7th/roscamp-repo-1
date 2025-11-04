import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Bool
from shopee_interfaces.msg import ArucoPose, PickeeMobileArrival
from rclpy.executors import MultiThreadedExecutor

# Pickee 전용 이동 함수 (직선 이동, 회전)
from pickee_mobile.module.module_go_strait import run
from pickee_mobile.module.module_rotate import rotate


class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking")   # ROS node 이름 설정

        self.cmd_vel = Twist()                              # 속도 명령 객체
        self.is_docking_active = False                      # 도킹 활성 상태
        self.search_enabled = False                         # 사전 탐색 활성 상태 (Nav2 도착 후 True)
        self.realign_once = False                           # 재정렬 1회만 수행 Flag
        self.aruco_id = 0
        self.last_x_offset = 0.0                            # 최근 x 오차값
        self.last_yaw_rad_offset = 0.0                      # 최근 yaw 오차값
        self.lost_count_during_docking = 0                  # 도킹 중 마커 유실 count
        self.lost_count_before_docking = 0                  # 도킹 전 마커 유실 count
        self.position_error_yaw_rad = 0.0                   # Nav2가 알려준 도착 시 회전 오차 (deg)
        self.pre_docking_search_angles_rad = [              # 도킹 전 탐색 회전 패턴
            math.radians(15),
            math.radians(-30),
            math.radians(45),
            math.radians(-60),
        ]
        self.limit_z = 190                                  # 도킹 거리 한계(mm)
        self.realign_yaw_scale_1 = 0.6                      # x & yaw 같은 방향일 때 scale
        self.realign_yaw_scale_2 = 0.7                      # 반대 방향일 때 scale
        self.aruco_map_positions = {
            1: {"x": 2.34, "y": 1.10, "yaw_rad": math.radians(90)},  # 중하
            2: {"x": 4.10, "y": -0.30, "yaw_rad": math.radians(180)}, # 중우
            3: {"x": 1.25, "y": 2.90, "yaw_rad": math.radians(90)},  # 우하
        }


        # 속도 publish 설정
        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel_modified", 10
        )

        # 도킹 완료 알림, False = 실패, True = 성공
        self.docking_in_progress_pub = self.create_publisher(
            Bool, "/pickee/mobile/docking_result", 10
        )

        # 도킹 완료 후 로봇의 현재 위치 업데이트 
        self.pose_update = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # Aruco marker 위치
        self.sub = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_docking_callback,
            1
        )

        # 로봇 도착 알림, 목적지 오차만 사용
        self.create_subscription(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            self.pickee_arrival_callback,
            10
        )

        self.get_logger().info("🤖 ArUco Docking FSM Started")


    # ==========================================================
    # ✅ ROS Callbacks
    # ==========================================================

    def pickee_arrival_callback(self, arrival_msg: PickeeMobileArrival):
        self.get_logger().info("📦 Arrival message received")
        self.position_error_yaw_rad = arrival_msg.position_error.theta
        self.search_enabled = True          # ✅ 도착 이벤트가 와야만 사전탐색 허용
        self.lost_count_before_docking = 0  # (선택) 카운터 리셋


    def aruco_docking_callback(self, msg: ArucoPose):
        x, z, yaw_deg = msg.x, msg.z, msg.pitch
        self.aruco_id = msg.aruco_id
        self.get_logger().info(f"✅ x = {x}, z = {z}, yaw_deg = {yaw_deg}°")
        yaw_rad = math.radians(yaw_deg)
        # If ArUco detected → start docking
        if z != 0.0 or x != 0.0 or yaw_rad != 0.0:
            self.is_docking_active = True

        # If Aruco not detected before docking start → search
        elif (not self.is_docking_active) and self.search_enabled:
            self.realign_before_docking()

        # If docking in progress → process movements
        if self.is_docking_active:
            self.set_docking_vel(x, z, yaw_rad)
            self.cmd_pub.publish(self.cmd_vel)
            self.last_x_offset, self.last_yaw_rad_offset = x, yaw_rad

    # ==========================================================
    # ✅ Docking Logic Functions
    # ==========================================================

    def set_docking_vel(self, x, z, yaw_rad):
        # ---------------------------------------
        # 좌우 오차, 각도오차 기반 회전, 이거 쓰고싶은데, 숫자만 바꾸면 잘 갈것 같기도?
        # ---------------------------------------
        # if abs(x) > 5 or abs(yaw) > 5:
        #     self.get_logger().info(" Adjust angle")
        #     self.lost_count_during_docking = 0

        #     scale_yaw = max(min((abs(yaw) / 20) * 0.1, 0.08), 0.0)
        #     scale_x = max(min((abs(x) / 500) * 0.1, 0.03), 0.0)
        #     scale = scale_yaw + scale_x if (x < 0) ^ (yaw < 0) else scale_yaw - scale_x
        #     self.cmd_vel.angular.z = scale if x < 0 else -scale

        # else:
        #     # scale_yaw = max(min((abs(yaw) / 20) * 0.1, 0.08), 0.0)
        #     # self.cmd_vel.angular.z = scale_yaw if scale_yaw < 0 else -scale_yaw
        #     self.cmd_vel.angular.z = 0

        # x 오차 기반 회전
        if abs(x) > 5:
            self.lost_count_during_docking = 0
            scale_yaw_rad = max(min((abs(x) / 20) * 0.1, 0.15), 0.03)

            # x가 음수면 좌회전(+), 양수면 우회전(-)
            self.cmd_vel.angular.z = scale_yaw_rad if x < 0 else -scale_yaw_rad
        else:
            self.cmd_vel.angular.z = 0.0

        # 전방 거리 기반 전진
        if z > self.limit_z:
            self.get_logger().info(" Moving forward")
            self.lost_count_during_docking = 0
            scale_z = max(min((z - self.limit_z) / 1000, 0.2), 0.05)
            self.cmd_vel.linear.x = scale_z

        # ---------------------------------------
        # 도킹중 마커 유실
        # ---------------------------------------
        elif z == 0.0 and x == 0.0 and yaw_rad == 0.0:
            self.lost_count_during_docking += 1
            self.realign_during_docking()
            return


        # ---------------------------------------
        # 거리 = 가까움, 각도 = 틀어짐
        # ---------------------------------------
        elif z <= 190 and abs(yaw_rad) > math.radians(5):
            self.get_logger().info("↩️ Final angle adjust")
            self.realign_during_docking()

        # ---------------------------------------
        # 도킹 성공
        # ---------------------------------------
        else:
            self.get_logger().info("✅ Docking success!")
            run(self, 0.11)  # 최종 도킹 동작
            time.sleep(1)
            # self.on_docking_complete(self.aruco_id) # 헌재 위치를 아르코 위치로 업데이트
            self.publish_stop()
            self.docking_in_progress_pub.publish(Bool(data=True))
            return

    # ---------------------------
    # 도킹 전 재정렬
    # ---------------------------
    def realign_before_docking(self):
        # ✅ 첫 번째 탐색 : Nav2로 도착 후 position_error 기반 정렬
        if self.lost_count_before_docking == 0:
            self.get_logger().info(
                f"🔍 [Pre-Docking Scan #1] Using position error yaw_deg: {math.degrees(self.position_error_yaw_rad):.2f}°"
            )
            rotate(self, -self.position_error_yaw_rad)
            time.sleep(0.5)
            run(self, -0.1)
            time.sleep(0.5)
            self.lost_count_before_docking += 1

        # ✅ 두 번째~다섯 번째 탐색 : 지정 각도 순차 회전
        elif self.lost_count_before_docking <= 4:
            idx = self.lost_count_before_docking - 1
            angle = self.pre_docking_search_angles[idx]
            self.get_logger().info(
                f"🔁 [Pre-Docking Scan #{self.lost_count_before_docking + 1}] "
                f"Rotate {angle:+.2f}° (Search pattern)"
            )
            rotate(self, angle)
            time.sleep(0.5)
            self.lost_count_before_docking += 1

        # ❌ 탐색 실패
        else:
            self.get_logger().warn(
                "❌ ArUco not found after multiple orientation attempts. Cancelling docking."
            )
            self.docking_in_progress_pub.publish(Bool(data=False))
            self.publish_stop()


    # ---------------------------
    # 도킹 중 재정렬
    # ---------------------------
    def realign_during_docking(self):
        self.get_logger().info(f"⚠️ Marker lost while docking (count={self.lost_count_during_docking})")

        # === 0) 이전에 한번 정렬 실행했다면 약한 보정만 실행 ===
        if self.realign_once:
            self.get_logger().info("🔂 Already realigned once → small corrective rotate")
            rotate(self, self.old_yaw_rad_diff / 2.0)
            self.realign_once = False
            return

        # === 1) 첫 2회 → 정교한 재정렬 ===
        if self.lost_count_during_docking <= 2:
            self.get_logger().info("🔄 Performing precision realign")

            self.realign_once = True
            scale = 0.7 if (self.last_yaw_rad_offset * self.last_x_offset) > 0 else 0.9
            self.old_yaw_rad_diff = float(self.last_yaw_rad_offset) * scale

            yaw_rad_adjust = 5.0 if self.last_yaw_rad_offset > 0 else -5.0
            yaw_rad_adjusted = float(self.last_yaw_rad_offset) + yaw_rad_adjust

            # 회전 → 뒤로 → 회전
            rotate(self, -(yaw_rad_adjusted + self.old_yaw_rad_diff))
            self.get_logger().info(f"🔁 rotate = {math.degrees(-(yaw_rad_adjusted + self.old_yaw_rad_diff))}°")

            time.sleep(0.5)
            run(self, -0.1)
            time.sleep(0.5)
            rotate(self, self.old_yaw_rad_diff + yaw_rad_adjust)
            time.sleep(1)
            return

        # === 2) 3~6회 → 사전탐색 패턴 ===
        if self.lost_count_during_docking <= 6:
            idx = self.lost_count_during_docking - 3
            angle = self.pre_docking_search_angles[idx]

            self.get_logger().info(
                f"🔁 During-docking scan #{self.lost_count_during_docking}: rotate {math.degrees(angle):+.2f}°"
            )
            rotate(self, angle)
            time.sleep(0.5)
            return

        # === 3) 그 외 → 실패 처리 ===
        self.get_logger().warn("❌ Marker lost too long, cancel docking")
        self.docking_in_progress_pub.publish(Bool(data=False))
        self.publish_stop()
        self.reset_docking_state()

    # ArUco 마커 도킹 이후 로봇의 현재 위치 업데이트
    def set_robot_pose(self, node, x, y, yaw_rad):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"

        # 좌표 설정
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y

        # yaw -> quaternion
        msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        msg.pose.covariance = [0.0] * 36  # covariance 기본

        self.pose_update.publish(msg)######################3
        node.get_logger().info(f"✅ Robot pose reset to map: ({x}, {y}, yaw_deg={math.degrees(yaw_rad)})°")

    def on_docking_complete(self, aruco_id):
        if aruco_id not in self.aruco_map_positions:
            self.get_logger().warn(f"Aruco ID {aruco_id} not registered!")
            return

        pos = self.aruco_map_positions[aruco_id]
        self.set_robot_pose(self, pos["x"], pos["y"], pos["yaw_rad"])




    # ---------------------------
    # 정지
    # ---------------------------
    def publish_stop(self):
        self.get_logger().info('Stop!!!')
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.linear.y = 0.0
        self.cmd_vel.angular.z = 0.0

        self.cmd_pub.publish(Twist())
        self.reset_docking_state()

    # ---------------------------
    # count, 상태 결정 변수들 reset
    # ---------------------------
    def reset_docking_state(self):
        self.get_logger().info("🔄 Reset docking state")

        self.lost_count_during_docking = 0
        self.lost_count_before_docking = 0
        self.realign_once = False
        self.is_docking_active = False
        self.search_enabled = False

        self.aruco_id = 0
        self.last_x_offset = 0.0
        self.last_yaw_rad_offset = 0.0




# ==========================================================
# ✅ Main
# ==========================================================

def main(args = None):
    rclpy.init(args=args)
    node = ArucoDocking()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
