import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from shopee_interfaces.msg import ArucoPose, PickeeMobileArrival
from rclpy.executors import MultiThreadedExecutor

# Pickee 전용 이동 함수 (직선 이동, 회전)
from pickee_mobile.module.module_go_strait import run
from pickee_mobile.module.module_rotate import rotate


class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking")   # ROS node 이름 설정

        self.cmd_vel = Twist()                              # 속도 명령 객체 생성
        self.last_x_offset = 0.0                            # 마지막 x 값 (재탐색 시 사용)
        self.last_yaw_offset = 0.0                          # 마지막 yaw 값 (재탐색 시 사용)
        self.lost_count_during_docking = 0                  # 도킹 중 마커 재탐색 동작 횟수
        self.aruco_lost_count_before_docking = 0            # 도킹 전 마커 재탐색 동작 횟수
        self.pre_docking_search_angles = [15, -30, 45, -60] # 도킹 전 마커 재탐색 회전 동작 순서
        self.position_error_yaw = 0                         # 목적지 도착 후 로봇의 회전 오차
        self.is_docking_active = False                      # 도킹 시작 여부 (처음 감지 안되는 문제 방지)
        self.realign_yaw_scale_1 = 0.4                      # 재탐색 회전 보정 scale 1
        self.realign_yaw_scale_2 = 0.5                      # 재탐색 회전 보정 scale 2
        self.realign_yaw_scale = 1.4                        # 재탐색 회전 보정
        self.realign_once = False                           # 재탐색 한 번만 수행하도록 flag
        self.limit_z = 190
        self.search_enabled = False

        # 속도 publish 설정
        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel_modified", 10
        )

        # 도킹 완료 알림, False = 실패, True = 성공
        self.docking_in_progress_pub = self.create_publisher(
            Bool, "/pickee/mobile/docking_result", 10
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
        self.position_error_yaw = math.degrees(arrival_msg.position_error.theta)
        self.search_enabled = True          # ✅ 도착 이벤트가 와야만 사전탐색 허용
        self.aruco_lost_count_before_docking = 0  # (선택) 카운터 리셋


    def aruco_docking_callback(self, msg: ArucoPose):
        x, z, yaw = msg.x, msg.z, msg.pitch

        # If ArUco detected → start docking
        if z != 0.0 or x != 0.0 or yaw != 0.0:
            self.is_docking_active = True

        # If Aruco not detected before docking start → search
        elif (not self.is_docking_active) and self.search_enabled:
            self.realign_before_docking()

        # If docking in progress → process movements
        if self.is_docking_active:
            self.set_docking_vel(x, z, yaw)
            self.cmd_pub.publish(self.cmd_vel)
            self.last_x_offset, self.last_yaw_offset = x, yaw

    # ==========================================================
    # ✅ Docking Logic Functions
    # ==========================================================

    def set_docking_vel(self, x, z, yaw):
        # ---------------------------------------
        # 좌우 오차, 각도오차 기반 회전
        # ---------------------------------------
        # if abs(x) > 5 or abs(yaw) > 5:
        #     self.get_logger().info(" Adjust angle")
        #     self.lost_count_during_docking = 0

        #     scale_yaw = max(min((abs(yaw) / 20) * 0.1, 0.08), 0.0)
        #     scale_x = max(min((abs(x) / 60) * 0.1, 0.03), 0.0)
        #     scale = scale_yaw + scale_x if (x < 0) ^ (yaw < 0) else scale_yaw - scale_x
        #     self.cmd_vel.angular.z = scale if x < 0 else -scale

        # else:
        #     # scale_yaw = max(min((abs(yaw) / 20) * 0.1, 0.08), 0.0)
        #     # self.cmd_vel.angular.z = scale_yaw if scale_yaw < 0 else -scale_yaw
        #     self.cmd_vel.angular.z = 0

        # 이게 가장 잘 감
        if abs(x) > 5:
            self.aruco_lost_count = 0
            scale_yaw = max(min((abs(x) / 20) * 0.1, 0.1), 0.0)
            if x < 0 and yaw > 0:
                self.cmd_vel.angular.z = scale_yaw
            elif x > 0 and yaw > 0:
                self.cmd_vel.angular.z = -scale_yaw
            elif x > 0 and yaw < 0:
                self.cmd_vel.angular.z = -scale_yaw
            elif x < 0 and yaw < 0:
                self.cmd_vel.angular.z = scale_yaw
        else:
            self.cmd_vel.angular.z = 0.0


        # ---------------------------------------
        # 전방 거리 기반 전진
        # ---------------------------------------
        if z > self.limit_z:
            self.get_logger().info(" Moving forward")
            self.lost_count_during_docking = 0
            scale_z = max(min((z - self.limit_z) / 1000, 0.2), 0.05)
            self.cmd_vel.linear.x = scale_z

        # ---------------------------------------
        # Lost marker during docking → recovery
        # ---------------------------------------
        elif z == 0.0 and x == 0.0 and yaw == 0.0:
            self.get_logger().info("⚠️ Marker lost while docking")
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.publish_stop()
            self.lost_count_during_docking += 1

            if self.realign_once:
                rotate(self, self.old_yaw_diff / 2.0)

            elif self.lost_count_during_docking <= 2:
                self.realign_during_docking()

            elif self.lost_count_during_docking <= 6:
                idx = self.lost_count_during_docking - 3
                angle = self.pre_docking_search_angles[idx]
                self.get_logger().info(
                    f"🔁 [During-Docking Scan #{self.lost_count_during_docking + 1}] "
                    f"Rotate {angle:+.2f}° (Search pattern)"
                )
                rotate(self, angle)
                time.sleep(0.5)
            
            else:
                self.get_logger().warn(
                    "⚠️ ArUco not found after multiple orientation attempts. Cancelling docking."
                )
                self.docking_in_progress_pub.publish(Bool(data=False))
                self.publish_stop()

                self.realign_once = False
                return

        # ---------------------------------------
        # Close but angle wrong → realign
        # ---------------------------------------
        elif z <= 190 and abs(yaw) > 5:
            self.get_logger().info("↩️ Final angle adjust")
            self.realign_during_docking()

        # ---------------------------------------
        # Docking success
        # ---------------------------------------
        else:
            self.get_logger().info("✅ Docking success!")
            run(self, 0.09)  # final push
            time.sleep(2)
            self.publish_stop()
            self.is_docking_active = False
            self.search_enabled = False
            self.docking_in_progress_pub.publish(Bool(data=True))
            return

    # ---------------------------
    # 💡 Pre-Docking Realignment
    # ---------------------------
    def realign_before_docking(self):
        # ✅ 첫 번째 탐색 : Nav2로 도착 후 position_error 기반 정렬
        if self.aruco_lost_count_before_docking == 0:
            self.get_logger().info(
                f"🔍 [Pre-Docking Scan #1] Using position error yaw: {self.position_error_yaw:.2f}°"
            )
            rotate(self, -self.position_error_yaw)
            time.sleep(0.5)
            self.aruco_lost_count_before_docking += 1

        # ✅ 두 번째~다섯 번째 탐색 : 지정 각도 순차 회전
        elif self.aruco_lost_count_before_docking <= 4:
            idx = self.aruco_lost_count_before_docking - 1
            angle = self.pre_docking_search_angles[idx]
            self.get_logger().info(
                f"🔁 [Pre-Docking Scan #{self.aruco_lost_count_before_docking + 1}] "
                f"Rotate {angle:+.2f}° (Search pattern)"
            )
            rotate(self, angle)
            time.sleep(0.5)
            self.aruco_lost_count_before_docking += 1

        # ❌ 탐색 실패
        else:
            self.get_logger().warn(
                "⚠️ ArUco not found after multiple orientation attempts. Cancelling docking."
            )
            self.docking_in_progress_pub.publish(Bool(data=False))
            self.publish_stop()


    # ---------------------------
    # 💡 Docking Realignment
    # ---------------------------
    def realign_during_docking(self):
        self.realign_once = True
        self.get_logger().info("🔄 Realigning during docking...")
        self.get_logger().info(f"last_x_offset = {self.last_x_offset}, last_yaw_offset = {self.last_yaw_offset}")

        # if self.last_yaw_offset > 0 and self.last_x_offset > 0:
        #     self.old_yaw_diff = self.last_yaw_offset * 0.6
        # elif self.last_yaw_offset > 0 and self.last_x_offset < 0:
        #     self.old_yaw_diff = self.last_yaw_offset * 0.7
        # elif self.last_yaw_offset < 0 and self.last_x_offset > 0:
        #     self.old_yaw_diff = self.last_yaw_offset * 0.7
        # elif self.last_yaw_offset < 0 and self.last_x_offset < 0:
        #     self.old_yaw_diff = self.last_yaw_offset * 0.6
        # else:
        #     self.old_yaw_diff = 0.0

        scale = 0.6 if (self.last_yaw_offset * self.last_x_offset) > 0 else 0.7
        self.old_yaw_diff = float(self.last_yaw_offset) * scale
        
        # if self.last_yaw_offset > 0:
        #     self.last_yaw_offset += 10
        # else:
        #     self.last_yaw_offset -= 10

        yaw_adjust = 10 if self.last_yaw_offset > 0 else -10
        yaw_adjusted = float(self.last_yaw_offset) + yaw_adjust



        rotate(self, -(yaw_adjusted + self.old_yaw_diff))

        time.sleep(0.5)
        run(self, -0.1)
        time.sleep(0.5)
        rotate(self, self.old_yaw_diff * 1.2)
        time.sleep(1)

    # ---------------------------
    # 🛑 Stop Command
    # ---------------------------
    def publish_stop(self):
        self.get_logger().info('Stop!!!')
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.linear.y = 0.0
        self.cmd_vel.angular.y = 0.0

        self.cmd_pub.publish(Twist())



# ==========================================================
# ✅ Main
# ==========================================================

def main(args = None):
    # rclpy.init()
    # node = ArucoDocking()
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()


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
