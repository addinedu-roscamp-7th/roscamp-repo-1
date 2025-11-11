import math
import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Vector3
from std_msgs.msg import Float32, Bool
from shopee_interfaces.msg import ArucoPose, PickeeMobileArrival
from nav_msgs.msg import Odometry

# Pickee 전용 이동 함수 (직선 이동, 회전)
from pickee_mobile.module.module_go_straight import run
from pickee_mobile.module.module_rotate import rotate

#⚙️ Pickee 전용 odom 받아서 정밀 이동 class
from pickee_mobile.module.module_go_straight_odom import GoStraight #⚙️ odom읽으면서 제어
from pickee_mobile.module.module_rotate_odom import Rotate #⚙️ odom읽으면서 제어

# 상태
# Idle(대기) 
# -> Before_docking (목적지 도착 신호를 받음, 마커 감지 안되면 탐색동작), Lost_before_docking (목적지 도착 했는데 마커가 안보임)
# -> Aligning_to_side (마커가 정면에 오도록 옆으로 이동) 
# -> Docking (마커가 어느정도 정면에 있고 z값 줄이면서 도킹 완료), Lost_during_docking (Docking 중에 마커 안보임)
# -> Ending (도킹 완료 신호 전송, 정지, 카운터 리셋, 상태변수 리셋)
# -> Idle(대기)

# -pi ~ pi 로 변환
def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

# 부호 판별
def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

# 마커 기준 로봇이 전진해야 하는 거리, 좌우 이동해야 하는 거리
def dist_from_xyz_pitch(x, z, pitch_rad):
    # 정면(법선) 거리
    dist_front = abs(x*math.sin(pitch_rad) + z*math.cos(pitch_rad))
    # 바닥면 좌우(부호 포함), 음수 : 카메라 기준 마커가 왼쪽에 있다.
    dist_side = x*math.cos(pitch_rad) - z*math.sin(pitch_rad)

    return dist_front, dist_side


class ArucoDocking(Node):
    def __init__(self):
        super().__init__("aruco_docking_once")   # ROS node 이름 설정

        self.cmd_vel = Twist()                              # 속도 명령 객체
        self.is_docking_active = False                      # 도킹 활성 상태
        self.search_enabled = False                         # 사전 탐색 활성 상태 (Nav2 도착 후 True)
        self.realign_once = False                           # 재정렬 1회만 수행 Flag
        self.aruco_id = 0
        self.last_x_offset = 0.0                            # 최근 x 오차값
        self.last_yaw_rad_offset = 0.0                      # 최근 yaw 오차값
        self.lost_count_during_docking = 0                  # 도킹 중 마커 유실 count
        self.lost_count_before_docking = 0                  # 도킹 전 마커 유실 count
        self.position_error_yaw_rad = 0.0                   # Nav2가 알려준 도착 시 회전 오차 (rad)
        self.old_yaw_rad_diff = 0.0
        self.DOCKING_STATE_LIST = ["Idle",                  # 상태 변경
                                    "Before_docking", 
                                    "Lost_before_docking",
                                    "Aligning_to_side",
                                    "Docking",
                                    "Lost_during_docking",
                                    "Ending"]
        self.current_state = "Idle"                         # 대기 상태
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
        self.stop_event = Event()

        # 기구 오프셋
        self.camera_offset_mm = 90.0           # 카메라가 로봇 중심보다 전방(+Z) 90 mm

        ## Publish
        # 속도 publish 설정
        self.cmd_pub = self.create_publisher(
            # Twist, "/cmd_vel_modified", 10
            Twist, "/cmd_vel", 10
        )

        # 도킹 완료 알림, False = 실패, True = 성공
        self.docking_in_progress_pub = self.create_publisher(
            Bool, "/pickee/mobile/docking_result", 10
        )

        # 도킹 완료 후 로봇의 현재 위치 업데이트 
        self.pose_update = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        ## Subscribe
        # Aruco marker 위치

        # self.cb_group = MutuallyExclusiveCallbackGroup()

        self.sub = self.create_subscription(
            ArucoPose,
            '/pickee/mobile/aruco_pose',
            self.aruco_docking_callback,
            1
            # callback_group=self.cb_group
        )

        # 로봇 도착 알림, 목적지 오차만 사용
        self.create_subscription(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            self.pickee_arrival_callback,
            10
        )

        self.rotate_node = Rotate() #⚙️
        self.go_straight_node = GoStraight() #⚙️


        self.get_logger().info("🤖 ArUco Docking FSM Started")


    # ==========================================================
    # ✅ ROS Callbacks
    # ==========================================================

    def pickee_arrival_callback(self, arrival_msg: PickeeMobileArrival):
        self.get_logger().info("📦 Arrival message received")
        if self.current_state == "Idle":

            self.current_state = "Before_docking"
            self.position_error_yaw_rad = arrival_msg.position_error.theta
            self.get_logger().info(f"🤖 Current state is {self.current_state}")


    def aruco_docking_callback(self, msg: ArucoPose):

        if self.current_state != "Idle":

            self.x, self.z, self.yaw_deg = msg.x, msg.z, msg.pitch
            self.aruco_id = msg.aruco_id
            self.yaw_rad = math.radians(self.yaw_deg)
            self.dist_front, self.dist_side = dist_from_xyz_pitch(self.x, self.z, self.yaw_rad)
            
            # 마커 감지 성공
            if self.z != 0.0 or self.x != 0.0 or self.yaw_rad != 0.0:
                self.get_logger().info(f"📦 x = {self.x}, z = {self.z}, yaw_deg = {self.yaw_deg}°")
                self.old_x, self.old_z, self.old_yaw_rad = self.x, self.z, self.yaw_rad
                self.lost_count_during_docking = 0

                # 도킹동작 전이라면 도킹 상태로
                if self.current_state == "Before_docking":

                    self.is_docking_active = True
                    self.current_state = "Aligning_to_side"
                    self.get_logger().info(f"📦 ArUco marker data received")
                    self.get_logger().info(f"🤖 Current state is {self.current_state}")

                elif self.is_docking_active:

                    if self.current_state == "Aligning_to_side":
                        
                        self.align_to_side()

                    elif self.current_state == "Docking":

                        self.docking()

            # 마커 감지 실패
            else:
                # 도킹 전 복구동작
                if self.current_state == "Before_docking":

                    self.current_state = "Lost_before_docking"
                    self.detect_marker_before_docking()
                
                # 접근 중 복구동작
                elif self.current_state == "Docking":

                    self.current_state = "Lost_during_docking"
                    self.detect_marker_during_docking()

            # If docking in progress → process movements
            # if self.is_docking_active:

            #     if self.current_state == "Aligning_to_side":

            #         self.align_x(x, z, yaw_rad)

            #     elif self.current_state == "Docking":

            #         self.set_docking_vel(x, z, yaw_rad)

    # ==========================================================
    # ✅ Docking Logic Functions
    # ==========================================================


    def detect_marker(self):
        pass

    # 마커 중심으로 이동, 마커 주시
    def align_to_side(self):
        
        if abs(self.dist_side) > 50:
            self.get_logger().info(f"✅ dist_front = {self.dist_front}, dist_side = {self.dist_side}, yaw_deg = {math.degrees(self.yaw_rad)}")

            # 마커 방향 x축에 수직이 되도록 회전
            turn_to_side_rad = sign(self.yaw_rad) * (math.radians(90) - abs(self.yaw_rad))
            # normalize_angle(turn_to_side_rad)
            self.get_logger().info(f"🔁 Rotating {math.degrees(turn_to_side_rad)}°")
            self.rotate_node.rotate(turn_to_side_rad)
            time.sleep(1.0)

            # 해당 축까지 전진
            self.get_logger().info(f"🚗 Going straight to ArUco axis {self.dist_side}mm")
            self.go_straight_node.go_straight(abs(self.dist_side/1000))
            time.sleep(1.0)

            # 마커 바라보기 회전
            self.get_logger().info(f"🔁 Rotating to ArUco Marker°")
            turn_to_front_rad = -sign(self.yaw_rad) * math.radians(90)
            # normalize_angle(turn_to_front_rad)
            self.rotate_node.rotate(turn_to_front_rad)
            time.sleep(1.0)
        
        else:

            self.get_logger().info(f"🤖 PickeeMobile is allready aligned to side. dist_side = {self.dist_side}")
            self.get_logger().info(f"🤖 Start docking")

        self.current_state = "Docking"

    def docking(self):
        now = time.time()
        # self.get_logger().info(f"✅ Aligned to x!!! Start Docking")
        
        self.get_logger().info(f"✅ dist_front = {self.dist_front}, dist_side = {self.dist_side}, yaw_deg = {math.degrees(self.yaw_rad)}")

        # ----- P 제어 계산 -----
        # dist_side, yaw_rad = self.cmd_vel.angular.z
        # +, + = - 작은
        # +, - = + dist_side 비례
        # -, + = - dist_side 비례
        # -, - = + 작은

        # 회전 각도 조절
        # if abs(self.dist_side) > 10:

        #     self.get_logger().info(f"🔁 111")

        #     scale_yaw = max(min((abs(self.dist_side)) / 1000, 0.1), 0.05)
        #     if abs(self.yaw_rad) > math.radians(10) and self.dist_side * self.yaw_rad < 0:
        #         scale_yaw *= 0.1
        #     self.cmd_vel.angular.z = scale_yaw if self.dist_side < 0 else -scale_yaw

        # else: # abs(self.dist_side) <= 5:

        #     self.get_logger().info(f"🔁 222")

        #     scale_yaw = max(min((abs(self.yaw_rad)) / 100, 0.5), 0.1)

        #     self.cmd_vel.angular.z = scale_yaw if self.dist_side < 0 else -scale_yaw


        if abs(self.dist_side) > 10:

            self.get_logger().info(f"🔁 111")

            goal_yaw_rad = math.radians(max(min((abs(self.dist_side)) / 5, 20), 0.0))
            goal_yaw_rad = goal_yaw_rad if self.dist_side < 0 else -goal_yaw_rad
            self.set_yaw(goal_yaw_rad)

        else: # abs(self.dist_side) <= 10:

            self.get_logger().info(f"🔁 222")

            goal_yaw_rad = math.radians(max(min((abs(self.dist_side)) / 14, 10), 0.0))
            goal_yaw_rad = goal_yaw_rad if self.dist_side < 0 else -goal_yaw_rad
            self.set_yaw(goal_yaw_rad)


        # 전진 속도 조절
        if self.dist_front > self.limit_z:

            self.get_logger().info(f"🚗 111")

            scale_z = max(min((self.dist_front - self.limit_z) / 1000, 0.07), 0.03)
            self.cmd_vel.linear.x = scale_z
        
        elif abs(self.yaw_rad) > math.radians(8):# or abs(self.dist_side) > 25:

            self.get_logger().info(f"🚗 222")

            self.detect_marker_during_docking()
        
        else:
            self.get_logger().info(f'✅ Last Docking Process')
            self.publish_stop()
            run(self, 0.115)
            self.get_logger().info(f"✅ Docking process completed!!! Ending Process")
            self.publish_stop()
            self.reset_docking_state()
            self.docking_in_progress_pub.publish(Bool(data=True)) # 도킹 작업 성공 알림

        self.cmd_pub.publish(self.cmd_vel)

    def set_yaw(self, goal_yaw_rad):
        if goal_yaw_rad > self.old_yaw_rad:
            self.cmd_vel.angular.z = 0.06
        
        else:
            self.cmd_vel.angular.z = -0.06


    def detect_marker_before_docking(self):
        self.get_logger().info(f"⚠️ ArUco marker lost before docking")
        self.get_logger().info(f"⚠️ Current state is {self.current_state}")
        
        if self.lost_count_before_docking == 0:
            self.get_logger().info(
                f"🔍 [Pre-Docking Scan #1] Using position error yaw_deg: {math.degrees(self.position_error_yaw_rad):.2f}°"
            )
            rotate(self, -self.position_error_yaw_rad)
            # self.rotate_node.rotate(-self.position_error_yaw_rad) #⚙️
            time.sleep(0.5)
            run(self, -0.1)
            # self.go_straight_node.go_straight(-0.1) #⚙️
            time.sleep(0.5)
            self.lost_count_before_docking += 1

        # ✅ 두 번째~다섯 번째 탐색 : 지정 각도 순차 회전
        elif self.lost_count_before_docking <= 4:
            idx = self.lost_count_before_docking - 1
            angle = self.pre_docking_search_angles_rad[idx]
            self.get_logger().info(
                f"🔁 [Pre-Docking Scan #{self.lost_count_before_docking + 1}] "
                f"Rotate {math.degrees(angle):+.2f}° (Search pattern)"
            )
            rotate(self, angle)
            # self.rotate_node.rotate(angle) #⚙️
            time.sleep(0.5)
            self.lost_count_before_docking += 1

        # ❌ 탐색 실패
        else:
            self.get_logger().warn(
                "❌ ArUco not found after multiple orientation attempts. Cancelling docking."
            )
            self.docking_in_progress_pub.publish(Bool(data=False))
            self.publish_stop()

        self.current_state = "Before_docking"

    def detect_marker_during_docking(self):
        self.get_logger().info(f"⚠️ ArUco marker lost during docking")
        self.get_logger().info(f"⚠️ Current state is {self.current_state}")
        # self.get_logger().info(f"⚠️ Marker lost while docking (count={self.lost_count_during_docking})")
        
        self.publish_stop()
        self.lost_count_during_docking += 1
        # === 0) 이전에 한번 정렬 실행했다면 약한 보정만 실행 ===
        # if self.realign_once:
        #     self.get_logger().info("🔂 Already realigned once → small corrective rotate")
        #     rotate(self, self.old_yaw_rad_diff / 2.0)
        #     self.realign_once = False
        #     return

        # === 1) 첫 2회 → 정교한 재정렬 ===
        if self.lost_count_during_docking <= 2:
            self.get_logger().info("🔄 Performing precision realign")

            self.realign_once = True
            scale = 0.7 if (self.old_yaw_rad * self.old_x) > 0 else 0.9
            self.old_yaw_rad_diff = float(self.old_yaw_rad) * scale

            # yaw_rad_adjust = math.radians(5.0) if self.last_yaw_rad_offset > 0 else math.radians(-5.0)
            yaw_rad_adjusted = float(self.old_yaw_rad)

            # 회전 → 뒤로 → 회전
            rotate(self, -(yaw_rad_adjusted + self.old_yaw_rad_diff))
            # self.rotate_node.rotate(-(yaw_rad_adjusted + self.old_yaw_rad_diff)) #⚙️
            self.get_logger().info(f"🔁 rotate = {math.degrees(-(yaw_rad_adjusted + self.old_yaw_rad_diff))}°")

            time.sleep(0.5)
            run(self, -0.1)
            # self.go_straight_node.go_straight(-0.1) #⚙️
            time.sleep(0.5)
            rotate(self, self.old_yaw_rad_diff)
            # self.rotate_node.rotate(self.old_yaw_rad_diff + yaw_rad_adjust) #⚙️
            time.sleep(1)

        # === 2) 3~6회 → 사전탐색 패턴 ===
        elif self.lost_count_during_docking <= 6:
            idx = self.lost_count_during_docking - 3
            angle = self.pre_docking_search_angles_rad[idx]

            self.get_logger().info(
                f"🔁 During-docking scan #{self.lost_count_during_docking}: rotate {math.degrees(angle):+.2f}°"
            )
            rotate(self, angle)
            # self.rotate_node.rotate(angle) #⚙️
            time.sleep(0.5)

        else:
            # === 3) 그 외 → 실패 처리 ===
            self.get_logger().warn("❌ Marker lost too long, cancel docking")
            self.docking_in_progress_pub.publish(Bool(data=False))
            self.publish_stop()
            self.reset_docking_state()
        
        self.current_state = "Docking"

    # ---------------------------
    # 정지
    # ---------------------------
    def publish_stop(self):
        self.get_logger().info('Stop!!!')
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.linear.y = 0.0
        self.cmd_vel.angular.z = 0.0

        self.cmd_pub.publish(Twist())

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
        self.current_state = "Idle"

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
