import rclpy
import cv2
import math
from rclpy.node import Node
from pickee_mobile.module.module_aruco_detect import ArucoPoseEstimator 
from geometry_msgs.msg import Pose2D
from shopee_interfaces.msg import ArucoPose, PickeeMobileArrival
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from collections import Counter
from std_msgs.msg import Bool
import threading, sys, termios, tty


class ArucoReaderNode(Node):
    def __init__(self):
        super().__init__('aruco_reader')  # ROS2 Node 이름
        self.get_logger().info("📷 ArUco Reader Node Started")

        self.docking_in_progress = False         # 도킹 활성 상태 flag
        self.target_id = 2                       # 탐지할 ArUco ID 설정 (여기선 ID = 2)

        # ✅ 카메라 + ArUco Pose Detector 초기화
        self.estimator = ArucoPoseEstimator(
            camera_id=2,                          # 사용 카메라 인덱스
            marker_length=50,                    # 마커 크기(mm)
            calibration_file="camera_calibration.pkl"  # 카메라 보정 파일
        )

        # ✅ ArUco 좌표 publish 하는 publisher
        self.pose_publisher = self.create_publisher(
            ArucoPose, '/pickee/mobile/aruco_pose', 10
        )

        # ✅ 도킹 진행상태 subscribe (도킹이 시작/중지 되면 콜백 실행)
        self.create_subscription(
            Bool,
            '/pickee/mobile/docking_in_progress',
            self.docking_status_callback,
            10
        )


        # 로봇이 목적지에 도착한거 감지
        self.create_subscription(PickeeMobileArrival,
                                '/pickee/mobile/arrival',
                                self.detect_aruco_callback,
                                10)

        # ✅ 키보드 스레드 시작 (z: 시작, x: 정지)
        thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        thread.start()
        self.get_logger().info("⌨️ Press 'z' to start ArUco detection, 'x' to stop")

    # --------------------------------------------------------------------
    # ✅ ROS Callbacks
    # --------------------------------------------------------------------
    def docking_status_callback(self, msg: Bool):
        """도킹 상태 갱신 (외부에서 True/False 들어올 때)"""
        self.docking_in_progress = msg.data
        if not self.docking_in_progress:
            self.get_logger().info("🛑 Docking process ended. Stopping ArUco scan.")

    def detect_aruco_callback(self, msg: PickeeMobileArrival):
        """🚦 Nav2 도착 콜백 (사용 안함 - 주석 처리됨)"""
        self.docking_in_progress = True
        self.get_logger().info("🚦 Arrival detected! Starting ArUco scan...")
        self.read_marker()

    # --------------------------------------------------------------------
    # ✅ ArUco 마커 읽기 루프
    # --------------------------------------------------------------------
    def read_marker(self):
        """ArUco 데이터를 계속 읽고 publish"""
        self.get_logger().info(f"{self.docking_in_progress}")

        while self.docking_in_progress:
            # 측정값 저장 공간 (다회 샘플 -> noise 제거)
            values = {"id": [], "x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []}

            for i in range(5):  # 5회 측정
                if not self.docking_in_progress:
                    break

                print(f"📸 Reading marker attempt {i+1}/5")

                ret, frame = self.estimator.cap.read()
                if not ret:
                    self.get_logger().warning("❌ 프레임을 읽을 수 없습니다.")
                    return

                frame_out, markers = self.estimator.process_frame(frame)

                if markers:
                    # ✅ 원하는 ID만 필터링
                    markers = [m for m in markers if m["id"] == self.target_id]
                    if not markers:
                        continue

                    m = markers[0]  # 해당 ID의 첫번째 마커만 사용

                    # ✅ 수집 (노이즈 대비)
                    for k in values:
                        values[k].append(m[k])

                    self.get_logger().info(
                        f"🟢 {i+1}/5 | ID={m['id']} | "
                        f"x={m['x']:.1f}, y={m['y']:.1f}, z={m['z']:.1f} | "
                        f"roll={m['roll']:.1f}, pitch={m['pitch']:.1f}, yaw={m['yaw']:.1f}"
                    )
                else:
                    self.get_logger().info(f"⚠️ {i+1}/5 | Marker not found")

            if not self.docking_in_progress:
                break

            if len(values["id"]) == 0:
                # ✅ 특정 ID 감지 실패 시 0 publish (로봇 fallback 가능)
                self.get_logger().error("❌ ArUco marker not found 5 times.")
                pose = ArucoPose()
                pose.aruco_id = 0
                pose.x = pose.y = pose.z = 0.0
                pose.roll = pose.pitch = pose.yaw = 0.0
                self.pose_publisher.publish(pose)
                continue

            # ✅ 최빈값(Mode)과 중앙값(Median) 계산 -> 안정화
            aruco_id = Counter(values["id"]).most_common(1)[0][0]
            median = {k: float(np.median(v)) for k, v in values.items() if k != "id"}

            self.get_logger().info(
                f"✅ Filter 완료 | ID={aruco_id} | x={median['x']:.1f}, y={median['y']:.1f}, z={median['z']:.1f} | yaw={median['yaw']:.1f}"
            )

            # ✅ ROS 메시지 Publish
            pose = ArucoPose()
            pose.aruco_id = aruco_id
            pose.x = median["x"]; pose.y = median["y"]; pose.z = median["z"]
            pose.roll = median["roll"]; pose.pitch = median["pitch"]; pose.yaw = median["yaw"]
            self.pose_publisher.publish(pose)

    # --------------------------------------------------------------------
    # ✅ 키보드 입력 스레드
    # --------------------------------------------------------------------
    def keyboard_listener(self):
        """콘솔 입력으로 Z/X 제어"""
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while True:
                key = sys.stdin.read(1)
                if key.lower() == 'z':  # 시작
                    self.get_logger().info("✅ Z pressed → Start ArUco scan")
                    self.docking_in_progress = True
                    self.read_marker()
                elif key.lower() == 'x':  # 정지
                    self.get_logger().info("🛑 X pressed → Stop ArUco scan")
                    self.docking_in_progress = False
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# --------------------------------------------------------------------
# ✅ Main
# --------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArucoReaderNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.estimator.cap.release()  # 카메라 해제
        cv2.destroyAllWindows()       # OpenCV 창 닫기
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
