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
        super().__init__('aruco_reader')
        self.get_logger().info("📷 ArUco Reader Node Started")

        self.docking_in_progress = False

        

        # ArucoPoseEstimator 초기화
        self.estimator = ArucoPoseEstimator(
            camera_id=2,
            marker_length=50,  # mm 단위
            calibration_file="camera_calibration.pkl"
        )

        self.pose_publisher = self.create_publisher(ArucoPose, 
                                                    '/pickee/mobile/aruco_pose', 
                                                    10)
        
        # self.create_subscription(PickeeMobileArrival,
        #                         '/pickee/mobile/arrival',
        #                         self.detect_aruco_callback,
        #                         10)
        
        self.create_subscription(Bool,
                                '/pickee/mobile/docking_in_progress',
                                self.docking_status_callback,
                                10)
        
        thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        thread.start()
        self.get_logger().info("⌨️ Press 'z' to start ArUco detection, 'x' to stop")

    def detect_aruco_callback(self, msg: PickeeMobileArrival):
        self.docking_in_progress = True
        self.get_logger().info("🚦 Arrival detected! Starting ArUco scan...")
        self.read_marker()

        
    def read_marker(self):
        self.get_logger().info(f"{self.docking_in_progress}")

        while self.docking_in_progress:
            values = {
                "id": [],
                "x": [], "y": [], "z": [],
                "roll": [], "pitch": [], "yaw": []
            }

            for i in range(5):

                if self.docking_in_progress == False:
                    break   
                print(f"📸 Reading marker attempt {i+1}/5")

                ret, frame = self.estimator.cap.read()
                if not ret:
                    self.get_logger().warning("❌ 프레임을 읽을 수 없습니다.")
                    return

                frame_out, markers = self.estimator.process_frame(frame)

                if markers:
                    m = markers[0]

                    values["id"].append(m["id"])
                    values["x"].append(m["x"])
                    values["y"].append(m["y"])
                    values["z"].append(m["z"])
                    values["roll"].append(m["roll"])
                    values["pitch"].append(m["pitch"])
                    values["yaw"].append(m["yaw"])

                    self.get_logger().info(
                        f"🟢 {i+1}/10 | ID={m['id']} | "
                        f"x={m['x']:.1f}, y={m['y']:.1f}, z={m['z']:.1f} | "
                        f"roll={m['roll']:.1f}, pitch={m['pitch']:.1f}, yaw={m['yaw']:.1f}"
                    )
                else:
                    self.get_logger().info(f"⚠️ {i+1}/5 | Marker not found")

            if self.docking_in_progress == False:
                break

            if len(values["id"]) == 0:
                self.get_logger().error("❌ 5회 측정 중 마커를 하나도 감지하지 못했습니다.")
                pose = ArucoPose()
                pose.aruco_id = 0
                pose.x = 0.0
                pose.y = 0.0
                pose.z = 0.0
                pose.roll = 0.0
                pose.pitch = 0.0
                pose.yaw = 0.0
                self.pose_publisher.publish(pose)

                self.get_logger().info("📤 Published MEDIAN + MODE filtered ArUco pose ✅")

                continue

            # ✅ aruco_id 최빈값 Mode 계산
            aruco_id = Counter(values["id"]).most_common(1)[0][0]

            # ✅ Pose Median 계산
            median = {k: float(np.median(v)) for k, v in values.items() if k != "id"}

            self.get_logger().info(
                f"✅ Filter 완료 (samples={len(values['id'])})\n"
                f"ID={aruco_id}, "
                f"x={median['x']:.1f}, y={median['y']:.1f}, z={median['z']:.1f}, "
                f"roll={median['roll']:.1f}, pitch={median['pitch']:.1f}, yaw={median['yaw']:.1f}"
            )

            # ✅ Publish
            pose = ArucoPose()
            pose.aruco_id = aruco_id
            pose.x = median["x"]
            pose.y = median["y"]
            pose.z = median["z"]
            pose.roll = median["roll"]
            pose.pitch = median["pitch"]
            pose.yaw = median["yaw"]
            self.pose_publisher.publish(pose)

            self.get_logger().info("📤 Published MEDIAN + MODE filtered ArUco pose ✅")
        
    def docking_status_callback(self, msg: Bool):
        self.docking_in_progress = msg.data
        if not self.docking_in_progress:
            self.get_logger().info("🛑 Docking process ended. Stopping ArUco scan.")
            

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while True:
                key = sys.stdin.read(1)

                if key.lower() == 'z':
                    self.get_logger().info("✅ Key 's' pressed → Starting ArUco scan")
                    self.docking_in_progress = True
                    self.read_marker()

                elif key.lower() == 'x':
                    self.get_logger().info("🛑 Key 'q' pressed → Stop ArUco scan")
                    self.docking_in_progress = False

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


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
        node.estimator.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
