import rclpy
import cv2
import pickle
import numpy as np
import os
import math
import sys, termios, tty, threading
from rclpy.node import Node
from collections import Counter
from shopee_interfaces.msg import ArucoPose
from std_msgs.msg import Bool


class ArucoPoseEstimator:
    def __init__(self, camera_id=0, marker_length=50, calibration_file="./camera_calibration.pkl"):
        self.camera_id = camera_id
        self.marker_length = marker_length

        pkl_path = '/home/lim/project/roscamp-repo-1/shopee_ros2/src/pickee_mobile/pickee_mobile/module/camera_calibration.pkl'
        with open(pkl_path, 'rb') as f:
            calib = pickle.load(f)
        self.camera_matrix = calib['camera_matrix']
        self.dist_coeffs = calib['dist_coeffs']

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {self.camera_id}")

        print("✅ ArucoPoseEstimator Ready.")

    def get_pose(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        undist = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(undist)

        results = []
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
                cv2.drawFrameAxes(undist, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length*0.5)
                pos = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = self.euler(R)
                results.append({
                    "id": int(marker_id[0]),
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "roll": roll, "pitch": pitch, "yaw": yaw
                })

        return undist, results

    def euler(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        pitch = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(R[1,0], R[0,0])
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class ArucoReaderNode(Node):
    def __init__(self):
        super().__init__("aruco_reader")

        self.docking = False
        self.estimator = ArucoPoseEstimator(camera_id=2, marker_length=50)

        self.pose_pub = self.create_publisher(ArucoPose, "/pickee/mobile/aruco_pose", 10)

        self.create_subscription(Bool, "/pickee/mobile/docking_in_progress", self.on_docking, 10)

        threading.Thread(target=self.keyboard, daemon=True).start()
        self.get_logger().info("✅ Press Z to start, X to stop")

    def on_docking(self, msg):
        self.docking = msg.data
        if not self.docking:
            self.get_logger().info("🛑 Docking Stopped")

    def publish_pose(self, pose_dict):
        msg = ArucoPose()
        msg.aruco_id = pose_dict["id"]
        msg.x = pose_dict["x"]
        msg.y = pose_dict["y"]
        msg.z = pose_dict["z"]
        msg.roll = pose_dict["roll"]
        msg.pitch = pose_dict["pitch"]
        msg.yaw = pose_dict["yaw"]
        self.pose_pub.publish(msg)

    def scan(self):
        values = {k: [] for k in ["id","x","y","z","roll","pitch","yaw"]}

        for _ in range(5):
            if not self.docking: return
            frame, results = self.estimator.get_pose()
            if results:
                m = results[0]
                for k in values: values[k].append(m[k])

        if len(values["id"]) == 0:
            self.publish_pose({"id":0,"x":0,"y":0,"z":0,"roll":0,"pitch":0,"yaw":0})
            return

        mode_id = Counter(values["id"]).most_common(1)[0][0]
        median = {k: float(np.median(v)) for k,v in values.items() if k!="id"}

        pose = {"id":mode_id, **median}
        self.publish_pose(pose)

    def keyboard(self):
        old = termios.tcgetattr(sys.stdin); tty.setcbreak(sys.stdin)
        try:
            while True:
                key = sys.stdin.read(1).lower()
                if key == 'z':
                    self.get_logger().info("▶️ Start scan")
                    self.docking = True
                elif key == 'x':
                    self.get_logger().info("⏹ Stop scan")
                    self.docking = False
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    def spin_loop(self):
        while rclpy.ok():
            if self.docking:
                self.scan()

            #------------ 영상 송출부, 필요없으면 주석처리
            frame, _ = self.estimator.get_pose()
            if frame is not None:
                cv2.imshow("ArUco Viewer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            #------------


def main():
    rclpy.init()
    node = ArucoReaderNode()

    try:
        node.spin_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.estimator.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
