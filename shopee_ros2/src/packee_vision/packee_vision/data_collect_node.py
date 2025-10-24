import time
import random
import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose6D

import socket
import cv2
import threading
import numpy as np
import pandas as pd
import os
import pickle


PORT = 6000
MAX_PACKET_SIZE = 65536

with open("/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/camera_calibration/calibration_data.pickle", "rb") as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data["camera_matrix"]
dist_coeff = calib_data["dist_coeff"]

class VideoReceiver:
    def __init__(self, port=PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.packet_buffer = {}

    def run(self):
        try:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(MAX_PACKET_SIZE + 100)
                    if b'||' not in data:
                        continue

                    header, img_data = data.split(b'||', 1)
                    frame_id, packet_num, total_packets = map(int, header.decode().split(','))

                    if frame_id not in self.packet_buffer:
                        self.packet_buffer[frame_id] = [None] * total_packets

                    self.packet_buffer[frame_id][packet_num] = img_data

                    if all(p is not None for p in self.packet_buffer[frame_id]):
                        complete_data = b''.join(self.packet_buffer[frame_id])
                        np_data = np.frombuffer(complete_data, np.uint8)
                        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.frame = frame
                        del self.packet_buffer[frame_id]

                except Exception as e:
                    print(f"[VideoReceiver] Error: {e}")
                    continue
        finally:
            self.sock.close()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

class DataCollector(Node):
    def __init__(self, video_receiver):
        super().__init__("data_collect_node")
        self.receiver = video_receiver
        self.object_dict = {1: "wasabi", 10: "fish", 12: "eclipse"}

        self.object_id = 12
        self.target_pose = [223.6, 82.6, 194.9, -175.03, -1.42, -91.84]
        self.save_dir = "./datasets"

        # 현재 관절 상태
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0

        # 데이터 저장
        self.datasets = {"image_current":[], "image_target": [], "class": [], "pose":[]}
        self.count = 0
        self.current_pose_index = 0

        # ROS2 통신
        self.publisher = self.create_publisher(Pose6D, '/packee1/move', 10)
        self.subscriber = self.create_subscription(
            Pose6D,
            '/packee1/pose',
            self.AngleCallback,
            10
        )

        os.makedirs(self.save_dir + f"/{self.object_dict[self.object_id]}", exist_ok=True)

        self.MoveJetcobot(self.target_pose)
        time.sleep(3.0)

        # 주기적 데이터 수집 (0.5초)
        self.timer = self.create_timer(0.1, self.CollectCallback)
        self.get_logger().info("data_collect_node started")


    def MoveJetcobot(self, pose):
        self.x, self.y, self.z, self.rx, self.ry, self.rz = pose
        msg = Pose6D()
        msg.x = self.x
        msg.y = self.y
        msg.z = self.z
        msg.rx = self.rx
        msg.ry = self.ry
        msg.rz = self.rz
        self.publisher.publish(msg)

    def AngleCallback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z
        self.rx = msg.rx
        self.ry = msg.ry
        self.rz = msg.rz
        self.get_logger().info(f"Current angles: {msg.x}, {msg.y}, {msg.z}, {msg.rx}, {msg.ry}, {msg.rz}")

    def CollectCallback(self):
        frame = self.receiver.get_frame()
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeff, None, newcameramtx)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        if self.count >= 400:
            self.get_logger().info(f"{self.object_id} 데이터 수집 완료")
            df = pd.DataFrame(self.datasets)
            df.to_csv(f"{self.save_dir}/{self.object_dict[self.object_id]}/datasets.csv", index=False)
            return

        # target_pose 주변에서 랜덤 오프셋 생성
        dx = random.uniform(-10, 10)
        dy = random.uniform(-10, 10)
        dz = random.uniform(-10, 10)
        drz = random.uniform(-10, 10)

        pose = [
            self.target_pose[0] + dx,
            self.target_pose[1] + dy,
            self.target_pose[2] + dz,
            self.target_pose[3],
            self.target_pose[4],
            self.target_pose[5] + drz
        ]

        self.MoveJetcobot(pose)
        time.sleep(1.0)

        current_file_name = f"{self.save_dir}/{self.object_dict[self.object_id]}/image_{self.count:04d}.jpg"
        target_file_name = f"{self.save_dir}/targets/{self.object_dict[self.object_id]}_target.jpg"
        cv2.imwrite(current_file_name, undistorted)
        self.datasets['image_current'].append(current_file_name)
        self.datasets['image_target'].append(target_file_name)
        self.datasets['class'].append(self.object_id)
        self.datasets['pose'].append([float(self.target_pose[0] - pose[0]), float(self.target_pose[1] - pose[1]), float(self.target_pose[2] - pose[2]), float(self.target_pose[3] - pose[3]), float(self.target_pose[4] - pose[4]), float(self.target_pose[5] - pose[5])])
        self.count += 1

def main():
    rclpy.init()
    video_receiver = VideoReceiver(port=PORT)
    video_thread = threading.Thread(target=video_receiver.run)
    video_thread.start()

    node = DataCollector(video_receiver)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        video_receiver.stop()
        video_thread.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
