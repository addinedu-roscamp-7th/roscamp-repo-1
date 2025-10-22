import time
import random
import rclpy
from rclpy.node import Node
from jetcobot_package_msgs.msg import PoseVel, Pose

import socket
import cv2
import threading
import numpy as np
import pandas as pd
import os
import pickle


PORT = 6000
MAX_PACKET_SIZE = 65536

with open("/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/calibration_img/calibration_data.pickle", "rb") as f:
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

        self.object_name = "buldak_can"
        self.target_pose = [10.19, -29.61, -47.02, -3.25, 1.93, 8.17]
        self.save_dir = f"./datasets/{self.object_name}"
        self.offsets = [
                            (20, 0, 0, 0),   # right
                            (-20, 0, 0, 0),  # left
                            (0, 20, 0, 0),   # forward
                            (0, -20, 0, 0),  # backward
                            (0, 0, 20, 0),   # up
                            (0, 0, -20, 0),  # down
                            (0, 0, 0, 10),   # rotate cw
                            (0, 0, 0, -10)   # rotate ccw
                        ]

        # 현재 관절 상태
        self.joint_1 = 0
        self.joint_2 = 0
        self.joint_3 = 0
        self.joint_4 = 0
        self.joint_5 = 0
        self.joint_6 = 0

        # 데이터 저장
        self.datasets = {"images":[], "class": [], "pose":[]}
        self.count = 0
        self.current_pose_index = 0

        # ROS2 통신
        self.publisher = self.create_publisher(PoseVel, '/packee1/move', 10)
        self.subscriber = self.create_subscription(
            Pose,
            '/packee1/pose',
            self.AngleCallback,
            10
        )

        os.makedirs(self.save_dir + "/images", exist_ok=True)

        self.MoveJetcobot(self.target_pose)

        # 주기적 데이터 수집 (0.5초)
        self.timer = self.create_timer(0.1, self.CollectCallback)
        self.get_logger().info("data_collect_node started")


    def MoveJetcobot(self, pose):
        self.joint_1, self.joint_2, self.joint_3, self.joint_4, self.joint_5, self.joint_6 = pose
        msg = PoseVel()
        msg.pose_1 = self.joint_1
        msg.pose_2 = self.joint_2
        msg.pose_3 = self.joint_3
        msg.pose_4 = self.joint_4
        msg.pose_5 = self.joint_5
        msg.pose_6 = self.joint_6
        msg.speed = 30
        self.publisher.publish(msg)

    def AngleCallback(self, msg):
        if len(msg.angles) >= 6:
            self.joint_1 = msg.angles[0]
            self.joint_2 = msg.angles[1]
            self.joint_3 = msg.angles[2]
            self.joint_4 = msg.angles[3]
            self.joint_5 = msg.angles[4]
            self.joint_6 = msg.angles[5]
            self.get_logger().info(f"Current angles: {msg.angles}")

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
            self.get_logger().info(f"{self.object_name} 데이터 수집 완료")
            df = pd.DataFrame(self.datasets)
            df.to_csv(f"{self.save_dir}/datasets.csv", index=False)
            return

        # target_pose 주변에서 랜덤 오프셋 생성
        dx = random.uniform(-5, 5)
        dy = random.uniform(-5, 5)
        dz = random.uniform(-5, 5)
        drz = random.uniform(-5, 5)

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

        file_name = f"{self.save_dir}/images/{self.object_name}_{self.count:04d}.jpg"
        cv2.imwrite(file_name, undistorted)
        self.datasets['images'].append(file_name)
        self.datasets['pose'].append(pose)
        self.datasets['class'].append(self.object_name)
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
