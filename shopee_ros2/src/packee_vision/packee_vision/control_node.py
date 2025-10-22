import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose6D
import socket
import cv2
import threading
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from pose_cnn.model import PoseCNN


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


class ControlNode(Node):
    def __init__(self, video_receiver):
        super().__init__("data_collect_node")
        self.receiver = video_receiver

        self.object_name = "buldak_can"
        self.target_pose = [37.88, -30.93, -108.1, 46.66, -3.42, 36.73]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/checkpoints/yolo_model.pt').to(device)

        num_classes = 5
        self.cnn = PoseCNN(num_classes=num_classes).to(device)
        self.cnn.load_state_dict(torch.load('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/packee_vision/packee_vision/checkpoints/cnn.pt', map_location=device))
        self.cnn.eval()

        self.publisher = self.create_publisher(Pose6D, '/packee1/move', 10)

        # self.MoveJetcobot(*self.target_pose)

        # 주기적 데이터 수집 (0.5초)
        self.timer = self.create_timer(0.5, self.ControlCallback)
        self.get_logger().info("control_node started")


    def MoveJetcobot(self, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6):
        msg = Pose6D()
        msg.joint_1 = joint_1
        msg.joint_2 = joint_2
        msg.joint_3 = joint_3
        msg.joint_4 = joint_4
        msg.joint_5 = joint_5
        msg.joint_6 = joint_6
        self.get_logger().info(f"current msg: {msg}")
        self.publisher.publish(msg)

    def predict(self, model, img, pose_mean=None, pose_std=None, class_names=None, device="cpu"):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pose_pred, cls_pred = model(img_t)
            pose_pred = pose_pred.cpu().numpy().flatten()
            cls_idx = cls_pred.argmax(dim=1).item()

        # 역정규화 (있을 경우)
        if pose_mean is not None and pose_std is not None:
            pose_pred = pose_pred * np.array(pose_std) + np.array(pose_mean)

        cls_name = class_names[cls_idx] if class_names else str(cls_idx)

        return pose_pred, cls_name


    def ControlCallback(self):
        frame = self.receiver.get_frame()
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeff, None, newcameramtx)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        pose, cls_name = self.predict(self.cnn, undistorted, None, None, ['buldak_can', 'eclipse', 'pork', 'wasabi', 'fish'], "cuda")

        self.MoveJetcobot(float(self.target_pose[0] - pose[0]), float(self.target_pose[1] - pose[1]), float(self.target_pose[2] - pose[2]), float(self.target_pose[3] - pose[3]), float(self.target_pose[4] - pose[4]), float(self.target_pose[5] - pose[5]))

        cv2.imshow("Undistorted", undistorted)
        cv2.waitKey(1)

        

def main():
    rclpy.init()
    video_receiver = VideoReceiver(port=PORT)
    video_thread = threading.Thread(target=video_receiver.run)
    video_thread.start()

    node = ControlNode(video_receiver)

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
