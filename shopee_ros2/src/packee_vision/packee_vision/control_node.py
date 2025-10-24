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
import time

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

class PoseCNN(nn.Module):
    def __init__(self, num_classes=3, pose_classes=6):
        super().__init__()
        
        base = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        
        self.reprogress = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        self.pose_head = nn.Linear(128, pose_classes)

        self.class_head = nn.Linear(128, num_classes)
    
    def forward(self, current_img, target_img):
        current_feature = self.feature_extractor(current_img).flatten(1)
        target_feature = self.feature_extractor(target_img).flatten(1)

        feature = torch.cat([current_feature, target_feature], dim=1)
        shared = self.reprogress(feature)

        pose_output = self.pose_head(shared)
        class_output = self.class_head(shared)
        

        return pose_output, class_output

class ControlNode(Node):
    def __init__(self, video_receiver):
        super().__init__("data_collect_node")
        self.receiver = video_receiver

        self.object_name = "eclipse"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/checkpoints/yolo_model.pt').to(device)

        num_classes = 3
        self.cnn = PoseCNN(num_classes=num_classes).to(device)
        self.cnn.load_state_dict(torch.load('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/packee_vision/packee_vision/checkpoints/best.pt', map_location=device))
        self.cnn.eval()

        self.publisher = self.create_publisher(Pose6D, '/packee1/move', 10)
        self.subscriber = self.create_subscription(
            Pose6D,
            '/packee1/pose',
            self.AngleCallback,
            10
        )

        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0

        self.MoveJetcobot(43.7, -42.6, 289.8, -153.26, 21.95, -84.37)
        time.sleep(3)

        # 주기적 데이터 수집 (0.5초)
        self.timer = self.create_timer(0.5, self.ControlCallback)
        self.get_logger().info("control_node started")

    def AngleCallback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z
        self.rx = msg.rx
        self.ry = msg.ry
        self.rz = msg.rz
        self.get_logger().info(f"Current angles: {msg.x}, {msg.y}, {msg.z}, {msg.rx}, {msg.ry}, {msg.rz}")


    def MoveJetcobot(self, x, y, z, rx, ry, rz):
        msg = Pose6D()
        msg.x = x
        msg.y = y
        msg.z = z
        msg.rx = rx
        msg.ry = ry
        msg.rz = rz
        self.publisher.publish(msg)

    def preprocess_image(self, cur_img, target_path, device):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        target_img = cv2.imread(target_path)
        if target_img is None:
            raise FileNotFoundError(target_path)

        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        cur_img_t = transform(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        target_img_t = transform(target_img).unsqueeze(0).to(device)
        return cur_img_t, target_img_t



    def predict(self, model, cur_img, target_path, class_names=None, device="cpu"):
        cur_img_t, tar_img_t = self.preprocess_image(cur_img, target_path, device)

        with torch.no_grad():
            pose_pred, cls_pred = model(cur_img_t, tar_img_t)
            pose_pred = pose_pred.cpu().numpy().flatten()
            cls_idx = cls_pred.argmax(dim=1).item()
            self.get_logger().info(f"Predicted class: {cls_idx}")

        cls_name = class_names[cls_idx]
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

        results = self.yolo_model(undistorted, conf=0.5, device='cuda')
        for result in results:
            if self.object_name in result.names.values():
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls.cpu().numpy())
                    cls_name = result.names[cls_id]
                    if cls_name == self.object_name:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

                        x_center = (x1 + x2) / 2
                        frame_center_x = undistorted.shape[1] / 2
                        offset = x_center - frame_center_x

                        # obj_crop = undistorted[y1:y2, x1:x2]

                        threshold = 0.15 * undistorted.shape[1]

                        if offset > threshold:
                            grid_key = "grid3"   # 오른쪽
                        elif offset < -threshold:
                            grid_key = "grid2"   # 왼쪽
                        else:
                            grid_key = "grid1"   # 중앙

                        target = f"/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/packee_vision/packee_vision/target_img/{self.object_name}_{grid_key}.jpg"


                        pose, cls_name = self.predict(
                            self.cnn, undistorted, target,
                            ['wasabi', 'fish', 'eclipse'], "cuda"
                        )


                        self.get_logger().info(f"Detected {cls_name} at {grid_key} with pose {pose}")

                        gain = 0.3  # 0.1~0.5 정도로 조정해보세요
                        self.MoveJetcobot(
                            float(self.x + gain * pose[0]),
                            float(self.y + gain * pose[1]),
                            float(self.z + gain * pose[2]),
                            float(self.rx + gain * pose[3]),
                            float(self.ry + gain * pose[4]),
                            float(self.rz + gain * pose[5])
                        )
        
        cv2.imshow("Undistorted", results[0].plot())
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
