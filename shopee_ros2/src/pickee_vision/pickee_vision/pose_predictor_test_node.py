import rclpy
from rclpy.node import Node
from shopee_interfaces.msg import Pose6D
import cv2
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import socket

class VideoReceiver:
    def __init__(self, port=6230):
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
                    data, addr = self.sock.recvfrom(65536)
                    if b'||' not in data: continue
                    header, img_data = data.split(b'||', 1)
                    frame_id, packet_num, total_packets = map(
                        int,
                        header.decode().split(',')
                    )
                    
                    if frame_id not in self.packet_buffer: 
                        self.packet_buffer[frame_id] = [None] * total_packets
                        
                    self.packet_buffer[frame_id][packet_num] = img_data
                    
                    if all(p is not None for p in self.packet_buffer[frame_id]):
                        complete_data = b''.join(self.packet_buffer[frame_id])
                        np_data = np.frombuffer(complete_data, np.uint8)
                        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock: self.frame = frame
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
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512
        self.pose_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(p=0.3), nn.Linear(256, 6)
        )
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Dropout(p=0.3), nn.Linear(128, num_classes)
        )
            
    def forward(self, x):
        f = self.backbone(x).flatten(1)
        pose_out = self.pose_head(f)
        cls_out = self.class_head(f)
        return pose_out, cls_out

class PickeeVisionControlNode(Node):
    def __init__(self, video_receiver):
        super().__init__("pose_predictor_test_node")
        self.get_logger().info("Initializing Pickee Vision Control Node...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.receiver = video_receiver

        # package_share_directory = get_package_share_directory('pickee_vision')
        # yolo_model_path = os.path.join(
        #     package_share_directory.replace(
        #         'install/pickee_vision/share/pickee_vision', 
        #         'src/pickee_vision'
        #     ), 
        #     'pickee_vision',
        #     '20251027_v11.pt'
        # )
        # cnn_model_path = os.path.join(
        #     package_share_directory.replace(
        #         'install/pickee_vision/share/pickee_vision', 
        #         'src/pickee_vision'
        #     ),
        #     'product_cnn_best.pt'
        # )

        # --- 의존성 클래스 초기화 (모델 파일 불러오기 위해) ---
        self.package_share_directory = get_package_share_directory('pickee_vision')
        # 1. 상품 인식용 세그멘테이션 모델
        yolo_model_path = os.path.join(self.package_share_directory, '20251027_v11.pt')
        # 2. 장바구니 인식용 클래시피케이션 모델
        cnn_model_path = os.path.join(self.package_share_directory, 'product_cnn_best.pt')


        try:
            self.yolo_model = YOLO(yolo_model_path).to(self.device)
            self.cnn_model = PoseCNN(num_classes=2).to(self.device)
            self.cnn_model.load_state_dict(
                torch.load(cnn_model_path, map_location=self.device)
            )
            self.cnn_model.eval()
            self.get_logger().info("YOLO and PoseCNN models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            return

        self.target_object_name = "14" # 6 = eclipse
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])

        self.publisher_ = self.create_publisher(Pose6D, '/pickee/arm/move_servo', 10)
        self.timer = self.create_timer(0.2, self.control_callback)
        self.get_logger().info("Pickee Vision Control Node started.")

    def predict_pose(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pose_pred, _ = self.cnn_model(img_tensor)
            return pose_pred.cpu().numpy().flatten()

    def control_callback(self):
        frame = self.receiver.get_frame()
        if frame is None:
            return

        results = self.yolo_model(frame, conf=0.5, device=self.device)

        for result in results:
            if not hasattr(result, 'boxes'): continue
            for box in result.boxes:
                # cls_id = int(box.cls.cpu().numpy())
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]
                if cls_name == self.target_object_name:
                    current_pose = self.predict_pose(frame)
                    target_image_path = os.path.join(self.package_share_directory, 'target_img_eclipce.jpg')

                    # 상품명으로 넣어줄 때 
                    # target_image_path = os.path.join(self.package_share_directory, 
                    #     f'target_img_{self.target_object_name}.png'
                    # )
                        
                    if not os.path.exists(target_image_path):
                        self.get_logger().warn(
                            f"Target image not found: {target_image_path}"
                        )
                        continue
                        
                    target_image = cv2.imread(target_image_path)
                    target_pose = self.predict_pose(target_image)
                    error = target_pose - current_pose
                    gain = 0.2
                    move_command = Pose6D()
                    
                    move_command.x, move_command.y, move_command.z, \
                    move_command.rx, move_command.ry, move_command.rz = (
                        float(gain * e) for e in error
                    )
                    
                    self.publisher_.publish(move_command)
                    self.get_logger().info(
                        f"Visual servoing... Error: {np.linalg.norm(error):.3f}"
                    )
                    break

        cv2.imshow("Pickee Vision", results[0].plot())
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    video_receiver = VideoReceiver()
    video_thread = threading.Thread(target=video_receiver.run)
    video_thread.start()

    node = PickeeVisionControlNode(video_receiver)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        video_receiver.stop()
        video_thread.join()
        rclpy.shutdown()

if __name__ == '__main__':
    main()