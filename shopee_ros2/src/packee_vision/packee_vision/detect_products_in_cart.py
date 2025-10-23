import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionDetectProductsInCart
from shopee_interfaces.msg import PackeeDetectedProduct
from shopee_interfaces.msg import BBox
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
    def __init__(self, num_classes=6):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512
        self.pose_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 6)
        )
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        pose_out = self.pose_head(f)
        cls_out = self.class_head(f)
        return pose_out, cls_out



class DetectProducts(Node):
    def __init__(self, video_receiver):
        super().__init__("detect_products_in_cart")
        self.receiver = video_receiver

        self.object_name = "wasabi"
        self.target_pose = {"grid1": [37.88, -30.93, -108.1, 46.66, -3.42, 36.73],
                            "grid2": [47.72, -36.47, -102.21, 55.63, -2.28, 52.55],
                            "grid3": [60.2, -51.76, -67.67, 31.81, -1.93, 59.58]}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/checkpoints/yolo_model.pt').to(device)

        num_classes = 5
        self.cnn = PoseCNN(num_classes=num_classes).to(device)
        self.cnn.load_state_dict(torch.load('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/packee_vision/packee_vision/checkpoints/cnn.pt', map_location=device))
        self.cnn.eval()

        self.server = self.create_service(
            PackeeVisionDetectProductsInCart,
            "packee/vision/detect_products_in_cart",
            self.callback_service
        )

        self.get_logger().info("DetectProducts started")

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

        if pose_mean is not None and pose_std is not None:
            pose_pred = pose_pred * np.array(pose_std) + np.array(pose_mean)

        cls_name = class_names[cls_idx] if class_names else str(cls_idx)

        return pose_pred, cls_name
    
    def callback_service(self, request, response):
        self.get_logger().info(
            f"Received request -> "
            f"robot_id: {request.robot_id},"
            f"order_id: {request.order_id},"
            f"expend_product_ids: {list(request.expected_product_ids)}"
        )

        frame = self.receiver.get_frame()
        if frame is None:
            response.success = False
            response.products = []
            response.total_detected = 0
            response.message = "No products detected"
            return response
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeff, None, newcameramtx)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        product_list = []


        for product_id in request.expected_product_ids:

            results = self.yolo_model(undistorted, conf=0.5, device='cuda')
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls.cpu().numpy())
                    if cls_id == product_id:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

                        x_center = (x1 + x2) / 2
                        frame_center_x = undistorted.shape[1] / 2
                        offset = x_center - frame_center_x

                        obj_crop = undistorted[y1:y2, x1:x2]

                        threshold = 0.15 * undistorted.shape[1]

                        if offset > threshold:
                            grid_key = "grid3"
                        elif offset < -threshold:
                            grid_key = "grid2"
                        else:
                            grid_key = "grid1"


                        pose, _ = self.predict(
                            self.cnn, obj_crop, None, None,
                            ['eclipse', 'wasabi', 'fish', 'apple', 'soymilk'], "cuda"
                        )

                        target = self.target_pose[grid_key]

                        p = PackeeDetectedProduct()
                        p.product_id = product_id
                        p.confidence = float(box.conf.item())
                        p.bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        p.position = Pose6D(joint_1=float(target[0] - pose[0]), joint_2=float(target[1] - pose[1]), joint_3=float(target[2] - pose[2]),
                                        joint_4=float(target[3] - pose[3]), joint_5=float(target[4] - pose[4]), joint_6=float(target[5] - pose[5]))
                        product_list.append(p)

        response.success = True
        response.products = product_list
        response.total_detected = len(product_list)

        if len(product_list) == 1:
            response.message = "Only one product detected"
        else:
            response.message = "All products detected"

        self.get_logger().info(
            f"Response -> success={response.success}, total_detected={response.total_detected}, "
            f"message={response.message}"
        )

        return response

def main(args=None):
    rclpy.init(args=args)
    video_receiver = VideoReceiver(port=PORT)
    video_thread = threading.Thread(target=video_receiver.run)
    video_thread.start()

    detect_product = DetectProducts(video_receiver)
   
    try:
        while rclpy.ok():
            rclpy.spin_once(detect_product, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        detect_product.destroy_node()
        rclpy.shutdown()
        video_receiver.stop()
        video_thread.join()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
