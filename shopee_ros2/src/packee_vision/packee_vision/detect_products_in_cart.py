import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionDetectProductsInCart
from shopee_interfaces.msg import DetectedProduct, BBox, Pose6D
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import ast

with open("./src/camera_calibration/calibration_data.pickle", "rb") as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data["camera_matrix"]
dist_coeff = calib_data["dist_coeff"]

class PoseCNN(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        resnet=models.resnet18(pretrained=True)
        self.backbone=nn.Sequential(*list(resnet.children())[:-1])
        feat_dim=512
        self.pose_head=nn.Sequential(
            nn.Linear(feat_dim,256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256,6)
        )
        self.class_head=nn.Sequential(
            nn.Linear(feat_dim,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        f=self.backbone(x).flatten(1)
        pose_out=self.pose_head(f)
        cls_out=self.class_head(f)
        return pose_out,cls_out

class DetectProducts(Node):
    def __init__(self):
        super().__init__("detect_products_in_cart")
        self.packee1_cap = cv2.VideoCapture(0)
        self.packee2_cap = cv2.VideoCapture(1)

        if not self.packee1_cap.isOpened():
            self.get_logger().warn("[WARN] packee1 카메라 사용 불가")

        if not self.packee2_cap.isOpened():
            self.get_logger().warn("[WARN] packee2 카메라 사용 불가")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('./src/pickee_vision/resource/20251027_v11.pt').to(self.device)

        num_classes = 3
        self.packee1_cnn = PoseCNN(num_classes=num_classes).to(self.device)
        self.packee1_cnn.load_state_dict(torch.load(f'./src/packee_vision/packee_vision/checkpoints/packee1_cnn.pt', map_location=self.device))
        self.packee1_cnn.eval()

        self.packee2_cnn = PoseCNN(num_classes=num_classes).to(self.device)
        self.packee2_cnn.load_state_dict(torch.load(f'./src/packee_vision/packee_vision/checkpoints/packee2_cnn.pt', map_location=self.device))
        self.packee2_cnn.eval()
        self.gain = 0.3

        self.server = self.create_service(
            PackeeVisionDetectProductsInCart,
            "packee/vision/detect_products_in_cart",
            self.callback_service
        )

        self.get_logger().info("DetectProducts started")


    def load_pose_stats(self, csv):
        df = pd.read_csv(csv)
        if "pose" in df.columns:
            poses = df["pose"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
            mat = np.stack(poses.values)
            return mat.mean(0), mat.std(0)
        else:
            return None, None

    def predict(self, model, img, target_img_path, pose_mean=None, pose_std=None, device="cpu"):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        if target_img_path is not None:
            img = cv2.imread(target_img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pose_pred, _ = model(img_t)
            pose_pred = pose_pred.cpu().numpy().flatten()

        if pose_mean is not None and pose_std is not None:
            pose_pred = pose_pred * np.array(pose_std) + np.array(pose_mean)

        return pose_pred
    
    def callback_service(self, request, response):
        self.get_logger().info(
            f"Received request → robot_id: {request.robot_id}, order_id: {request.order_id}, expected_product_ids: {list(request.expected_product_id)}"
        )

        product_list = []
        camera_sources = {
            "left": (self.packee1_cap, self.packee1_cnn, 1),
            "right": (self.packee2_cap, self.packee2_cnn, 2)
        }

        for cam_side, (cap, model, cam_id) in camera_sources.items():
            ret, frame = cap.read()
            if not ret:
                self.get_logger().warn(f"[WARN] {cam_side} 카메라 프레임을 읽지 못했습니다.")
                continue

            # 카메라 왜곡 보정
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeff, (w, h), 1, (w, h))
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeff, None, newcameramtx)
            x, y, w, h = roi
            undistorted = undistorted[y:y + h, x:x + w]

            try:
                results = self.yolo_model(undistorted, conf=0.6)
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls.cpu().numpy())
                        if cls_id not in request.expected_product_id:
                            continue

                        class_name = self.yolo_model.names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

                        # ROI 중심으로 grid 구분
                        x_center = (x1 + x2) / 2
                        frame_center_x = undistorted.shape[1] / 2
                        offset = x_center - frame_center_x
                        threshold = 0.15 * undistorted.shape[1]
                        if offset > threshold:
                            grid_key = "grid3"
                        elif offset < -threshold:
                            grid_key = "grid2"
                        else:
                            grid_key = "grid1"

                        target_img_path = f"./src/packee_vision/target_img/packee{cam_id}_{class_name}_{grid_key}.jpg"
                        pose_mean, pose_std = self.load_pose_stats(f"./packee{cam_id}_datasets/labels.csv")

                        current_pose = self.predict(model, frame, None, pose_mean, pose_std, self.device)
                        target_pose = self.predict(model, None, target_img_path, pose_mean, pose_std, self.device)
                        delta_pose = self.gain * (target_pose - current_pose)

                        product = DetectedProduct()
                        product.product_id = cls_id
                        product.confidence = float(box.conf.item())
                        product.bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        product.pose = Pose6D(
                            x=float(current_pose[0] + delta_pose[0]),
                            y=float(current_pose[1] + delta_pose[1]),
                            z=float(current_pose[2] + delta_pose[2]),
                            rx=float(current_pose[3] + delta_pose[3]),
                            ry=float(current_pose[4] + delta_pose[4]),
                            rz=float(current_pose[5] + delta_pose[5])
                        )
                        product.arm_side = cam_side

                        product_list.append(product)
                        self.get_logger().info(f"[{cam_side}] Detected {class_name}")

            except Exception as e:
                self.get_logger().error(f"[{cam_side}] detection error: {e}")

        response.success = len(product_list) > 0
        response.products = product_list
        response.total_detected = len(product_list)
        response.message = f"Detected {len(product_list)} products from both cameras"
        return response

def main(args=None):
    rclpy.init(args=args)
    detect_product = DetectProducts()
   
    try:
        while rclpy.ok():
            rclpy.spin_once(detect_product, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if detect_product.packee1_cap.isOpened():
            detect_product.packee1_cap.release()
        if detect_product.packee2_cap.isOpened():
            detect_product.packee2_cap.release()
        detect_product.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
