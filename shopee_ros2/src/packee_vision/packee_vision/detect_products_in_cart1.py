import rclpy 
from rclpy.node import Node
from shopee_interfaces.srv import PackeeVisionDetectProductsInCart
from rclpy.executors import MultiThreadedExecutor
from shopee_interfaces.msg import Pose6D
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
    def __init__(self, packee_num, video_cap):
        super().__init__(f"packee{packee_num}_vision_node")
        self.packee_num = packee_num
        self.video_cap = video_cap

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('./src/pickee_vision/resource/20251104_v11_ver1_ioudefault.pt').to(self.device)

        num_classes = 3
        self.cnn = PoseCNN(num_classes=num_classes).to(self.device)
        self.cnn.load_state_dict(torch.load(f'./src/packee_vision/packee_vision/checkpoints/packee{self.packee_num}_cnn.pt', map_location=self.device))
        self.cnn.eval()

        self.products = {1: "wasabi", 12: "fish", 14: "eclipse"}

        self.server = self.create_service(
            PackeeVisionDetectProductsInCart,
            f"packee{self.packee_num}/vision/detect_products_in_cart",
            self.callback_service
        )

        self.get_logger().info(f"packee{self.packee_num} vision node started")


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
            f"Received request → robot_id: {request.robot_id}, order_id: {request.order_id}, expected_product_id: {request.expected_product_id}"
        )

        if not self.video_cap.isOpened():
            self.get_logger().warn(f"[WARN] packee {self.packee_num} 카메라가 열리지 않았습니다.")
            response.success = False
            response.current_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.target_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.message = f"[packee{self.packee_num}] 카메라 열기 실패"
            return response

        ret, frame = self.video_cap.read()
        if not ret:
            self.get_logger().warn(f"[WARN] packee {self.packee_num} 카메라 프레임을 읽지 못했습니다.")
            response.success = False
            response.current_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.target_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.message = f"[packee{self.packee_num}] 카메라 프레임 읽기 실패"
            return response

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
                if not hasattr(result, "masks") or result.masks is None:
                    self.get_logger().warn(f"[packee{self.packee_num}] 세그멘테이션 마스크가 없습니다.")
                    continue

                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i].cpu().numpy())
                    class_name = int(self.yolo_model.names[cls_id])

                    if class_name != request.expected_product_id:
                        continue

                    # --- 세그멘테이션 중심 계산 ---
                    mask_np = mask.cpu().numpy()
                    ys, xs = np.nonzero(mask_np)
                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    mask_center_x = np.mean(xs)
                    mask_center_y = np.mean(ys)

                    frame_center_x = undistorted.shape[1] / 2
                    offset = mask_center_x - frame_center_x
                    threshold = 0.15 * undistorted.shape[1]

                    if offset > threshold:
                        grid_key = "grid1"
                    elif offset < -threshold:
                        grid_key = "grid3"
                    else:
                        grid_key = "grid2"

                    target_img_path = f"./src/packee_vision/packee_vision/target_img/packee{self.packee_num}_{self.products[int(class_name)]}_{grid_key}.jpg"
                    pose_mean, pose_std = self.load_pose_stats(f"./src/packee_vision/packee_vision/packee{self.packee_num}_datasets/labels.csv")

                    current_pose = self.predict(self.cnn, frame, None, pose_mean, pose_std, self.device)
                    target_pose = self.predict(self.cnn, None, target_img_path, pose_mean, pose_std, self.device)

                    response.success = True
                    response.current_pose = Pose6D(x=float(current_pose[0]), y=float(current_pose[1]), z=float(current_pose[2]), rx=float(current_pose[3]), ry=float(current_pose[4]), rz=float(current_pose[5]))
                    response.target_pose = Pose6D(x=float(target_pose[0]), y=float(target_pose[1]), z=float(target_pose[2]), rx=float(target_pose[3]), ry=float(target_pose[4]), rz=float(target_pose[5]))
                    response.message = f"[packee{self.packee_num}] pose 예측을 성공하였습니다."

                    self.get_logger().info(f"[packee{self.packee_num}] current_pose: {current_pose},  target_pose: {target_pose}")

        except Exception as e:
            self.get_logger().error(f"[packee {self.packee_num}] segmentation error: {e}")
            response.success = False
            response.current_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.target_pose = Pose6D(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
            response.message = f"[packee{self.packee_num}] pose 예측을 실패하였습니다."

        return response

def main(args=None):
    rclpy.init(args=args)

    packee1_cap = cv2.VideoCapture(1)

    packee1_node = DetectProducts(1, packee1_cap)

    executor = MultiThreadedExecutor()
    executor.add_node(packee1_node)

    try:
        while rclpy.ok():
            rclpy.spin_once(packee1_node, timeout_sec=0.01)

    except KeyboardInterrupt:
        pass

    finally:
        packee1_node.destroy_node()
        rclpy.shutdown()

        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
