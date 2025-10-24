# 실행
# python3 inference.py --image  /home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/wasabi/images/wasabi_image_0001.jpg

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import argparse
import ast
import pandas as pd
import os

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

def load_model(model_path, num_classes, device):
    model = PoseCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def predict(model, img_path, pose_mean=None, pose_std=None, class_names=None, device="cpu"):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/cnn.pt")
    parser.add_argument("--csv", type=str, default="./datasets/labels.csv")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    # class 이름 추출
    df = pd.read_csv(args.csv)
    class_names = sorted(df["class"].unique())
    num_classes = len(class_names)

    # 포즈 통계 계산
    if "pose" in df.columns:
        poses = df["pose"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
        pose_mat = np.stack(poses.values)
        pose_mean, pose_std = pose_mat.mean(0).tolist(), pose_mat.std(0).tolist()
    else:
        pose_mean = pose_std = None

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, num_classes, device)

    # 예측 수행
    pose, cls_name = predict(model, args.image, pose_mean, pose_std, class_names, device)

    # 출력
    print(f"\n 예측 결과:")
    print(f"상품 클래스: {cls_name}")
    print(f"예측된 포즈: {pose.round(3)}")

if __name__ == "__main__":
    main()