import torch
import torch.nn as nn
from torchvision import transforms
from pose_cnn.model import PoseCNN
import numpy as np
import cv2
import argparse
import ast
import pandas as pd

def load_model(model_path, num_classes, device):
    model = PoseCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"[INFO] Loaded model: {model_path}")
    return model


def load_pose_stats(csv):
    df = pd.read_csv(csv)
    if "pose" in df.columns:
        poses = df["pose"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))
        mat = np.stack(poses.values)
        return mat.mean(0), mat.std(0)
    else:
        return None, None


def predict(model, img_path, class_names=None, device="cpu", pose_mean=None, pose_std=None):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] 이미지 파일을 불러올 수 없습니다: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = transform(img).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        pose_pred, cls_pred = model(img_t)
        pose_pred = pose_pred.cpu().numpy().flatten()
        cls_idx = cls_pred.argmax(dim=1).item()

    cls_name = class_names[cls_idx] if class_names else str(cls_idx)

    # pose 정규화 복원
    if pose_mean is not None and pose_std is not None:
        pose_pred = pose_pred * pose_std + pose_mean

    return pose_pred, cls_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/packee2_cnn.pt")
    parser.add_argument("--csv", type=str, default="./packee2_datasets/labels.csv")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 클래스 이름 추출
    df = pd.read_csv(args.csv)
    class_names = sorted(df["class"].unique())
    num_classes = len(class_names)

    # 모델 로드
    model = load_model(args.model, num_classes, device)

    # pose 통계 로드 (선택사항)
    pose_mean, pose_std = load_pose_stats(args.csv)

    # 예측 수행
    pose, cls_name = predict(
        model=model,
        img_path=args.image,
        class_names=class_names,
        device=device,
        pose_mean=pose_mean,
        pose_std=pose_std
    )

    # 출력
    print("\n[결과]")
    print(f"상품 클래스: {cls_name}")
    print(f"예측된 포즈: {np.round(pose, 3)}")


if __name__ == "__main__":
    main()
