# 실행 예시:
# python3 inference.py \
# --image /home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/wasabi/images/wasabi_image_0001.jpg \
# --target /home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/wasabi/target.jpg \
# --model ./checkpoints/best.pt \
# --csv ./datasets/labels.csv

import torch
import numpy as np
import cv2
import argparse
import ast
import pandas as pd
from torchvision import transforms
from pose_cnn.model import PoseCNN


def load_model(model_path, num_classes, device):
    model = PoseCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    return model


def preprocess_image(path, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t


def predict(model, current_path, target_path, pose_mean=None, pose_std=None, class_names=None, device="cpu"):
    cur_img = preprocess_image(current_path, device)
    tar_img = preprocess_image(target_path, device)

    with torch.no_grad():
        pose_pred, cls_pred = model(cur_img, tar_img)
        pose_pred = pose_pred.cpu().numpy().flatten()
        cls_idx = cls_pred.argmax(dim=1).item()

    # 역정규화
    if pose_mean is not None and pose_std is not None:
        pose_pred = pose_pred * np.array(pose_std) + np.array(pose_mean)

    cls_name = class_names[cls_idx] if class_names else f"cls_{cls_idx}"
    return pose_pred, cls_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/best.pt", help="학습된 모델 경로")
    parser.add_argument("--csv", type=str, default="./datasets/labels.csv", help="클래스 및 포즈 통계 CSV")
    parser.add_argument("--image", type=str, required=True, help="현재 이미지 경로 (current image)")
    parser.add_argument("--target", type=str, help="목표 이미지 경로 (target image)")
    args = parser.parse_args()

    # CSV에서 class 이름 추출
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

    # target 이미지 자동 선택 (지정 안 했을 경우)
    if args.target is None:
        default_target = f"{'/'.join(args.image.split('/')[:-2])}/target.jpg"
        if not cv2.haveImageReader(default_target):
            raise FileNotFoundError("target image가 지정되지 않았고, 기본 경로에서도 찾을 수 없습니다.")
        args.target = default_target
        print(f"📸 자동 선택된 target 이미지: {args.target}")

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, num_classes, device)

    # 예측 수행
    pose, cls_name = predict(model, args.image, args.target, pose_mean, pose_std, class_names, device)

    # 출력
    print("\n 예측 결과")
    print(f"상품 클래스: {cls_name}")
    print(f"예측된 포즈 (x,y,z,Rx,Ry,Rz): {pose.round(3)}")


if __name__ == "__main__":
    main()
