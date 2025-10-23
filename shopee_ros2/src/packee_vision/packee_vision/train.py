# 실행 예시:
# python3 train_twostream.py --csv ./datasets/labels.csv --outdir ./checkpoints --epochs 60 --batch 32 --lr 1e-4

import os, ast, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from pose_cnn.model import PoseCNN   # ← 방금 만든 Pose+Class 모델 (Two-stream)

# ------------------------------
# Dataset
# ------------------------------
class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, pose_mean=None, pose_std=None, class_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # 클래스 매핑
        if "class" in self.df.columns:
            classes = sorted(self.df["class"].unique())
            self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = {}

        # 포즈 정규화용 통계
        self.pose_mean = np.array(pose_mean, dtype=np.float32) if pose_mean is not None else None
        self.pose_std  = np.array(pose_std, dtype=np.float32) if pose_std is not None else None


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cur_path = row["image_current"]
        tar_path = row["image_target"]

        cur_img = cv2.imread(cur_path)
        tar_img = cv2.imread(tar_path)
        if cur_img is None or tar_img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {cur_path}, {tar_path}")

        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            cur_img = self.transform(cur_img)
            tar_img = self.transform(tar_img)
        else:
            cur_img = torch.tensor(cv2.resize(cur_img, (224, 224)), dtype=torch.float32).permute(2, 0, 1) / 255.0
            tar_img = torch.tensor(cv2.resize(tar_img, (224, 224)), dtype=torch.float32).permute(2, 0, 1) / 255.0

        pose_str = row["pose"]
        pose = np.array(ast.literal_eval(pose_str), dtype=np.float32)
        if self.pose_mean is not None:
            pose = (pose - self.pose_mean) / (self.pose_std + 1e-8)

        label = self.class_to_idx.get(row.get("class", "candy"), 0)
        return cur_img, tar_img, torch.tensor(pose), torch.tensor(label)


# ------------------------------
# 학습/검증 루프
# ------------------------------
def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def mae(a, b):
    return (a - b).abs().mean().item()

def train_one_epoch(model, loader, opt, device, crit_pose, crit_cls, w_pose, w_cls):
    model.train()
    logs = {"loss": 0, "pose": 0, "cls": 0, "acc": 0, "mae": 0, "n": 0}
    for cur, tar, pose_gt, cls_gt in loader:
        cur, tar, pose_gt, cls_gt = cur.to(device), tar.to(device), pose_gt.to(device), cls_gt.to(device)
        opt.zero_grad()
        pose_pred, cls_pred = model(cur, tar)

        loss_p = crit_pose(pose_pred, pose_gt)
        loss_c = crit_cls(cls_pred, cls_gt)
        loss = w_pose * loss_p + w_cls * loss_c

        loss.backward()
        opt.step()

        logs["loss"] += loss.item() * len(cur)
        logs["pose"] += loss_p.item() * len(cur)
        logs["cls"] += loss_c.item() * len(cur)
        logs["acc"] += accuracy(cls_pred, cls_gt) * len(cur)
        logs["mae"] += mae(pose_pred, pose_gt) * len(cur)
        logs["n"] += len(cur)

    for k in list(logs.keys())[:-1]:
        logs[k] /= logs["n"]
    return logs


@torch.no_grad()
def validate(model, loader, device, crit_pose, crit_cls, w_pose, w_cls):
    model.eval()
    logs = {"loss": 0, "pose": 0, "cls": 0, "acc": 0, "mae": 0, "n": 0}
    for cur, tar, pose_gt, cls_gt in loader:
        cur, tar, pose_gt, cls_gt = cur.to(device), tar.to(device), pose_gt.to(device), cls_gt.to(device)
        pose_pred, cls_pred = model(cur, tar)

        loss_p = crit_pose(pose_pred, pose_gt)
        loss_c = crit_cls(cls_pred, cls_gt)
        loss = w_pose * loss_p + w_cls * loss_c

        logs["loss"] += loss.item() * len(cur)
        logs["pose"] += loss_p.item() * len(cur)
        logs["cls"] += loss_c.item() * len(cur)
        logs["acc"] += accuracy(cls_pred, cls_gt) * len(cur)
        logs["mae"] += mae(pose_pred, pose_gt) * len(cur)
        logs["n"] += len(cur)

    for k in list(logs.keys())[:-1]:
        logs[k] /= logs["n"]
    return logs


# ------------------------------
# 그래프 시각화
# ------------------------------
def plot_training(history, outpath):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Total Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(history["train_mae"], label="Train MAE")
    plt.plot(history["val_mae"], label="Val MAE")
    plt.legend(); plt.title("Pose MAE")

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"📈 그래프 저장됨: {outpath}")


# ------------------------------
# 메인 학습 루프
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./checkpoints_twostream")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_pose", type=float, default=0.8)
    parser.add_argument("--w_cls", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 포즈 통계 계산
    df = pd.read_csv(args.csv)
    poses = np.stack(df["pose"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)))
    pose_mean, pose_std = poses.mean(0), poses.std(0)

    # 데이터 분할
    df = df.sample(frac=1).reset_index(drop=True)
    split = int(0.8 * len(df))
    df.iloc[:split].to_csv("train_split.csv", index=False)
    df.iloc[split:].to_csv("val_split.csv", index=False)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    train_set = PoseDataset("train_split.csv", transform=transform, pose_mean=pose_mean, pose_std=pose_std)
    val_set = PoseDataset("val_split.csv", transform=transform, pose_mean=pose_mean, pose_std=pose_std, class_to_idx=train_set.class_to_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseCNN(num_classes=len(train_set.class_to_idx)).to(device)
    crit_pose, crit_cls = nn.MSELoss(), nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    patience_counter = 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_mae": [], "val_mae": []}

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, crit_pose, crit_cls, args.w_pose, args.w_cls)
        va = validate(model, val_loader, device, crit_pose, crit_cls, args.w_pose, args.w_cls)

        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["train_acc"].append(tr["acc"])
        hist["val_acc"].append(va["acc"])
        hist["train_mae"].append(tr["mae"])
        hist["val_mae"].append(va["mae"])

        print(f"[{ep:03d}] ValLoss={va['loss']:.4f} Acc={va['acc']*100:.1f}% MAE={va['mae']:.4f}")

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
            print("Best model updated")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("조기 종료: Validation loss 개선 없음")
                break

    plot_training(hist, os.path.join(args.outdir, "train_plot.png"))
    print("🎉 학습 완료")


if __name__ == "__main__":
    main()