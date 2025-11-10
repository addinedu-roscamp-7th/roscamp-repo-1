# 실행 
# python3 train.py --csv /home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/src/packee_vision/packee_vision/datasets/labels.csv --outdir ./checkpoints --epochs 60 --batch 32 --lr 1e-4

import os, ast, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pose_cnn.model import PoseCNN
import cv2

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir=None, transform=None, class_to_idx=None, pose_mean=None, pose_std=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
        # 클래스 인덱스 매핑
        if 'class' in self.df.columns:
            classes = sorted(self.df['class'].unique())
            self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = {}
        
        self.pose_mean = np.array(pose_mean, dtype=np.float32) if pose_mean is not None else None
        self.pose_std = np.array(pose_std, dtype=np.float32) if pose_std is not None else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 이미지 로드
        img_path = row.get('image_path') or row.get('images')
        if self.img_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.img_dir, os.path.basename(img_path))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        else:
            img = cv2.resize(img, (224, 224))
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Pose 파싱
        if 'pose' in self.df.columns:
            pose_str = row['pose']
            pose = np.array(ast.literal_eval(pose_str) if isinstance(pose_str, str) else pose_str, dtype=np.float32)
        else:
            cols = [c for c in self.df.columns if c.lower() in ['x', 'y', 'z', 'rx', 'ry', 'rz']]
            pose = np.array([float(row[c]) for c in cols], dtype=np.float32)

        if self.pose_mean is not None:
            pose = (pose - self.pose_mean) / (self.pose_std + 1e-8)

        # 클래스 인덱스 매핑
        if 'class' in self.df.columns:
            label_name = row['class']
            label = self.class_to_idx.get(label_name, 0)
        else:
            label = 0

        return img, torch.tensor(pose), torch.tensor(label)

def compute_pose_stats(csv):
    df=pd.read_csv(csv)
    if 'pose' in df.columns:
        poses=df['pose'].apply(lambda s:np.array(ast.literal_eval(s),dtype=np.float32))
        M=np.stack(poses.values)
    else:
        M=df[['x','y','z','rx','ry','rz']].values.astype(np.float32)
    return M.mean(0).tolist(), M.std(0).tolist()

def accuracy(logits,labels): return (logits.argmax(1)==labels).float().mean().item()
def mae(a,b): return (a-b).abs().mean().item()

def train_one_epoch(model,loader,opt,device,crit_pose,crit_cls,w_pose,w_cls):
    model.train()
    logs={"loss":0,"pose":0,"cls":0,"acc":0,"mae":0,"n":0}
    for imgs,poses,labels in loader:
        imgs,poses,labels=imgs.to(device),poses.to(device),labels.to(device)
        opt.zero_grad()
        pose_pred,cls_pred=model(imgs)
        loss_p=crit_pose(pose_pred,poses)
        loss_c=crit_cls(cls_pred,labels)
        loss=w_pose*loss_p+w_cls*loss_c
        loss.backward(); opt.step()

        logs["loss"]+=loss.item()*len(imgs)
        logs["pose"]+=loss_p.item()*len(imgs)
        logs["cls"]+=loss_c.item()*len(imgs)
        logs["acc"]+=accuracy(cls_pred,labels)*len(imgs)
        logs["mae"]+=mae(pose_pred,poses)*len(imgs)
        logs["n"]+=len(imgs)
    for k in list(logs.keys())[:-1]:
        logs[k]/=logs["n"]
    return logs

@torch.no_grad()
def validate(model,loader,device,crit_pose,crit_cls,w_pose,w_cls):
    model.eval()
    logs={"loss":0,"pose":0,"cls":0,"acc":0,"mae":0,"n":0}
    for imgs,poses,labels in loader:
        imgs,poses,labels=imgs.to(device),poses.to(device),labels.to(device)
        pose_pred,cls_pred=model(imgs)
        loss_p=crit_pose(pose_pred,poses)
        loss_c=crit_cls(cls_pred,labels)
        loss=w_pose*loss_p+w_cls*loss_c
        logs["loss"]+=loss.item()*len(imgs)
        logs["pose"]+=loss_p.item()*len(imgs)
        logs["cls"]+=loss_c.item()*len(imgs)
        logs["acc"]+=accuracy(cls_pred,labels)*len(imgs)
        logs["mae"]+=mae(pose_pred,poses)*len(imgs)
        logs["n"]+=len(imgs)
    for k in list(logs.keys())[:-1]:
        logs[k]/=logs["n"]
    return logs


def plot_training(history,outpath):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(history["train_loss"],label="Train Loss")
    plt.plot(history["val_loss"],label="Val Loss")
    plt.legend(); plt.title("Total Loss")

    plt.subplot(1,3,2)
    plt.plot(history["train_acc"],label="Train Acc")
    plt.plot(history["val_acc"],label="Val Acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1,3,3)
    plt.plot(history["train_mae"],label="Train MAE")
    plt.plot(history["val_mae"],label="Val MAE")
    plt.legend(); plt.title("Pose MAE")

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"그래프 저장됨: {outpath}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",type=str,default="./datasets/all_labels.csv")
    parser.add_argument("--outdir",type=str,default="./checkpoints_v3")
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--w_pose",type=float,default=0.9)
    parser.add_argument("--w_cls",type=float,default=0.1)
    parser.add_argument("--patience",type=int,default=10,help="조기 종료 기준 epoch 수")
    args=parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    pose_mean,pose_std=compute_pose_stats(args.csv)
    df=pd.read_csv(args.csv).sample(frac=1).reset_index(drop=True)
    split=int(0.8*len(df))
    df_train,df_val=df[:split],df[split:]
    df_train.to_csv("train_split.csv",index=False)
    df_val.to_csv("val_split.csv",index=False)

    transform_train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ColorJitter(0.3,0.3,0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0,translate=(0.1,0.1),scale=(0.9,1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    transform_val=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_set=PoseDataset("train_split.csv",transform=transform_train,pose_mean=pose_mean,pose_std=pose_std)
    val_set=PoseDataset("val_split.csv",transform=transform_val,pose_mean=pose_mean,pose_std=pose_std,class_to_idx=train_set.class_to_idx)
    train_loader=DataLoader(train_set,batch_size=args.batch,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=args.batch,shuffle=False)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PoseCNN(num_classes=len(train_set.class_to_idx)).to(device)
    crit_p,crit_c=nn.MSELoss(),nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=args.lr)

    best=float("inf")
    patience_counter=0
    hist={"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "train_mae":[], "val_mae":[]}

    for ep in range(1,args.epochs+1):
        tr=train_one_epoch(model,train_loader,opt,device,crit_p,crit_c,args.w_pose,args.w_cls)
        va=validate(model,val_loader,device,crit_p,crit_c,args.w_pose,args.w_cls)
        hist["train_loss"].append(tr["loss"]); hist["val_loss"].append(va["loss"])
        hist["train_acc"].append(tr["acc"]); hist["val_acc"].append(va["acc"])
        hist["train_mae"].append(tr["mae"]); hist["val_mae"].append(va["mae"])

        print(f"[{ep:03d}] ValLoss={va['loss']:.4f} Acc={va['acc']*100:.1f}% MAE={va['mae']:.4f}")

        # Early Stopping logic
        if va["loss"] < best:
            best = va["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.outdir,"best.pt"))
            print("Best model updated")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("조기 종료: Validation loss 개선 없음")
                break

    plot_training(hist, os.path.join(args.outdir,"train_plot.png"))
    print("학습 완료")

if __name__=="__main__":
    main()