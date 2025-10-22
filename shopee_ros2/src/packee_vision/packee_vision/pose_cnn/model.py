import torch
import torch.nn as nn
from torchvision import models

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