import torch
import torch.nn as nn
from torchvision import models

class PoseCNN(nn.Module):
    def __init__(self, num_classes=3, pose_classes=6):
        super().__init__()
        
        base = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        
        self.reprogress = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        self.pose_head = nn.Linear(128, pose_classes)

        self.class_head = nn.Linear(128, num_classes)
    
    def forward(self, current_img, target_img):
        current_feature = self.feature_extractor(current_img).flatten(1)
        target_feature = self.feature_extractor(target_img).flatten(1)

        feature = torch.cat([current_feature, target_feature], dim=1)
        shared = self.reprogress(feature)

        pose_output = self.pose_head(shared)
        class_output = self.class_head(shared)
        

        return pose_output, class_output
    
if __name__ == "__main__":
    model = PoseCNN()
    current_img = torch.randn(1, 3, 224, 224)
    target_img = torch.randn(1, 3, 224, 224)

    pred_pose, pred_class = model(current_img, target_img)
    print(f"pose shape: {pred_pose.shape}")
    print(f"class shape: {pred_class.shape}")