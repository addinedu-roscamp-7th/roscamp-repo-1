import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

class CnnClassifier:
    """
    CNN 이미지 분류 모델을 로드하고 추론을 수행하는 클래스.
    """
    def __init__(self, model_path, num_classes=2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 모델 구조 정의 (ResNet18)
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 저장된 가중치 불러오기
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 이미지 전처리 파이프라인 정의
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 클래스 이름
        self.class_names = ['empty_cart', 'full_cart']
        print(f"CNN classification model loaded from {model_path}")

    def classify(self, frame: np.ndarray) -> tuple[int, float, str]:
        """
        주어진 이미지 프레임의 클래스를 예측하고 결과를 반환합니다.

        :param frame: OpenCV 이미지 프레임 (numpy.ndarray)
        :return: (가장 확률이 높은 클래스 ID, 해당 확률, 클래스 이름) 튜플.
                 예: (0, 0.95, 'empty_cart')
        """
        try:
            # OpenCV는 BGR이므로 RGB로 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 전처리 + 배치 차원 추가
            input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

            # 추론
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                conf_val = conf.item()
                pred_idx_val = pred_idx.item()
                class_name = self.class_names[pred_idx_val]

            return (pred_idx_val, conf_val, class_name)

        except Exception as e:
            print(f"Error during classification: {e}")
            return (-1, 0.0, "error")