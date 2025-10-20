import os
from ultralytics import YOLO
import numpy as np

class YoloDetector:
    """
    YOLOv8 모델을 로드하고 객체 인식을 수행하는 클래스.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = YOLO(model_path)
        print(f"YOLOv8 model loaded from {model_path}")

    def detect(self, frame: np.ndarray) -> list:
        """
        주어진 이미지 프레임에서 객체를 탐지하고 결과를 반환합니다.

        :param frame: OpenCV 이미지 프레임 (numpy.ndarray)
        :return: 감지된 객체 정보 리스트. 예: 
                 [{'class_id': 0, 'confidence': 0.95, 'polygon': [[x1, y1], [x2, y2], ...]}, ...]
        """
        results = self.model(frame)
        detections = []

        if results[0].masks is None:
            return detections

        # masks.xy는 폴리곤 좌표의 리스트 (각각 numpy 배열)
        # boxes는 클래스 ID와 confidence를 포함
        for mask, box in zip(results[0].masks.xy, results[0].boxes):
            bbox = box.xyxy[0].tolist() # BBox 좌표 [x1, y1, x2, y2]
            detection = {
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'polygon': mask.tolist(),  # numpy 배열을 리스트로 변환
                'bbox': [int(coord) for coord in bbox] # 정수형으로 변환
            }
            detections.append(detection)
        
        return detections
