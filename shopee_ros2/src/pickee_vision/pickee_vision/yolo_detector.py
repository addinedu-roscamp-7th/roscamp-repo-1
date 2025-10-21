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
        results = self.model.predict(source=frame, conf=0.8, iou=0.7) # stream=False가 기본값
        detections = []

        # predict의 결과는 리스트 형태이므로 순회합니다. (단일 이미지이므로 루프는 한 번만 실행됨)
        for result in results:
            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls)
                class_name = result.names[class_id]
                bbox = box.xyxy[0].tolist()
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(box.conf),
                    'polygon': mask.tolist(),
                    'bbox': [int(coord) for coord in bbox]
                }
                detections.append(detection)
        
        return detections
