import cv2
import cv2.aruco
import numpy as np
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

class CameraThread(QThread):
    frame_ready = Signal(QImage)
    marker_detected = Signal(int, np.ndarray, np.ndarray)
    person_detected = Signal(int, int, int, int, float, str)  # 사람 인식 결과 [(x1, y1, x2, y2, confidence, track_id, position), ...]
    person_position = Signal(str, int, int)  # (위치: 'left'/'center'/'right', center_x, center_y)

    def __init__(self, camera_index=2):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.last_marker_emit_time = 0.0 
        self.marker_emit_time = 0.2  # 마커 정보 발행 간격 (초)

        # 아르코 마커 딕셔너리 및 파라미터 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        # 카메라 행렬 및 왜곡 계수 (캘리브레이션 결과)
        self.camera_matrix = np.array([
            [2.07267968e+03, 0.00000000e+00, 3.87826573e+02],
            [0.00000000e+00, 1.01210308e+04, 2.19265005e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.dist_coeffs = np.array([[-2.83991246, -23.92614276, 0.03117976, -0.19365304, 8.5383573]])
        self.marker_length = 0.05

        # YOLOv8 모델 로드 (사람 인식용)
        try:
            self.yolo_model = YOLO('./employee_cloth_best.pt')  # employee_best.pt 경로로 변경
            self.yolo_enabled = True
            # 오탐지 방지를 위한 시간적 필터링 변수
            self.employee_detect_history = []  # 최근 N 프레임의 탐지 결과 저장
            self.HISTORY_SIZE = 10  # 최근 10 프레임 추적
            self.DETECTION_RATIO_THRESHOLD = 0.7  # 70% 이상 탐지되어야 최종 인식
        except Exception as e:
            print(f'YOLO 모델 로드 실패: {e}')
            self.yolo_enabled = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # 영상 왜곡 보정
                h, w = frame.shape[:2]
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
                undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)

                # 아르코 마커 검출 (ArucoDetector 기반)
                corners, ids, rejected = self.aruco_detector.detectMarkers(undistorted_frame)

                if ids is not None:
                    undistorted_frame = cv2.aruco.drawDetectedMarkers(undistorted_frame, corners, ids)
                    
                    # OpenCV 4.7+ 호환: 각 마커의 pose 추정
                    rvecs = []
                    tvecs = []
                    for corner in corners:
                        # 마커의 3D 좌표 (마커 중심 기준)
                        half_size = self.marker_length / 2.0
                        obj_points = np.array([
                            [-half_size, half_size, 0],
                            [half_size, half_size, 0],
                            [half_size, -half_size, 0],
                            [-half_size, -half_size, 0]
                        ], dtype=np.float32)
                        
                        # solvePnP로 pose 추정
                        success, rvec, tvec = cv2.solvePnP(
                            obj_points, corner[0], self.camera_matrix, self.dist_coeffs, 
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        if success:
                            rvecs.append(rvec)
                            tvecs.append(tvec)

                    current_time = time.time()
                    if current_time - self.last_marker_emit_time >= self.marker_emit_time:
                        for i, marker_id in enumerate(ids.flatten()):
                            if i < len(tvecs):
                                self.marker_detected.emit(marker_id, tvecs[i], rvecs[i])
                                cv2.drawFrameAxes(undistorted_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.2)

                        self.last_marker_emit_time = current_time
                    else:
                        # 3초가 안 지났어도 좌표축은 표시
                        for i, marker_id in enumerate(ids.flatten()):
                            if i < len(tvecs):
                                cv2.drawFrameAxes(undistorted_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.2)

                # YOLOv8로 사람 인식
                if self.yolo_enabled:
                    # conf: confidence threshold, iou: NMS IoU threshold
                    results = self.yolo_model(undistorted_frame, verbose=False, classes=[0], conf=0.75, iou=0.4)  # classes=[0]: person만
                    all_person_boxes = []
                    
                    # 프레임 크기 가져오기
                    frame_height, frame_width = undistorted_frame.shape[:2]
                    frame_center_x = frame_width // 2
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # 클래스 0: person (COCO 데이터셋 기준)
                            if int(box.cls[0]) == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0])
                                
                                # 신뢰도가 0.75 이상인 경우만 처리 (높은 임계값으로 오탐지 방지)
                                if confidence > 0.75:
                                    # 박스 크기 계산 (가까운 사람일수록 박스가 큼)
                                    box_area = (x2 - x1) * (y2 - y1)
                                    center_x = int((x1 + x2) / 2)
                                    center_y = int((y1 + y2) / 2)
                                    
                                    all_person_boxes.append({
                                        'x1': int(x1), 'y1': int(y1), 
                                        'x2': int(x2), 'y2': int(y2),
                                        'confidence': confidence,
                                        'center_x': center_x,
                                        'center_y': center_y,
                                        'area': box_area
                                    })
                    
                    # 현재 프레임 탐지 여부 기록 (시간적 필터링)
                    current_frame_detected = len(all_person_boxes) > 0
                    self.employee_detect_history.append(current_frame_detected)
                    
                    # 히스토리 크기 유지
                    if len(self.employee_detect_history) > self.HISTORY_SIZE:
                        self.employee_detect_history.pop(0)
                    
                    # 최근 N 프레임 중 70% 이상 탐지되었는지 확인
                    detection_ratio = sum(self.employee_detect_history) / len(self.employee_detect_history)
                    is_stable_detection = detection_ratio >= self.DETECTION_RATIO_THRESHOLD
                    
                    # 가장 큰 박스(가장 가까운 사람) 1명만 선택
                    if all_person_boxes and is_stable_detection:
                        closest_person = max(all_person_boxes, key=lambda p: p['area'])
                        
                        # 위치 판단 (왼쪽/가운데/오른쪽)
                        # 화면을 3등분하여 판단
                        left_threshold = frame_width * 0.4
                        right_threshold = frame_width * 0.6
                        
                        if closest_person['center_x'] < left_threshold:
                            position = 'left'
                            color = (0, 0, 255)  # 빨간색
                        elif closest_person['center_x'] > right_threshold:
                            position = 'right'
                            color = (255, 0, 0)  # 파란색
                        else:
                            position = 'center'
                            color = (0, 255, 0)  # 녹색
                        
                        # 가장 가까운 사람 박스 그리기
                        cv2.rectangle(undistorted_frame, 
                                    (closest_person['x1'], closest_person['y1']), 
                                    (closest_person['x2'], closest_person['y2']), 
                                    color, 3)
                        
                        # 위치 정보 텍스트 표시 (탐지 안정성 비율 추가)
                        text = f'{position.upper()} {closest_person["confidence"]:.2f} ({detection_ratio:.1%})'
                        cv2.putText(undistorted_frame, text, 
                                  (closest_person['x1'], closest_person['y1'] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # 중심점 표시
                        cv2.circle(undistorted_frame, 
                                 (closest_person['center_x'], closest_person['center_y']), 
                                 8, color, -1)
                        
                        # 화면 중앙 기준선 표시
                        cv2.line(undistorted_frame, (frame_center_x, 0), (frame_center_x, frame_height), (128, 128, 128), 1)
                        
                        # 시그널 발생
                        current_time = time.time()
                        if current_time - self.last_marker_emit_time >= self.marker_emit_time:
                            self.person_detected.emit(closest_person['x1'], closest_person['y1'],
                                                    closest_person['x2'], closest_person['y2'],
                                                    closest_person['confidence'], position)
                            self.person_position.emit(position, closest_person['center_x'], closest_person['center_y'])

                            self.last_marker_emit_time = current_time

                # 프레임을 QImage로 변환하여 시그널 송출
                rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()