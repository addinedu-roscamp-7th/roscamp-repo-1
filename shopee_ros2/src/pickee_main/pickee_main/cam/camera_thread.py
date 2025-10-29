import cv2
import cv2.aruco
import numpy as np
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

class CameraThread(QThread):
    frame_ready = Signal(QImage)
    marker_detected = Signal(int, np.ndarray, np.ndarray)

    def __init__(self, camera_index=2):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.last_marker_emit_time = 0.0 
        self.marker_emit_time = 1.0  # 마커 정보 발행 간격 (초)

        # 아르코 마커 딕셔너리 및 파라미터 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # 카메라 행렬 및 왜곡 계수 (캘리브레이션 결과)
        self.camera_matrix = np.array([
            [2.07267968e+03, 0.00000000e+00, 3.87826573e+02],
            [0.00000000e+00, 1.01210308e+04, 2.19265005e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.dist_coeffs = np.array([[-2.83991246, -23.92614276, 0.03117976, -0.19365304, 8.5383573]])
        self.marker_length = 0.05

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # 영상 왜곡 보정
                h, w = frame.shape[:2]
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
                undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)

                # 아르코 마커 검출 (함수 기반)
                corners, ids, rejected = cv2.aruco.detectMarkers(undistorted_frame, self.aruco_dict, parameters=self.aruco_params)

                if ids is not None:
                    undistorted_frame = cv2.aruco.drawDetectedMarkers(undistorted_frame, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                    )

                    current_time = time.time()

                    if current_time - self.last_marker_emit_time >= self.marker_emit_time:
                        for i, marker_id in enumerate(ids.flatten()):
                            self.marker_detected.emit(marker_id, tvecs[i], rvecs[i])
                            cv2.drawFrameAxes(undistorted_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.2)

                        self.last_marker_emit_time = current_time
                    else:
                        # 3초가 안 지났어도 좌표축은 표시
                        for i, marker_id in enumerate(ids.flatten()):
                            cv2.drawFrameAxes(undistorted_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.2)

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