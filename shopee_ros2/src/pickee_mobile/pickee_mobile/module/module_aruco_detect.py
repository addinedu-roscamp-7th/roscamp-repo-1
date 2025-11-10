import cv2
import pickle
import numpy as np
import os
import math
import rclpy
from geometry_msgs.msg import Pose2D
from shopee_interfaces.msg import ArucoPose
from rclpy.executors import MultiThreadedExecutor

class ArucoPoseEstimator:
    def __init__(self, camera_id=0, marker_length=50, calibration_file="./camera_calibration.pkl"):
        """
        ArUco 마커를 인식하고 6DoF Pose(x, y, z, roll, pitch, yaw)를 계산하는 클래스

        Args:
            camera_id (int): 사용할 카메라 인덱스 (기본: 0)
            marker_length (float): 마커 한 변의 길이 (mm 단위)
            calibration_file (str): 카메라 보정 파일(.pkl) 경로
        """
        # --- 기본 파라미터 ---
        self.camera_id = camera_id
        self.marker_length = marker_length
        self.calibration_file = calibration_file

        # --- 보정 데이터 불러오기 ---
        self.camera_matrix, self.dist_coeffs = self.load_calibration()

        # --- ArUco 설정 ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # --- 카메라 열기 ---
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 카메라(ID={self.camera_id})를 열 수 없습니다.")

        print("✅ ArucoPoseEstimator 초기화 완료")

    # ------------------------------------------------------------------------
    def load_calibration(self):
        """camera_calibration.pkl 파일에서 내·외부 파라미터 불러오기"""
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # pkl_path = os.path.join(base_dir, self.calibration_file)
        # pkl_path = '/home/lim/project/roscamp-repo-1/shopee_ros2/src/pickee_mobile/pickee_mobile/module/camera_calibration.pkl'
        pkl_path = '/home/wonho/tech_research/Shopee/shopee_ros2/src/pickee_mobile/pickee_mobile/module/camera_calibration.pkl'

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"❌ 보정 파일을 찾을 수 없습니다: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            calib_data = pickle.load(f)

        print(f"📁 보정 파일 로드 완료: {pkl_path}")
        return calib_data['camera_matrix'], calib_data['dist_coeffs']

    # ------------------------------------------------------------------------
    def get_euler_angles(self, R):
        """회전 행렬(R)을 오일러 각(roll, pitch, yaw)으로 변환"""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll  = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2(R[1, 0], R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = 0

        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    # ------------------------------------------------------------------------
    def process_frame(self, frame):
        """단일 프레임에서 ArUco 마커 검출 및 Pose 계산"""

        # 왜곡 보정
        frame_undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        # 마커 검출
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        results = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
                cv2.drawFrameAxes(frame_undistorted, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

                pos = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = self.get_euler_angles(R)

                result = {
                    "id": int(marker_id[0]),
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw
                }
                results.append(result)



        return frame_undistorted, results

    # ------------------------------------------------------------------------
    def run(self):
        """실시간 실행 루프"""
        print("🎥 ArUco 마커 인식 시작 (종료: 'q')")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break

            frame_out, markers = self.process_frame(frame)

            for m in markers:
                print(f"🟢 ID {m['id']}")
                print(f"   위치(mm): x={m['x']:.1f}, y={m['y']:.1f}, z={m['z']:.1f}")
                print(f"   회전(°): roll={m['roll']:.1f}, pitch={m['pitch']:.1f}, yaw={m['yaw']:.1f}\n")

            cv2.imshow("ArUco Marker Detection", frame_out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    
    estimator = ArucoPoseEstimator(
        camera_id=2,
        marker_length=50,  # mm 단위
        calibration_file="camera_calibration.pkl"
    )
    estimator.run()





if __name__ == "__main__":
    main()
