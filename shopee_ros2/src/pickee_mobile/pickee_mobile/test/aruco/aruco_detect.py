import cv2
import pickle
import numpy as np
import os
import math

# ✅ Euler 변환 함수 (Rodrigues → roll, pitch, yaw)
def get_euler_angles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# === Load camera calibration ===
base_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(base_dir, "camera_calibration.pkl")

with open(pkl_path, 'rb') as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# === ArUco settings ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(2)
marker_length = 50  # mm

print("🎥 ArUco 인식 시작 (q 눌러 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(frame_undistorted)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
            cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

            pos = tvec.flatten()
            x, y, z = pos[0], pos[1], pos[2]

            # ✅ Rodrigues → Euler(roll, pitch, yaw)
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = get_euler_angles(R)

            print(
                f"🟢 ID {marker_id[0]} | "
                f"x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm | "
                f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"
            )

    cv2.imshow("ArUco Marker Detection", frame_undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
