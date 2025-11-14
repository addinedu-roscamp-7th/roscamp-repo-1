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

# def lateral_offsets_perp_to_normal(tvec, rvec, up=np.array([0.0, 1.0, 0.0])):
#     """
#     반환:
#       lateral_dist : 법선(정면) 방향에 '수직인' 평면(마커 평면)으로의 이동 거리 '크기'
#       lateral_LR   : 바닥면에서 좌우 한 축(부호 포함)으로의 필요 이동 거리
#       d_normal     : 정면(법선) 거리
#     """
#     t = np.asarray(tvec, dtype=float).reshape(3)
#     R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
#     n = R[:, 2]                       # marker normal in camera frame

#     # 정면(법선) 거리
#     d_normal = float(np.dot(t, n))

#     # 법선 수직(마커 평면) 성분과 그 크기
#     t_parallel = t - d_normal * n
#     lateral_dist = float(np.linalg.norm(t_parallel))

#     # 바닥면 기준 좌우 축(법선과 up에 모두 수직)
#     up = up / np.linalg.norm(up)
#     lr_axis = np.cross(up, n)
#     if np.linalg.norm(lr_axis) < 1e-6:
#         # 법선이 거의 수직일 때 대체축 사용(예: 카메라 z축)
#         fallback = np.array([0.0, 0.0, 1.0])
#         lr_axis = np.cross(fallback, n)
#     lr_axis = lr_axis / np.linalg.norm(lr_axis)

#     # 부호 있는 좌우 오프셋(바닥면 한 축)
#     lateral_LR = float(np.dot(t, lr_axis))

#     return lateral_dist, lateral_LR, d_normal

# 마커 기준 로봇이 전진해야 하는 거리, 좌우 이동해야 하는 거리
def dist_from_xyz_pitch(x, z, pitch_rad):
    # 정면(법선) 거리
    dist_front = x*math.sin(pitch_rad) + z*math.cos(pitch_rad)
    # 바닥면 좌우(부호 포함), 음수 : 카메라 기준 마커가 왼쪽에 있다.
    dist_side = x*math.cos(pitch_rad) - z*math.sin(pitch_rad)

    return dist_front, dist_side




# === Load camera calibration ===
# base_dir = os.path.dirname(os.path.abspath(__file__))
# pkl_path = os.path.join(base_dir, "camera_calibration.pkl")

# with open(pkl_path, 'rb') as f:
#     calib_data = pickle.load(f)

# camera_matrix = calib_data['camera_matrix']
# dist_coeffs = calib_data['dist_coeffs']

camera_matrix = np.array([
    [7.97685154e+02, 0.00000000e+00, 2.82175616e+02],
    [0.00000000e+00, 7.98389022e+02, 2.82054906e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=float)

dist_coeffs = np.array([
    -4.08433569e-01,
     7.75362715e-01,
    -1.58047124e-03,
    -2.69813496e-04,
    -2.79637393e+00
], dtype=float)


# === ArUco settings ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

aruco_params = cv2.aruco.DetectorParameters()

aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshConstant = 7
aruco_params.minMarkerPerimeterRate = 0.02
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.polygonalApproxAccuracyRate = 0.02
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 50


# ✅ Video capture
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # exposure manual
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_GAIN, 1)




marker_length = 50  # mm

print("🎥 ArUco 인식 시작 (q 눌러 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break
    
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
    # th = cv2.adaptiveThreshold(gray, 255, 
    #                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                        cv2.THRESH_BINARY, 11, -1)
    # print(gray)
    ret, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(frame_undistorted)

    if ids is None:
        
        corners, ids, _ = detector.detectMarkers(th)
        if ids is not None:
            print(f'!!!! GRAY !!!!')


    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
            # cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 1.5)

            pos = tvec.flatten()
            x, y, z = pos[0], pos[1], pos[2]

            # ✅ Rodrigues → Euler(roll, pitch, yaw)
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = get_euler_angles(R)
            # pitch = 0
            pitch_rad = math.radians(pitch)
            # aruco_distance = z * math.cos(pitch_rad) + x * math.sin(pitch_rad)
            # aruco_diff = z * math.sin(pitch_rad) + x * math.cos(pitch_rad)

            dist_front, dist_side = dist_from_xyz_pitch(x, z, pitch_rad)

            print(
                # f"🟢 ID {marker_id[0]} | "
                f"x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm | "
                f"roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°"
                f"aruco_distance = {dist_front}"
                f"aruco_diff = {dist_side}"
            )

            h, w = frame_undistorted.shape[:2]

            text1 = f"x={x:.1f} y={y:.1f} z={z:.1f}"
            text2 = f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}"
            text3 = f"front={dist_front:.1f} side={dist_side:.1f}"

            # cv2.putText(frame_undistorted, text1, (10, h - 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 2)

            # cv2.putText(frame_undistorted, text2, (10, h - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 2)

            # cv2.putText(frame_undistorted, text3, (10, h - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)



    cv2.imshow("GRAY", gray)
    cv2.imshow("THRESH", th)
    cv2.imshow("CAM", frame_undistorted)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
