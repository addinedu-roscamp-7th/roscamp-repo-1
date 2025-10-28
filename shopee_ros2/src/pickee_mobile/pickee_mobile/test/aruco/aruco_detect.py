import cv2
import pickle
import numpy as np

# 🔹 1. 저장된 카메라 캘리브레이션 데이터 불러오기
with open('camera_calibration.pkl', 'rb') as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# 🔹 2. ArUco 딕셔너리 및 파라미터 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()

# 🔹 3. 카메라 장치 열기
cap = cv2.VideoCapture(2)  # 필요시 인덱스를 2 등으로 변경

# 🔹 마커 한 변의 실제 길이(mm 단위)
marker_length = 50  # 예: 3cm짜리 마커라면 30mm

print("🎥 ArUco 인식 시작 ('q' 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # 🔹 4. 왜곡 보정
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # 🔹 5. 마커 검출
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(frame_undistorted)

    if ids is not None:
        # 🔹 마커 표시
        cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        # 🔹 6. 각 마커의 자세(회전·이동 벡터) 계산
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
            # 축 그리기 (x:빨강, y:초록, z:파랑)
            cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

            # 좌표 출력
            pos = tvec.flatten()
            print(f"🟢 ID {marker_id[0]} 위치(mm): x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")

    # 🔹 7. 화면 표시
    cv2.imshow("ArUco Marker Detection", frame_undistorted)

    # 🔹 8. 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
