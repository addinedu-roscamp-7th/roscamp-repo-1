import cv2
import cv2.aruco
import numpy as np

# 카메라 행렬 및 왜곡 계수 (캘리브레이션 결과)
CAMERA_MATRIX = np.array([
    [2.07267968e+03, 0.00000000e+00, 3.87826573e+02],
    [0.00000000e+00, 1.01210308e+04, 2.19265005e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
DIST_COEFFS = np.array([[-2.83991246, -23.92614276, 0.03117976, -0.19365304, 8.5383573]])
MARKER_LENGTH = 0.05 # 마커의 실제 한 변의 길이 (미터 단위, 예: 0.05m)

# 아르코 마커 딕셔너리 및 파라미터 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# USB 카메라 열기 (장치 인덱스 2)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print('카메라를 열 수 없습니다.')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임을 읽을 수 없습니다.')
        break
    
    # 영상 왜곡 보정
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)

    # 아르코 마커 검출
    corners, ids, rejected = aruco_detector.detectMarkers(undistorted_frame)

    # 마커가 검출된 경우
    if ids is not None:
        # 검출된 마커 표시
        undistorted_frame = cv2.aruco.drawDetectedMarkers(undistorted_frame, corners, ids)

        # 마커의 자세(위치, 회전) 추정
        rvecs = []
        tvecs = []
        for marker_corners in corners:
            # solvePnP로 마커의 자세 추정
            obj_points = np.array([
                [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
                [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
                [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
                [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]
            ], dtype=np.float32)

            img_points = marker_corners[0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(np.zeros((3, 1)))
                tvecs.append(np.zeros((3, 1)))

        for i, marker_id in enumerate(ids.flatten()):
            # tvecs: 마커 중심의 카메라 좌표계 위치 [x, y, z]
            # rvecs: 마커의 회전 벡터
            print(f'ID: {marker_id}, 위치: {tvecs[i].flatten()}, 회전: {rvecs[i].flatten()}')

            # 마커의 좌표축을 영상에 표시
            cv2.drawFrameAxes(undistorted_frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], MARKER_LENGTH * 0.1)

    # 영상 출력
    cv2.imshow('USB Camera', undistorted_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()