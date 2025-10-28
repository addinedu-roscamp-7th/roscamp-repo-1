import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    # 체커보드의 차원 정의
    CHECKERBOARD = (7,10)  # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = [] 
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
    images = glob.glob('./checkerboards/*.png')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray,
                                               CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # 코너 그리기 및 표시
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # 카메라 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    # 결과 출력
    print("Camera matrix : \n")
    print(mtx)
    print("\ndist : \n")
    print(dist)
    print("\nrvecs : \n")
    print(rvecs)
    print("\ntvecs : \n")
    print(tvecs)
    
    # 캘리브레이션 결과를 파일로 저장
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    return calibration_data

def live_video_correction(calibration_data):
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기 가져오기
        h, w = frame.shape[:2]
        
        # 최적의 카메라 행렬 구하기
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 왜곡 보정
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # ROI로 이미지 자르기
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # 원본과 보정된 이미지를 나란히 표시
        original = cv2.resize(frame, (640, 480))
        corrected = cv2.resize(dst, (640, 480))
        combined = np.hstack((original, corrected))
        
        # 결과 표시
        cv2.imshow('Original | Corrected', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 이미 캘리브레이션 파일이 있는지 확인
    if os.path.exists('camera_calibration.pkl'):
        print("Loading existing calibration data...")
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print("Performing new camera calibration...")
        calibration_data = calibrate_camera()
    
    # 실시간 비디오 보정 실행
    print("Starting live video correction...")
    live_video_correction(calibration_data)