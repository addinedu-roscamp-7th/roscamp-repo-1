import cv2
import numpy as np
import os

# 체커보드 내부 코너 개수 (행, 열)
CHECKERBOARD = (7, 6)

# 3D 객체 포인트 준비
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D 포인트
imgpoints = []  # 2D 포인트

# 이미지 폴더 경로
image_dir = './calib_images'
images = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        # 코너 표시
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 카메라 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('카메라 행렬:\n', mtx)
print('왜곡 계수:\n', dist)

# 왜곡 보정 테스트
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow('Undistorted', dst)
    cv2.waitKey(500)

cv2.destroyAllWindows()