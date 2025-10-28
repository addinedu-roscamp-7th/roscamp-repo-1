import cv2
import datetime
import os

# 카메라 장치 열기
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 캡처 저장 폴더 확인 및 생성
save_dir = "./checkerboards"
os.makedirs(save_dir, exist_ok=True)

print("🎥 카메라가 실행되었습니다. 'a'를 눌러 캡처, 'q'를 눌러 종료하세요.")

# 영상 캡처 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 카메라에서 프레임을 가져올 수 없습니다.")
        break

    # 프레임을 화면에 표시
    cv2.imshow("Video", frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF

    # 'a' 키를 누르면 프레임 캡처하여 저장
    if key == ord('a'):
        filename = datetime.datetime.now().strftime(f"{save_dir}/capture_%Y%m%d_%H%M%S.png")
        success = cv2.imwrite(filename, frame)

        if success and os.path.exists(filename):
            print(f"✅ 이미지가 성공적으로 저장되었습니다: {filename}")
        else:
            print(f"⚠️ 이미지 저장 실패: {filename}")

    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        print("🛑 프로그램 종료")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
