from ultralytics import YOLO
import cv2
import datetime

# 모델 불러오기 (사전 학습된 모델 or 학습된 모델 경로)
model = YOLO("./20251021_iou0.75.pt")  # or "runs/detect/train/weights/best.pt"

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0번 카메라


# # ✅ 저장할 비디오 설정
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # FPS가 0일 경우 기본값 30
# # ✅ 저장 파일 이름 (날짜+시간 기반)
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = f"output_{current_time}.mp4"
# # ✅ 저장 VideoWriter 객체 생성
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID'
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(source=frame, conf=0.8, iou=0.9)

    for result in results:
        # ✅ 예측된 프레임 가져오기 (마스크 포함된 이미지)
        im_array = result.plot()  # 바운딩 박스 + 세그멘테이션 마스크 시각화됨
        cv2.imshow('',im_array)

        if result.masks is None:
                continue
        
        # # ✅ 비디오 파일로 프레임 저장
        # out.write(im_array)

        for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls)
                class_name = result.names[class_id]
                bbox = box.xyxy[0].tolist()
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(box.conf),
                    'polygon': len(mask.tolist()),
                    'bbox': [int(coord) for coord in bbox]
                }
                print(detection)

    # ❌ 종료 조건: ESC 키
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
# out.release()  # ✅ 저장 종료
cv2.destroyAllWindows()