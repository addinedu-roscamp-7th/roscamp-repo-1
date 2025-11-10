import cv2

cap1 = cv2.VideoCapture(4)

while cap1.isOpened():
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    cv2.imshow('Camera 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()