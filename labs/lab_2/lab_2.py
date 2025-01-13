import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
thickness = 2

video_path = 'videos/catball.mp4'
cap = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # для мяча
    low = np.array([20, 110, 110])
    high = np.array([40, 255, 255])
    mask = cv2.inRange(hsv_image, low, high)

    # для реального видео
    # low = np.array([20, 110, 110])
    # high = np.array([40, 255, 255])
    # mask = cv2.inRange(hsv_image, low, high)

    contours, hierarchy = cv2.findContours(mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE)
    # только внешние контуры, сохраняются все точки контура
    count = 0

    cv2.rectangle(frame, (0, 0), (450, 100), (115, 255, 201), thickness=cv2.FILLED)

    if len(contours) == 0:
        cv2.putText(frame, 'No object detected', (35, 50), font, fontScale, (200, 200, 50), thickness)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            count += 1
        # все контуры, цвет (255,0,255), толщина 2,
        
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cX, cY), 10, (255, 50, 0), -1)

            center = f'Center: x={cX}; y={cY}'

            cv2.putText(frame, center, (35, 50), font, fontScale, (200, 200, 50), thickness)

        cv2.drawContours(frame, contour, -1, (255,0,255), 3)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
