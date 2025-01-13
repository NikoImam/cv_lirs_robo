import cv2
import time

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
thickness = 2
color = (100, 100, 100)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

face_min_size = (30, 30) # Минимальный размер лиц
face_max_size = (300, 300) # Максимальный размер лиц

time_prev = 0
c = 0
fps_txt_prev = ''
fs = []

while True:
    time_current = time.time()
    time_difference = time_current - time_prev
    time_prev = time_current
    
    fps = int(1 / time_difference)

    ret, frame = cap.read()
    fs.append(fps)


    if c >= 5:
        fps_txt = f'{fps}'

        
        fps_txt_prev = fps_txt
        c = 0
    c += 1
    cv2.putText(frame, fps_txt_prev, (5, 35), font, fontScale, (0, 255, 255), thickness)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=face_min_size, maxSize=face_max_size)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eye_min_size = (15, 15) # Минимальный размер глаз
        eye_max_size = (50, 50) # Максимальный размер глаз
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=eye_min_size, maxSize=eye_max_size)

        if len(eyes) < 2:
            cv2.putText(frame, f'Открой глаза', (25, 380), font, fontScale, (100, 255, 50), thickness)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            smile_min_size = (50, 50) # Минимальный размер улыбок
            smile_max_size = (150, 150) # Максимальный размер улыбок

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'Улыбнись', (25, 420), font, fontScale, (70, 90, 255), thickness)


    cv2.imshow('Be happy', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

sum = 0
count = 0

for f in fs:
    sum += f
    count += 1

print(f'Average FPS={sum/count}')
