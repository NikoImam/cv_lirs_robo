import cv2
import sys

image = cv2.imread('images/ph2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(gray, 150, 240, cv2.THRESH_BINARY_INV)

filtered = cv2.GaussianBlur(thresh, (15, 25), 0)

contours, hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

maxA = 0
minA = sys.maxsize
count = 0

maxX = 0
maxY = 0

minX = 0
minY = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if(area > 30000):
        count += 1
        
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(image, (cX, cY), 10, (255, 50, 0), -1)

        cv2.drawContours(image, contour, -1, (0, 255, 0), 10)

        if area > maxA:
            max = area
            maxX = cX
            maxY = cY
            
        if area < minA:
            minA = area
            minX = cX
            minY = cY


countText = f'Objects count: {count}'
maxCord = f'Biggest object: x={maxX}; y={maxY}'
minCord = f'Smallest object: x={minX}; y={minY}'

txts = [countText, maxCord, minCord]

cv2.rectangle(image, (750, 710), (1500, 870), (115, 255, 201), thickness=cv2.FILLED)

for i in range(len(txts)):
    pos = (770, 700 + 50 * (i + 1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (100, 100, 100)
    thickness = 2

    cv2.putText(image, txts[i], pos, font, fontScale, color, thickness)


cv2.imshow('Happy New Year!!!', image)
cv2.waitKey(0)
cv2.destroyAllWindows()