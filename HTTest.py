import cv2
import time
import HandTrackerModule as htm

cap = cv2.VideoCapture(1)
detector = htm.handDetector()
previousTime = 0
currentTime = 0

tips = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        fingers = []

        # Thumb
        if landmarkList[tips[0]][1] > landmarkList[tips[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1,5):
            if landmarkList[tips[id]][2] < landmarkList[tips[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        count = fingers.count(1)
        cv2.putText(img, str(count), (50,200),cv2.FONT_HERSHEY_PLAIN,
                    3, (255,0,255),3)
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)