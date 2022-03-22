import cv2
import time
import HandTrackerModule as htm
import streamlit as st


st.title("Finger Counter")
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])


cap = cv2.VideoCapture(0)
detector = htm.handDetector()

tips = [4, 8, 12, 16, 20]
while run:
  success, img = cap.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    for id in range(1, 5):
      if landmarkList[tips[id]][2] < landmarkList[tips[id]-2][2]:
        fingers.append(1)
      else:
        fingers.append(0)

    count = fingers.count(1)
    cv2.putText(img, str(count), (50, 200), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)
  FRAME_WINDOW.image(img)
else:
  st.write('Stopped')
