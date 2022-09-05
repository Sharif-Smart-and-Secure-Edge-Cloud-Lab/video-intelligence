import cv2
import numpy as np

cap = cv2.VideoCapture('rtsp://admin:cameraF1376@10.42.0.20/cam/realmonitor?channel=1&subtype=0')

while(True):
    ret, frame = cap.read()
    imgO = frame
    height = int(imgO.shape[0] /1.5)
    width = int(imgO.shape[1] / 2)
    dim = (width , height)
    img2 = cv2.resize(imgO , dim , interpolation = cv2.INTER_AREA)
    cv2.imshow('frame',img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
