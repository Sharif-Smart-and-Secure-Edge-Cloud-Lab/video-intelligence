import cv2
import face_recognition
import time

img = cv2.imread("images/Amin.jpg")
height = int(img.shape[0] / 2)
width = int(img.shape[1] / 2)
dim = (width , height)
img = cv2.resize(img , dim , interpolation = cv2.INTER_AREA)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]


# Load Camera
cap = cv2.VideoCapture('rtsp://admin:cameraF1376@10.42.0.20/cam/realmonitor?channel=1&subtype=0')

while True:
    ret, frame = cap.read()
    imgO = frame
    height = int(imgO.shape[0] / 3)
    width = int(imgO.shape[1] / 4)
    dim = (width , height)
    img2 = cv2.resize(imgO , dim , interpolation = cv2.INTER_AREA)
    #print(img2.shape)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding2)
    print("Result: ", result)

    #cv2.imshow("Img", img)
    cv2.imshow("Img 2", img2)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
