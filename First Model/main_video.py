import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture('rtsp://admin:cameraF1376@10.42.0.20/cam/realmonitor?channel=1&subtype=0')

kk = 0
while True:
	kk+=1
	ret, frame = cap.read()
	if kk % 5 ==1:	
		imgO = frame
		height = int(imgO.shape[0] / 1.5)
		width = int(imgO.shape[1] / 2)
		dim = (width , height)
		img2 = cv2.resize(imgO , dim , interpolation = cv2.INTER_AREA)
		 
		frame = img2
		# Detect Faces
		face_locations, face_names = sfr.detect_known_faces(frame)
		for face_loc, name in zip(face_locations, face_names):
			y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

			cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

		cv2.imshow("Frame", frame)

		key = cv2.waitKey(1)
    
		print(kk//5)
		
		if key == 27:
			break

cap.release()
cv2.destroyAllWindows()
