import argparse
import time
import numpy as np
import cv2 as cv
import os
import psutil


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()



def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_name(n):
	if n==0:
		msg = 'Amin'
	elif n==1:
		msg = 'Farbod'
	elif n==2:
		msg = 'Elon'
	elif n==3:
		msg = 'Barrack'
	else:
		msg = 'Unknown'
	return msg
		

if __name__ == '__main__':
    timestart = time.time()
    ## [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(args.face_detection_model,"",(480, 640),args.score_threshold,args.nms_threshold,args.top_k)
    ## [initialize_FaceDetectorYN]

    tm = cv.TickMeter()

    img1 = cv.imread('img1.jpg')
    img1Width = int(img1.shape[1]*args.scale)
    img1Height = int(img1.shape[0]*args.scale)
    img1 = cv.resize(img1, (img1Width, img1Height))
        
        
        
    img2 = cv.imread('img2.jpg')
    img2Width = int(img2.shape[1]*args.scale)
    img2Height = int(img2.shape[0]*args.scale)
    img2 = cv.resize(img2, (img2Width, img2Height))
    
		
    img3 = cv.imread('img3.jpg')
    img3Width = int(img3.shape[1]*args.scale)
    img3Height = int(img3.shape[0]*args.scale)
    img3 = cv.resize(img3, (img3Width, img3Height))
    
    
    
    img4 = cv.imread('img4.jpg')
    img4Width = int(img4.shape[1]*args.scale)
    img4Height = int(img4.shape[0]*args.scale)
    img4 = cv.resize(img4, (img4Width, img4Height))
    
    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)
    faces3 = detector.detect(img3)
    faces4 = detector.detect(img4)
    
    
    
    ## [initialize_FaceRecognizerSF]
    recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model,"")
    
    face1_align = recognizer.alignCrop(img1, faces1[1][0])
    face2_align = recognizer.alignCrop(img2, faces2[1][0])
    face3_align = recognizer.alignCrop(img3, faces3[1][0])
    face4_align = recognizer.alignCrop(img4, faces4[1][0])
		
    # Extract features
    face1_feature = recognizer.feature(face1_align)
    face2_feature = recognizer.feature(face2_align)
    face3_feature = recognizer.feature(face3_align)
    face4_feature = recognizer.feature(face4_align)
		
		
    cap = cv.VideoCapture(1)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
    detector.setInputSize([frameWidth, frameHeight])
    #cosine_similarity_threshold = 0.363
    cosine_similarity_threshold = 0.4
    l2_similarity_threshold = 1.128
    count = 0
    remainder = 2
    intersector = 0
    inter_list = [[],[],[]]
    while cv.waitKey(1) < 0:
        if count%remainder == 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv.resize(frame, (frameWidth, frameHeight))
      

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()
            
            #print(faces)
            
            
            faces_list = []
            if faces[1] is not None:
                ## [facerecognizer]
                # Align faces
                for face in faces[1][:]:
                    face_align = recognizer.alignCrop(frame, face)

                    # Extract features
                    face_feature = recognizer.feature(face_align)


                    ## [match]
                    cosine_score1 = recognizer.match(face_feature, face1_feature, cv.FaceRecognizerSF_FR_COSINE)
                    cosine_score2 = recognizer.match(face_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
                    cosine_score3 = recognizer.match(face_feature, face3_feature, cv.FaceRecognizerSF_FR_COSINE)
                    cosine_score4 = recognizer.match(face_feature, face4_feature, cv.FaceRecognizerSF_FR_COSINE)
           
                    #l2_score1 = recognizer.match(face_feature, face1_feature, cv.FaceRecognizerSF_FR_NORM_L2)
                    #l2_score2 = recognizer.match(face_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)
                    #l2_score3 = recognizer.match(face_feature, face3_feature, cv.FaceRecognizerSF_FR_NORM_L2)
                    #l2_score4 = recognizer.match(face_feature, face4_feature, cv.FaceRecognizerSF_FR_NORM_L2)
            
                    cos_list = [cosine_score1,cosine_score2,cosine_score3,cosine_score4]
                    nearest_cosine = max(cos_list)
                    nearest_cosine_arg = cos_list.index(nearest_cosine) 
                    ## [match]
                    msg = 'different identities'
                    if nearest_cosine >= cosine_similarity_threshold:
                        '''and l2_score1 <= l2_similarity_threshold'''
                        msg = (nearest_cosine_arg)
                        
                        #cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                    coords = face.astype(np.int32)
                    cv.putText(frame, get_name(msg), (coords[0] + 6, coords[1]+coords[3] - 6), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                    faces_list.append(get_name(msg))
                    #print('face'+msg)
#####################################
            

            # Draw results on the input image
            os.system("clear")
            print("Up time: {:.2f}".format(time.time()-timestart))
            # CPU and Memory usage
            print('Cpu % used:',psutil.cpu_percent())
            print('Memory % used:', psutil.virtual_memory()[2])
            print("-------------------")
            inter_list[intersector] = faces_list
            intersector+=1
            intersector = intersector%3
            final_list = set(inter_list[0])&set(inter_list[1])&set(inter_list[2])
            if len(faces_list)>0:
                print(faces_list)
            else:
                print("No faces")
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            cv.imshow('Live', frame)
        count = count%remainder
        count+=1
cv.destroyAllWindows()



'''
            msg = 'different identities'
            if l2_score1 <= l2_similarity_threshold:
                msg = 'Amin'
            print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score1, l2_similarity_threshold))
        
        
        
            msg = 'different identities'
            if l2_score2 <= l2_similarity_threshold:
                msg = 'Farbod'
            print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score2, l2_similarity_threshold))
        
''' 
