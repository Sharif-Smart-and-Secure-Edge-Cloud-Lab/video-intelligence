# Video-Intelligence

In this project, i implemented an efficient face recognition for edge devices.
## How to run models?!
in folder of each model, we have only one python code, for example for model 3, we have dnnRec.py, just run the file:
```
    python3 dnnRec.py 
```
### Requirements
```
numpy
dlib
opencv
```
## Device installation and camera
If you are using an ARM-based device, on of the good choices as OS, is Armbian OS, which is based on ubuntu.\
first, we were using IPcamera, but the camera had delay,by changing the camera to a simple USB camera, delay was improved, and decreased from 5-6 seconds to 1-2 seconds:\
![Hardware setup](/img5.jpg "Hardware setup")
First model:\
This model is implemented with help of Amin Khodaverdian. this model uses Haarcascade and opencv, and uses a few pictures to learn somebody's face.
The performance wasn't satisfing, e.g. it had many false positive face detections.\
\
Second model:\
In this method, we've used dlib library, which is a Resnet-34 basd Neural network. Face detection and recognition accuracy was good and acceptable,
but in difficult scenarios, it may not detect and recognize a face. also its performance may drop in recognition of a face with mask.\
\
Third model:\
In this state of the art model, we used an implementation of a Resnet-10 based model. good parts of this model is using a git repository, which is the official implementation of paper.

this model performs very well in almost any scenario, but it has two major drawbacks:\
    a) It is hungery for CPU usage!\
    b) False positive rate is higher than second model.\
    
 But in this method we have much better face recognition with mask, even sometimes with mask and sunglasses at same time!\
 


A useful webpage for comparing differeent models, which proves the dominanc of dnn method of Opencv, compared to Haarcascade, and dlib library:\
https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/\

![comparison](/img1comp.jpg "dnn dlib haarcascade comparison")

![comparison](/img2comp.jpg "dnn dlib haarcascade comparison")

![comparison](/img3comp.jpg "dnn dlib haarcascade comparison")

\As we see, Opencv's dnn has much better performance with trade-off of higher CPU usage.

## Results of tests:
### Model 1 testing:
![comparison](/img6.png "dnn dlib haarcascade comparison")

### Model 2 testing:
Detecting and recognizing multiple faces:\
![comparison](/img6.png "Model2")
Distinguishing similar faces from each other:\
![comparison](/img7.png "Model2")
Recognizing faces with mask:\
![comparison](/img8.png "Model2")

### Model 3 testing:
This model is implemented using OpenCV Face Detection and Face Recognition(ONNX format) modules.
Benchmarks of the model:
![comparison](/img11.png "Model2")
YuNet(Face detection) is a light-weight, fast and accurate face detection model, which achieves 0.834(AP_easy), 0.824(AP_medium), 0.708(AP_hard) on the WIDER Face validation set.

[YuNet repository](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)

SFace(Face recognition): Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
[SFace repository](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface)

A small demo of model:\
![comparison](/Model3Test.gif "dnn dlib haarcascade comparison")\
As we see, it can recognize a known face, from different angles.\
The model can detect and recognize faces from long distances, easily!\
![comparison](/img10.png "dnn dlib haarcascade comparison")
