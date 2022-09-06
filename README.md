# video-intelligence

In this project, i tried to implement an efficient face recognition for edge devices.\
## Device installation and camera
If you are using an ARM-based device, on of the good choices as OS, is Armbian OS, which is based on ubuntu.



First model:\
This model is implemented with help of Amin Khodaverdian. this model uses Haarcascade and opencv, and uses a few pictures to learn somebody's face.
The performance wasn't satisfing, e.g. it had many false positive face detections.\
\
Second model:\
In this method, we've used dlib library, which is a Resnet-34 basd Neural network. Face detection and recognition accuracy was good and acceptable,
but in difficult scenarios, it may not detect and recognize a face. also its performance may drop in recognition of a face with mask.\
\
Third model:\
In this model, we implement a Resnet-10 based model. this model performs very well in almost any scenario, but it has two major drawbacks:\
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
### Model 2 testing:

### Model 3 testing:
