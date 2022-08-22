# Drowsiness
In this project, I implemented a drowsiness detector using Yolo and PyTorch. The main parts are described in the code. The custom model can be used in different locations.
The custom model is in this directory:

```
./yolov5/runs/train/exp#
```

The number of exp shows the latest trained model, and you have your model that can use it.

I want to show how to label captured pictures using the code software mentioned.

## LabelImg
First, open labelImg.py in the LabelImg folder.

After that, you have something like the below picture:

![LabelImg](https://github.com/Sharif-Smart-and-Secure-Edge-Cloud-Lab/video-intelligence/blob/main/Drowsiness/readmePic/LabelImg.JPG)

Choose "**open dir**" from the left bar and change it to "**./data/images.**" Also, choose "**change save dir**" and select "**./data/labels.**"

You can see **Yolo** in the left bar too. If it is not Yolo, change it to Yolo.

Caution: for creating **dataset.yml**, see the yolov5 Github, or you can use the one in the readmePic directory

Reference: 
Nicholas Renotte




#Face detection
another part of the project is to recognize faces in real-time video, which received from an ip-camera.
as a demo, we implemented a code using open-cv.
to introduce a person to this code, we give a picture of him/her. but it is not possible to give more than one picture of a person, but we can do this by naming the picture like: "amin1", "amin2", ...

some pictures of the project:

![LabelImg](https://github.com/Sharif-Smart-and-Secure-Edge-Cloud-Lab/video-intelligence/blob/main/Drowsiness/readmePic/demo1.JPG)

![LabelImg](https://github.com/Sharif-Smart-and-Secure-Edge-Cloud-Lab/video-intelligence/blob/main/Drowsiness/readmePic/demo2.JPG)

![LabelImg](https://github.com/Sharif-Smart-and-Secure-Edge-Cloud-Lab/video-intelligence/blob/main/Drowsiness/readmePic/demo3.JPG)

to get better results, we should use deep learning methods, so i spent last few days learning tensorflow, CNN, ... 




