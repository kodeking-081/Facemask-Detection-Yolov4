#  Facemask-Detection Using Yolov4:

https://github.com/user-attachments/assets/99464508-06e6-4808-b202-23b3169708fd

# Overview
This project is aimed at building a custom object detector using the YOLO (You Only Look Once) algorithm for precise face mask detection in images. The system can help in ensuring compliance with face mask guidelines at both privata and public places.

# Key Features:
* Utilizes YOLO for real-time object detection.
* Customizes YOLO for face mask detection.
* Training on a custom dataset of images with annotations.
*  High accuracy and efficiency in detecting face masks.

# Dependencies
To run this project, you will need following dependencies:
* OpenCV (cv2)
* Darknet (YOLO)
* Python 3
* Google Colab( for GPU usage)
Incase of CPU based training, you will need to install CUDA and cuDNN (for GPU support) on your device and create a darknet architechture on your CPU.

# Dataset
Images were downloaded from various kaggle dataset and compiled into a single folder. So, I have uploaded a zip file containing dataset.The dataset contains 3 classes: with_mask, without_mask and incorrect_mask.
It consists of 1517, 905 and 1371 images for with_mask class, without_mask and incorrect_mask class respectively.

# GETTING STARTED:
Get a copy of all the code you require at this colab notebook : https://colab.research.google.com/drive/1kIwNFNqZ3y0sYW3zgav4wu2cPlNZATQQ?usp=sharing

## 1.Required Project Directory:
First create a yolov4 folder on your google drive and within that folder, create a subfolder "training".This "training" folder will later contain our weights file obtained while training our mask-detector model.

## 2.Clone the darknet repository:
You can clone the repository from official AlexeyAB github account using "!git clone https://github.com/AlexeyAB/darknet".

## 3.Upload the required files:
You can get all the files at my [GitHub](https://github.com/kodeking-081). Here is the list of files you should upload within your yolov4 folder:
 ### 3(a) Upload the Labeled custom dataset obj.zip file to the yolov4 folder on your drive
 Upload the dataset containing image and its corresponding annotation text file. The annotation file should specify the object class amd bounding box coordinates for the objects in the image. You can create such text file using "labelimg". You can see detailed video for annotating image at [Youtube](https://www.youtube.com/watch?v=1d7u8wTmA80).

 ### 3(b) Create your custom config file and upload it to your drive
 Download the yolov4-custom.cfg file from darknet/cfg directory, make changes to it, and upload it to the yolov4 folder on your drive .

You can also download the custom config file from the official [AlexeyAB Github](https://github.com/AlexeyAB/darknet).

You need to make the following changes in your custom config file:

* Change line batch to batch=64
* Change line subdivisions to subdivisions=16
* Change line max_batches to (classes*2000 but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
* Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
* Set network size width=416 height=416 or any value multiple of 32
* Change line classes=80 to your number of objects in each of 3 [yolo]-layers
* Change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers. So if classes=1 then it should be filters=18. If classes=2 then write filters=21.

### 3(c) Create your obj.data and obj.names files and upload to your drive:
* obj.data:

  
![image](https://github.com/user-attachments/assets/21e37e00-829b-47f7-9724-d1fa572d1239)



* obj.names:


![image](https://github.com/user-attachments/assets/203350b2-6649-4711-bc61-897b93ce5859)


### 3(d) Upload the process.py script file to the yolov4 folder on your drive
To divide all image files into 2 parts. 80% for train and 20% for test.

This process.py script creates the files train.txt & test.txt where the train.txt file has paths to 90% of the images and test.txt has paths to 10% of the images.


## 4. Modify the make file to enable GPU and OPENCV.
* Set GPU and OPENCV to 1.
* Enable CUDNN, CUDNN_HALF and LIBSO by setting them to 1.
After this, save the changes and run the make command to build darknet.Then, clear the cfg folder inside darknet and copy the updated yolov4-custom_cfg file into darknet/cfg folder.
Besides this, you also need to copy the obj.zip file(i.e custom dataset), obj.data and obj.names  inside the darknet/data folder. You also need to copy the process.py file into darknet directory.

# TRAINING:
## Download the yolov4 pre-trained weights file
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

## Train your custom detector


![image](https://github.com/user-attachments/assets/ca5abda3-3f86-42d8-9378-f9854e29108d)


There is high chance that your training might get interrupted due to inactivity or free daily GPU runtime being disconnected on google colab. In such cases, you can always restart training with yolov4-custom_last.weights file saved in training folder.

![image](https://github.com/user-attachments/assets/8a83ec08-ceb8-4f70-82bb-ec28185c6f85)

*Note: It is not possible to retrieve training chart once training is interrupted. So it is important to save your training logs on a separate folder. Go through logparser.py file inside the darknet/scripts folder to do so. Later you can use matplotlib to generate chart from saved training logs.*

## Training Chart:
Darknet itself generates a chart.png for uninterrupted training. But, in my case training got interrupted at 1000 epochs. So, i am uploading training chart at 1000 epoch and a standard training chart i found at [GitHub](https://github.com/MINED30).

<table>
  <tr>
    <td align="center">
      <h3>My Chart</h3>
      <img src="https://drive.google.com/file/d/1xMn0HTGLSuUzODYSBQ4-sQte6LDW5HfT/view?usp=sharing" width="400" />
    </td>
    <td align="center">
      <h3>Standard Chart</h3>
      <img src="https://drive.google.com/file/d/1mxS58WWZJ4NvnFSsGQjjta7hr731Rrst/view?usp=sharing" width="400" />
    </td>
  </tr>
</table>








