#  Facemask-Detection Using Yolov4:


![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/webcambestweight.png)


https://github.com/user-attachments/assets/99464508-06e6-4808-b202-23b3169708fd

Original Video: Cottonbro by Pexels.

# Overview
It is a deep learning project utilizing the YOLOv4 object detection algorithm to train a detection model for precise face mask detection in images, videos and live webcam video. Transfer learning is applied by using a pretrained convolutional neural network (CNN) as the base network for YOLOv4. The model is then configured for training on a new dataset by specifying anchor boxes and defining new object classes. The system can help in ensuring compliance with face mask guidelines at both private and public places.

# Key Features:
* Utilizes YOLO for real-time object detection.
* Customizes YOLO for face mask detection.
* Training on a custom dataset of images with annotations.
* High accuracy and efficiency in detecting face masks.

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


# Model Structure:
CSPDarknet -53


![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/darknet53.jpg)


YOLOv4 builds upon its predecessors with several key architectural advancements to improve both performance and efficiency, especially in real-time object detection tasks. YOLOv4 utilizes CSPDarknet53 as its backbone, which is a more efficient version of Darknet53 and introduces Cross-Stage Partial (CSP) connections to enhance gradient flow and reduce redundant computations. This backbone consists of 53 convolutional layers, similar to Darknet53, but with the added benefits of CSP to improve the flow of information between layers. YOLOv4 incorporates skip connections and residual blocks to improve training stability and allow deeper networks without sacrificing performance. Unlike YOLOv3, which suffered from limitations in detecting small objects, YOLOv4 integrates several advanced components such as Spatial Pyramid Pooling (SPP) and Path Aggregation Network (PANet) to enhance feature extraction at multiple scales, significantly improving small object detection. Furthermore, YOLOv4 leverages mosaic data augmentation and self-adversarial training (SAT) to further boost its generalization and robustness to diverse datasets. By combining these elements, YOLOv4 achieves high accuracy and real-time performance, making it one of the most efficient object detection models to date.


## Why 161 layers?

YOLOv4's 161 layers were selected to balance computational efficiency and model depth. The CSPDarknet53 backbone (53 layers), the SPP (Spatial Pyramid Pooling) and PANet (Path Aggregation Network) necks, and the detection head are some of the components that make up the network. YOLOv4 can capture more intricate hierarchical features and better multi-scale object representations because to the improved depth, which improves detection accuracy, particularly for small and obscured objects. Real-time object detection depends on improved feature aggregation, which is another benefit of the more layers. To guarantee efficient training without excessive computational cost or overfitting, the design is optimised with residual connections and skip connections, which makes 161 layers the best option for striking a balance between efficiency and performance in real-time settings.


# GETTING STARTED:
Get a copy of all the code you require at this colab notebook : https://colab.research.google.com/drive/1kIwNFNqZ3y0sYW3zgav4wu2cPlNZATQQ?usp=sharing

## 1.Required Project Directory:
First create a yolov4 folder on your google drive and within that folder, create a subfolder "training".This "training" folder will later contain our weights file obtained while training our mask-detector model.

## 2.Clone the darknet repository:
You can clone the repository from official AlexeyAB github account using
"!git clone https://github.com/AlexeyAB/darknet".

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
Darknet itself generates a chart.png for uninterrupted training. But, In my case training got interrupted at 1000 epochs. So, I am uploading training chart at 1000 epoch and a standard training chart obtained from [GitHub](https://github.com/MINED30).

<table>
  <tr>
    <td align="center">
      <h3>My Chart</h3>
      <img src="https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/chart1000.png" width="400" />
    </td>
    <td align="center">
      <h3>Standard Chart</h3>
      <img src="https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/standardChartyolo.png" width="400" />
    </td>
  </tr>
</table>

## Evaluation


### Average precision:
* For yolov4-custom_1000.weights:

<table>
 <thead>
  <td>class_id</td>
  <td>class_name</td>
  <td>TP</td>
  <td>FP</td>
  <td>AP</td>
 </thead>
 <tbody>
  <tr>
   <td>0</td>
   <td>with_mask</td>
   <td>163</td>
   <td>38</td>
   <td>95.59%</td>
  </tr>
  <tr>
   <td>1</td>
   <td>without_mask</td>
   <td>89</td>
   <td>5</td>
   <td>99.75%</td>
  </tr>
  <tr>
   <td>2</td>
   <td>incorrect_mask</td>
   <td>135</td>
   <td>42</td>
   <td>96.69%</td>
  </tr>
 </tbody>
</table>



* For yolov4-custom_2000.weights:
  
<table>
 <thead>
  <td>class_id</td>
  <td>class_name</td>
  <td>TP</td>
  <td>FP</td>
  <td>AP</td>
 </thead>
 <tbody>
  <tr>
   <td>0</td>
   <td>with_mask</td>
   <td>187</td>
   <td>6</td>
   <td>99.91%</td>
  </tr>
  <tr>
   <td>1</td>
   <td>without_mask</td>
   <td>91</td>
   <td>2</td>
   <td>100%</td>
  </tr>
  <tr>
   <td>2</td>
   <td>incorrect_mask</td>
   <td>137</td>
   <td>0</td>
   <td>100%</td>
  </tr>
 </tbody>
</table>


* For yolov4-custom_best.weights:
  
*These weights have achieved the best performance based on the evaluation metric*

 <table>
 <thead>
  <td>class_id</td>
  <td>class_name</td>
  <td>TP</td>
  <td>FP</td>
  <td>AP</td>
 </thead>
 <tbody>
  <tr>
   <td>0</td>
   <td>with_mask</td>
   <td>163</td>
   <td>38</td>
   <td>95.59%</td>
  </tr>
  <tr>
   <td>1</td>
   <td>without_mask</td>
   <td>89</td>
   <td>5</td>
   <td>99.75%</td>
  </tr>
  <tr>
   <td>2</td>
   <td>incorrect_mask</td>
   <td>135</td>
   <td>42</td>
   <td>96.69%</td>
  </tr>
 </tbody>
</table>


### F1-score and Average IoU:

<table>
 <thead>
  <th>weights</th>
  <th>precision</th>
  <th>recall</th>
  <th>F1-score</th>
  <th>TP</th>
  <th>FP</th>
  <th>FN</th>
  <th>average IoU</th>
 </thead>
 <tbody>
  <tr>
   <td>1000</td>
   <td>0.82</td>
   <td>0.93</td>
   <td>0.87</td>
   <td>387</td>
   <td>85</td>
   <td>28</td>
   <td>61.80%</td>
   
  </tr>
  <tr>
    <td>2000</td>
    <td>0.98</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>415</td>
    <td>8</td>
    <td>0</td>
    <td>81.78%</td>
  </tr>
  <tr>
    <td>best</td>
    <td>0.82</td>
    <td>0.93</td>
    <td>0.87</td>
    <td>387</td>
    <td>85</td>
    <td>28</td>
    <td>61.80%</td>
  </tr>
 </tbody>
</table>

<h3 style="color: #FFFF00;">Mean Average Precision(mAP):</h3>


<table>
  <thead>
     <tr>
      <th>Epoch</th>
      <th>1000</th>
      <th>2000</th>
      <th>3000</th>
      <th>4000</th>
      <th>5000</th>
      <th>6000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mAP Value</td>
      <td>97.3421%</td>
      <td>99.9713%</td>
      <td>99.9866%</td>
      <td>99.9943%</td>
      <td>99.9991%</td>
      <td>99.9981%</td>
    </tr>
  </tbody>
</table>

The mAP values during first 1000 epochs were promising , but after that the model appears to have started overfitting. Due to this reason, I used yolov4-custom_1000 weights for training evaluation metrics and testing purpose.

*Note: I have uploaded the weights file in yolov4/training folder.*


# TESTING

Before testing the model on different inputs, we need to define a helper function "imshow.py" to load and display image using opencv and matplotlib.
Following changes should be made to the yolov4-custom_cfg.file:
* Change "batch" to 1
* Change subdivisions to 1

A simple Flask based UI is dseigned to test the model:
![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/Screenshot%202025-02-04%20173731.png)

## Run detector on an Image:
Run detector on an image using:
![image](https://github.com/user-attachments/assets/e1a9d0a3-6783-457d-9e2c-4bd677196e28)


Output:
* Using yolov4-custom_1000.weights:

![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/mewithmask1k.png)


* Using yolov4-custom_2000.weights:

![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/withoutmask2k.png)




## Run detector on webcam image:

To test mask detector model on webcam image , use webcamdetect.py.

Output:

* Using yolov4-custom_best.weights:

![image](https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/webcambestweight.png)




## Run detector on Video:
Use the information on videoDetect.txt for running the detector on video inputs.

Output:

https://github.com/user-attachments/assets/99464508-06e6-4808-b202-23b3169708fd


## Run detector on live webcam:
Source code is available from theAIGuysCode [Github](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/blob/master/yolov4_webcam.ipynb).

*Adjust for your custom YOLOv4 trained weights, config and obj.data files*

Output:

https://github.com/kodeking-081/Facemask-Detection-Yolov4/blob/main/images/livewebcam.mp4


# Contributing

Contributions are welcome! Please open an issue or create a pull request if you have any suggestions or improvements.







  












