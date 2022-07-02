# Landmark-Recognition-App
<div align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;Detecting similar images is a fundamental problem in computer vision: having an initial image, can we find similar images in a large data set? This issue is very important especially in the case of images that contain landmarks, which represent a big part of what people like to photograph. For this reason, in recent years, research in this field has grown, which has led to the emergence of many algorithms capable of detecting similar images in a large data set.
</div>
<br>

<div align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;This paper presents the methodology implemented to perform the task of recognizing and extracting similar images in a mobile application.
</div>
<br>

<div align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The system architecture is composed of two main components, the front-end, represented by an Android application, and the back-end, represented by the API for recognition and extraction of similar images, located on a GPU server. In the application on the phone, the user can take pictures of the landmarks, can send them to the server for recognition, can see information about the photographed landmarks, can locate the images, can sort them and can see images similar to these that were sent from the server. Within the back-end component, the image sent by the user is recognized using a convolutional neural network called MobileNetV3 (available at https://github.com/rwightman/pytorch-image-models), a compact, effective and computationally efficient network, and the encoding resulting from the recognition is used in the unsupervised learning algorithm, which in our case is MiniBatchKMeans, to identify the cluster of images of which it is part, respectively the images with which the given photo looks like. In addition, the API provides the user with geographical information about the image, as well as the Wikipedia page corresponding to the tourist attraction in the image.
</div>
<br>

<div align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The convolutional neural network behind the algorithm for recognizing and detecting similar images, MobileNetV3, was trained on the simplified and cleaned version of the most famous and bulky set of data with tourist objectives, namely Google Landmarks Dataset, available at https://github.com/cvdfoundation/google-landmark, which has 1.6 million of photos of 81313 landmarks from all over the world. For training, we used a set of several configurations, in order to obtain the best possible accuracy of the neural network. Subsequently, we use the best such configuration for the final recognition algorithm in our system.
</div>
<br>
