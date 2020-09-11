# Face Identification using Tensorflow Conv2d 
The idea of the project is to identify the faces in the image or set of images in a folder using simple Tensorflow CNN architecture. Face recognition is being used in day to day applications like mobile authentication, attendance system, cctv, crowd monitoring etc. In this project, I have used MTCNN to extract the faces from both datasets, test images. Tesorflow Convolutional layers is used to extract the features from the image, Dense layer is used to classify the image. 

Project will create a Tensorflow model using the training dataset images categoried by folders. model will predict the faces in the given images (test folder) and output will be placed in the output folder. 

## Features
## Folder Structure

* Face Identification
  - data_set
      - train
        - Image category Folder 1
        - Image category folder 2
        - Image category folder 3
  - output
  - test


1.	Data_set/train – folder store training images segregated by folders. Ex: each category should be differentiated by folder
2.	Output  - folder to store the output images. 
3.	Test – folder where the test images are placed. 
4.	Classify.py – Python file executed to classify the images from test folder and place the output in the output folder. 
5.	Create_model.py – python file is executed to create the model based the folders extracted from the train folder images. 
6.	Face_detection.py – python file is executed to extract the faces from the training (train) folder and save the images in array format. 
7.	Face_classification.h5 – tensorflow model created using the training images. 
8.	Faces.npz – image array and classification array. Images are the faces extracted from the training folder images. 
9.	Requirment.txt – modules required to run the project
10.	Readme.md – readme file about the project. 

***face_classification.h5 will be created after running create_model.py. unable to upload due to file size restriction***

## Requirments
  - [x] Tensorflow 2.0
  - [x] Sklearn 
  - [x] MTCNN
  - [x] Opencv
  - [x] Matplotlib
  - [x] sys, os

## MTCNN
Implementation of the MTCNN face detector for Keras in Python3.4+. It is written from scratch, using as a reference the implementation of MTCNN from David Sandberg (FaceNet’s MTCNN) in Facenet. It is based on the paper Zhang, K et al. (2016) [ZHANG2016]. 
Refer: https://pypi.org/project/mtcnn/

## Tensorflow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.[4] It is used for both research and production at Google. TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache License 2.0 on November 9, 2015
Refer:  https://www.tensorflow.org/api_docs/python/tf

## Convolutional Nerual Network
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

## Features
1. New data set can be added under /train folder with with appropriate folder name
2. Any number or classification can be included, by running the create_model.py Tensorflow model will be created. 

## Limitations/Improvement required
1. Training dataset - Images should be good quality, Images should have single image to categorize it accordingly. 
2. Model has 87% confidence level on the given dataset, any new addition in the dataset might need to create the model again. 
3. Negative scenarios like testing with radom images is not considered. as the categorical will work only based on give datasets. 




