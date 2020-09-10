# Face Identification using Tensorflow Conv2d 
The idea of the project is to identify the faces in the image. Face recognition is the now being used daily applications like mobile unlock, attendance system, cctv, crowd monitoring etc. In this project, I have used MTCNN to extract the faces from both datasets, test images. Tesorflow Convolutional layers is used to extract the features from the image, Dense layer is used to classify the image. 

## Features
## Folder Structure
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

**bold****italic** face_classification.h5 will be created after running create_model.py. unable to uploade due to file size restriction
