import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sys, os
from mtcnn.mtcnn import MTCNN
import cv2

#Global variables & MTCNN detector

detector = MTCNN()

#result = detector.detect_faces(image)

m_path =sys.path[0]+"/"
path = m_path+"data_set/train/"
img_size = (200,200)
data_x = []
y=[]

#extract function to extract the face from image. 
def extract_face(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect_faces(img)
    x,y,x1,y1 =res[0]['box']
    out_image = img[y:y+y1,x:x+x1]    
    out_image = cv2.resize(out_image, (200,200))
    print(file_name + "  completed ......" )
    return out_image

#function to read the folder and extract the images from each folder and update the image array
def load_data(path):
    for i in os.listdir(path):
        for sub_dir in os.listdir(path+"/"+i):
            print("Reading images from: "+i+" file name : " + sub_dir)
            #print(path) 
            dir_path = path+i+'/'+sub_dir
            #print(dir_path)
            data_x.append(extract_face(dir_path))
            #data_x.append(extract_face(dir_path))
            y.append(i)
    print("Face detection completed .......")
    return np.asarray(data_x),np.asarray(y)
            

#extract face image is stored in image array and saved as faces.npz compressed format.     
print(path) 
X, y = load_data(path)
np.savez_compressed(path+"faces.npz",X,y)
print("Face_details updated.....")