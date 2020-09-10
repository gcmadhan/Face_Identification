
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sys, os
from mtcnn.mtcnn import MTCNN
import cv2
#print(sys.path[0])

#Global Variables, MTCNN detector, system path
detector=MTCNN()
m_path=sys.path[0]+"/"
data = np.load(m_path+"faces.npz")
X=data['arr_0']
y_classes=data['arr_1']
classes = np.unique(y_classes)


#Extract face coordinate function to extract the fact from given image and also the coordinates to draw rectangel in output
def extract_face_coordinate(img, res):
    out_image=np.zeros((len(res),200,200,3), dtype=np.int32)
    face_coor = np.zeros((len(res),4), dtype=np.int32)
    for i in range(len(res)):
        x,y,x1,y1 =res[i]['box']
        height = y+y1
        width = x+x1
        out = img[y:height,x:width]
        out_image[i] = cv2.resize(out, (200,200))
        face_coor[i] = x,y,width, height
        #plt.imshow(out_image[i])
        print(face_coor[i])
    return out_image, face_coor

#extract face to read image and to extact the coordinates for each faces in given file and returs image array and coordinates

def extract_face(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.filter2D(img, -1, kernel)
    res = detector.detect_faces(img)
    print(len(res))
    faces=[]
    face_coord=[]
    #if len(res)>0:
    a = extract_face_coordinate(img, res)
    faces.extend(a[0])
    face_coord.extend(a[1])
    print(file_name + "  completed ......" )
    
    return np.asarray(faces), np.asarray(face_coord)

#load face classification model we created using create_model.py
model = load_model(m_path+"face_classification.h5")


#main function reads the test path and extract the face details
if __name__=='__main__':
    path =m_path+"test/"
    for f in os.listdir(path):
        f_path = path+f
        img_details = extract_face(f_path)
        img = img_details[0]
        img_coor= img_details[1]
        #print(img_coor[0][1])
        # #img = img.reshape(1,200,200,3)
        
        # extracted image details sent to model.predict to predict the image classification. 
        pre = np.argmax(model.predict(img),axis=-1)
        name_image = cv2.imread(f_path)
        name_image = cv2.cvtColor(name_image, cv2.COLOR_BGR2RGB)
        #for loop to draw the rectangle in output image and save it in output folder
        for i in range(len(pre)):
            name = classes[pre[i]]
            cv2.rectangle(name_image, (img_coor[i][0],img_coor[i][1]),(img_coor[i][2],img_coor[i][3]), (0,255,0),2)
            cv2.putText(name_image, name, (img_coor[i][0]-10,img_coor[i][1]-10),cv2.FONT_HERSHEY_PLAIN, color=(0,255,0),fontScale=2, thickness=2)
            print("person is : " + classes[pre[i]])
        plt.imsave(m_path+"output/"+f,name_image)



