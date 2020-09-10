import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from mtcnn.mtcnn import MTCNN
import cv2

#Global variables and paths
path=sys.path[0]+"/"

#Reading data from face.npz file saved from face_detection.py
data = np.load(path+"faces.npz")
X=data['arr_0']
y_classes=data['arr_1']

#getting the categorical int from y_classes
y=pd.Series(y_classes, dtype='category').cat.codes.values
X=X/200

#Tesorflow models & layers, Earlystopping to stop the training based on validation loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Flatten, Dense, Dropout, MaxPool2D, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

#model creation. CNN 2D & Dense layer
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=(X[0].shape), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=(X[0].shape), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',input_shape=(X[0].shape), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Conv2D(128,kernel_size=(3,3),padding='same',input_shape=(X[0].shape), activation='relu'))
#model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
#model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(64,activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(set(y)),activation='softmax'))

#defining the call back with patience 2 
callback = EarlyStopping(monitor='val_loss', patience=2)

#compiling the model with categorical loss function and optimizer. 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#using sklearn to split the train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

#train the model with validation data and call back function 
model.fit(X_train, y_train,epochs=30, verbose=2, validation_data=(X_test, y_test), callbacks=[callback])
print("model compiled and trained")

model.save(path+'face_classification.h5')

print("model saved")