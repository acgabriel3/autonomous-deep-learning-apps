# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:10:57 2021

@author: gabri

Creation of architecture to develop interatives applications by the mean of an
artificial network - The use here is to overfitting the data

"""

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import cv2
import pandas as pd
import random 

fluxograma = pd.read_csv("data/app_minimo/fluxograma.csv", sep = ";")

shape = (int(970*1/2), int(910*1/2),3)
length = shape[0]*shape[1]*shape[2]
dim = (shape[0],shape[1])
click_examples = 10

#im1 = cv2.imread("data/app_minimo/" + str(fluxograma.tela[0]))
#im1 = cv2.resize(im1, dim, interpolation = cv2.INTER_AREA)
#im1 = im1.flatten()
#im1 = im1.reshape(shape)
#cv2.imshow("a", im1)
#cv2.waitKey(0)

x = np.zeros((len(fluxograma)*click_examples,length + 2))
#y = np.zeros((len(fluxograma)*click_examples,length))
y = []
pos = 0
for i,row in fluxograma.iterrows():
    for j in range(0,click_examples):
        coord1 = random.uniform(int(row.min1), int(row.max1))
        coord2 = random.uniform(int(row.min1), int(row.min2))
        im1 = cv2.imread("data/app_minimo/" + str(row.tela))
        im1 = cv2.resize(im1, dim, interpolation = cv2.INTER_AREA).flatten()
        im1 = im1.flatten()
        coord = np.asarray([coord1,coord2])
        x[pos] = np.append(coord, im1)
        
        im2 = cv2.imread("data/app_minimo/" + str(row.proxima_tela))
        #y[pos] = cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)
        y.append(cv2.resize(im2, dim, interpolation = cv2.INTER_AREA))
        pos = pos + 1

x = x.astype('float32')
y = np.asarray(y)#.astype('float32')

x[:][2:] = x[:][2:]/2000000
        #final_data.append([coord1,coord2,cv2.imread("data/app_minimo/" + str(row.tela)),cv2.imread("data/app_minimo/" + str(row.proxima_tela))])
#x = np.array(x)
#y = np.array(y)

#for i in range(0,len(x)):
#    x[i] = np.asarray(x[i]).astype('float32')

#x = np.asarray(aux)


#y = final_data[:,3]

#for i in range(0,len(y)):
#    y[i] = y[i].flatten()

#n_cols = final_data.shape[1] - 1
model = Sequential()
#model.add(layers.Flatten())
model.add(layers.Dense(500, activation = "sigmoid", input_shape = (length+2,)))
model.add(layers.Dense(int((500)/2), activation = "sigmoid"))
model.add(layers.Dense(int((500)/4), activation = "sigmoid"))
model.add(layers.Dense(int((500)/8), activation = "sigmoid"))
model.add(layers.Dense(int((500)/4), activation = "sigmoid"))
model.add(layers.Dense(int((500)/2), activation = "sigmoid"))
model.add(layers.Dense(length))
model.add(layers.Reshape((shape[1],shape[0],shape[2])))
#para o modelo convolucional:
# input_shape = (1,1,(970,970,3))

#aprender como usar o l1 regularizer
#encoded = layers.Dense(encoding_dim, activation='relu',
#                activity_regularizer=regularizers.l1(10e-5))(input_img)
    
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#model.add(optimizer = "sgd")

model.fit(x, y,
          epochs=100,
          workers = 3)

teste = model.predict(x[158:159])#*255
teste = teste[0].astype("uint8")
cv2.imshow("teste", teste[0].astype("uint8"))
cv2.waitKey(0)

import h5py
model_structure = model.to_json()

with open("data/app_minimo/arquitetura_1_mlp.json",'w') as json_file:
  json_file.write(model_structure)

model.save_weights("data/app_minimo/pesos_1_modelo.json")

teste = x[12][2:]*255
teste = teste.reshape(shape).astype("uint8")
cv2.imshow("teste",teste)
cv2.waitKey(0)


model2 = Sequential()
#model.add(layers.Flatten())
model2.add(layers.Dense(500, activation = "sigmoid", input_shape = (length+2,)))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
#model.add(Dropout(0.5)) #posso pensar em usar dropout aqui, mas primeiro partir para o modelo mais complexo
model2.add(layers.Dense(int((500)), activation = "sigmoid"))
model2.add(layers.Dense(length))
model2.add(layers.Reshape((shape[1],shape[0],shape[2])))

model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#model.add(optimizer = "sgd")

model2.fit(x, y,
          epochs=50,
          workers = 3)

teste = model2.predict(x[0:1])#*255
cv2.imshow("teste", teste[0].astype("uint8"))
cv2.waitKey(0)


model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model2.fit(x, y,
          epochs=50,
          workers = 3)
