# -*- coding: utf-8 -*-
"""

Training an RNN network to acomplish the same task
now with just one surroundig layer (like demanded in Robust learning)

@author: gabri
"""
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import random 


fluxograma = pd.read_csv("data/app_minimo/fluxograma.csv", sep = ";")
#fluxograma = fluxograma[0:7]  #more simple train

#More complex train, with very distinc screens
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.sample(frac=1).reset_index(drop=True)

shape = (int(970*1/8), int(910*1/8),3)
length = shape[0]*shape[1]*shape[2]
dim = (shape[0],shape[1])
dim2 = (shape[0]*4,shape[1]*4)
click_examples = 20


xclick = []
xim = []
#y = np.zeros((len(fluxograma)*click_examples,length))
y = []
pos = 0
for i,row in fluxograma.iterrows():
    for j in range(0,click_examples):
        coord1 = random.uniform(int(row.min1), int(row.max1))
        coord2 = random.uniform(int(row.min1), int(row.min2))
        im1 = cv2.imread("data/app_minimo/" + str(row.tela))
        im1 = np.histogram(im1.flatten(), 256, [0, 256])
        coord = np.asarray([coord1,coord2])
        xclick.append(coord)
        xim.append(im1[0])
        
        im2 = cv2.imread("data/app_minimo/" + str(row.proxima_tela))
        #y[pos] = cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)
        y.append(cv2.resize(im2, dim2, interpolation = cv2.INTER_AREA))
        pos = pos + 1

xim = np.asarray(xim)#.astype('float32')
xclick = np.asarray(xclick)
y = np.asarray(y)#.astype('float32')

array = range(0,len(xim))
sample = random.sample(array, len(array) )

xim = xim[sample]
xclick = xclick[sample]
y = y[sample]

#RNN Model
Gmodel = Model()
Gmodel = layers.SimpleRNN(100, input_shape = (277,), return_sequence = False)(Gmodel)
Gmodel = layers.LeakyReLu(0.2)(Gmodel)
Gmodel = layers.Dense(50)(Gmodel)
Gmodel = layers.LeakyReLu(0.2)(Gmodel)
Gmodel = layers.Dense(277)(Gmodel) #aqui a saida pode ser a indicacao da action tomada e a proxima tela
Gmodel = layers.ReLu()(Gmodel)

#Aqui posso gerar as proximas imagens com um generator em seguida

renderizator = Model()
renderizator = layers.Dense(500, input_shape = (277,), kernel_regularizer=regularizers.l1(0.1))(renderizator)#aqui posso dar mais importancia para o click
renderizator = layers.LeakyReLU(0.2)(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.Dense(1000, kernel_regularizer=regularizers.l1(0.1))(renderizator)
renderizator = layers.LeakyReLU(0.2)(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.Dense(2000, kernel_regularizer=regularizers.l1(0.1))(renderizator)
renderizator = layers.LeakyReLU(0.2)(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.Dense(shape[0]*shape[1]*shape[2], kernel_regularizer=regularizers.l1(0.1))(renderizator)
renderizator = layers.LeakyReLU(0.2)(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.Reshape(shape[1],shape[0],shape[2])(renderizator)
renderizator = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.LeakyReLU(alpha=0.2)(renderizator)
renderizator = layers.BatchNormalization()(renderizator)
renderizator = layers.Conv2DTranspose(3, (10,10), strides=(2,2), padding='same')(renderizator)



#nos modelos de aprendizado robusto, a saida seria a mesma da entrada
#como eu poderia reduzir isso matematicamente???
#so se fosse escolhido um outro problema
#como definir uma matriz robusta?? Para isso preciso estudar o conteudo





