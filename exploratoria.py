# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:10:57 2021

@author: gabri

Creation of architecture to develop interatives applications by the mean of an
artificial network - The use here is to overfitting the data

"""

import tensorflow
from tensorflow.keras import Sequential
import os
import numpy as np

count = []
for directory in os.listdir('data/filtered_traces'):
    subdirectory = os.listdir("data/filtered_traces/" + directory)
    for sub in subdirectory:
        archives = os.listdir("data/filtered_traces/" + directory + "/" + sub + "/screenshots")
        count.append([directory, int(len(archives))])
count = np.asarray(count)

count[:,1] = count[:,1].astype(int)
count[:,1]

count = count[np.argsort(count[:,1].astype(int))]
n = 10
ranked = np.argsort(count[:,1].astype(int))
largest_indexs = count[::-1][:n]

#provavelmente terei de desenvolver os meus prorpios dados... Poderia usar estes como base...