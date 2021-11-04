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
import cv2
import pandas as pd

fluxograma = pd.read_csv("data/app_minimo/fluxograma.csv", sep = ';')



