# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:28:45 2020

@author: Bangkit Team
"""

# Import Library
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv("https://github.com/AkhmadMuzanni/TestingBangkit/releases/download/1/cardio_train.csv")