# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:46:13 2020

@author: KX764QE
"""

# data
TRAINING_DATA_FILE = 'Google_Stock_Price_Train.csv'
TEST_DATA_FILE = 'Google_Stock_Price_Test.csv'

WINDOW_SIZE = 60

# input variables 
FEATURES = ['Open']

# model fitting
BATCH_SIZE = 32
EPOCHS = 150

# model persisting
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"