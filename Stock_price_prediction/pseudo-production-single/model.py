# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:55:52 2020

@author: KX764QE
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import config

def lstm_model(unit_size = 50,
               input_shape = (config.WINDOW_SIZE, 1),
               droput_rate = 0.2
               ):
    
    regressor = Sequential()
    #add 1st lstm layer
    regressor.add(LSTM(units = unit_size, 
                       return_sequences = True, 
                       input_shape = input_shape))
    
    regressor.add(Dropout(rate = droput_rate))
    
    ##add 2nd lstm layer: 50 neurons
    regressor.add(LSTM(units = unit_size, return_sequences = True))
    regressor.add(Dropout(rate = droput_rate))
    
    ##add 3rd lstm layer
    regressor.add(LSTM(units = unit_size, return_sequences = True))
    regressor.add(Dropout(rate = droput_rate))
    
    ##add 4th lstm layer
    regressor.add(LSTM(units = unit_size, return_sequences = False))
    regressor.add(Dropout(rate = droput_rate))

    ##add output layer
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = 'adam', 
                      loss = 'mean_squared_error')
    
    return regressor


checkpoint = ModelCheckpoint(config.MODEL_PATH, 
                             monitor = 'loss', 
                             verbose = 1, 
                             save_best_only = True,
                             mode = 'min')

callback_list = [checkpoint]



if __name__ == '__main__':
    model = lstm_model()
    model.summary()
    
    
    















