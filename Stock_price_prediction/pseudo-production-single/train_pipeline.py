# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:26:14 2020

@author: KX764QE
"""

import preprocessors as pp
import config
import model
import data_management as dm
import joblib


def run_training(save_result):
    
    dataset = dm.load_dataset(config.TRAINING_DATA_FILE)
    X_train, y_train = dm.create_dataset(dataset)
    y_train = dm.reshape_dataset_y(y_train)
    
    scaler = pp.DataScaler()
    scaler.fit(dataset)
    
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = dm.reshape_dataset_x(X_train_scaled)
    
    y_train_scaled = scaler.transform(y_train)
    
    model_instance = model.lstm_model()
    model_instance.fit(X_train_scaled, y_train_scaled, 
                        batch_size = config.BATCH_SIZE, 
                        epochs = config.EPOCHS, 
                        callbacks = model.callback_list)
    if save_result:
        joblib.dump(scaler, config.SCALER_PATH)
        # joblib.dump(model_instance, config.MODEL_PATH)
        model_instance.save(config.MODEL_PATH)
    
    
if __name__ == '__main__':
    run_training(save_result = True)
    
    
    