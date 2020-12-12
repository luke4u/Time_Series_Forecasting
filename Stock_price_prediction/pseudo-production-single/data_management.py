# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:23:50 2020

@author: KX764QE
"""

import numpy as np
import pandas as pd
import config

def load_dataset(data_folder):
    """
    load dataset from input folder
    convert to numpy array
    """
    dataset = pd.read_csv(data_folder, usecols = config.FEATURES)
    dataset = dataset.values
    
    return dataset

#create sliding window data strucutre
def create_dataset(dataset):
    """
    use a pre-set sliding window to preduce a train set and target set
    """

    X= []
    y = []
    for i in range(config.WINDOW_SIZE, len(dataset)):
        X.append(dataset[i - config.WINDOW_SIZE: i, 0])
        y.append(dataset[i, 0])
        
    X = np.array(X)
    y = np.array(y)
    
    
    return X, y

def reshape_dataset_x(dataset):
    """
    add dimension for indicator
    """
    dataset = dataset.reshape( (dataset.shape[0], dataset.shape[1], 1) )
    return dataset

def reshape_dataset_y(dataset):
    """
    add dimension for batch
    """
    dataset = dataset.reshape(-1, 1)
    return dataset
    
if __name__ == '__main__':
    
    data = load_dataset(config.TRAINING_DATA_FILE)
    print(data.shape)
    X_train, y_train = create_dataset(data)
    print(X_train.shape, y_train.shape)
    X_train = reshape_dataset_x(X_train)
    y_train = reshape_dataset_y(y_train)
    print(X_train.shape, y_train.shape)






