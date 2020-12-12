# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:51:47 2020

@author: KX764QE
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import config
import joblib
import data_management as dm

class DataScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, scaler = MinMaxScaler()):
        self.scaler = scaler
    
    def fit(self, X, y = None):

        self.scaler.fit(X)
        return self
    
    def inverse_transform(self, X):
        X = X.copy()
        X = self.scaler.inverse_transform(X)
        
        return X
        
    def transform(self, X):
        X = X.copy()
        X = self.scaler.transform(X)
        return X
                 

if __name__ == '__main__':
    
    dataset = dm.load_dataset(config.TRAINING_DATA_FILE)
    
    X_train, y_train = dm.create_dataset(dataset)
    y_train = dm.reshape_dataset_y(y_train)
    print(X_train[: 3, ])
    
    scaler = DataScaler()
    scaler.fit(dataset)
    
    X_train_scaled = scaler.transform(X_train)
    print(X_train_scaled[: 3, ])
    y_train_scaled = scaler.transform(y_train)
    print(X_train_scaled.shape, y_train_scaled.shape)
    joblib.dump(scaler, config.SCALER_PATH)
    