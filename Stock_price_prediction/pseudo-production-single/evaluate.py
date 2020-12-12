# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:43:04 2020

@author: KX764QE
"""
import data_management as dm
import config
import joblib
import matplotlib.pyplot as plt
import model as m

def make_prediction(path_to_data):
    
    dataset = dm.load_dataset(path_to_data)
    X_test, y_test = dm.create_dataset(dataset)
    y_test = dm.reshape_dataset_y(y_test)
    

    scaler = joblib.load(config.SCALER_PATH)
    
    
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = dm.reshape_dataset_x(X_test_scaled)
    
    # y_test_scaled = scaler.transform(y_test)
    
    model = m.lstm_model()
    model.load_weights(config.MODEL_PATH)
    
    prediction = model.predict(X_test_scaled)
    prediction = scaler.inverse_transform(prediction)
    
    plt.plot(y_test, color = 'red', label = 'Real price')
    plt.plot(prediction, color = 'blue', label = 'Predicted price')
    
    plt.title('Google price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.savefig('eval_results.png',  bbox_inches='tight')
    
    
if __name__ == '__main__':
    make_prediction(config.TEST_DATA_FILE)