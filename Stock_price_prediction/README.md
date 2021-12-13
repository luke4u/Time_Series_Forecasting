# LSTM-Stock-Price-Prediction
This is to predict the upwards and downwards trend of stock price using LSTM-based models. 

1) The 1st model is trained using the Open price from 2012-01-03 to 2017-12-29, and to predict Open price.

2) The 2nd model is trained using 'Open', 'High', 'Low', 'Volume', 'Close' and to predict Close price.

Conclusion - in the parts of prediction which contain spikes, the model lags behind the actual prices, but in the parts that contain smooth changes, the model manages to follow upwards and downward trends. Please review the output for details.

******Production level codes are now updated for the 1st model.******

******Dockerfile and Flask API dev are to be updated. Stay tuned******
