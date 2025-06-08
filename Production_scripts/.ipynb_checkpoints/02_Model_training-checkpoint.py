#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib
from datetime import datetime,date 


# # Reading and preprocessing of dataset


num_cols = ['ASHOKLEY','CANBK','LICI','ONGC','SBIN']
today_date = date.today().strftime("%Y-%m-%d")


dataset = pd.read_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_dataset\\train\\Stocks_train_{today_date}.csv")


# Pivot with stock column
dataset_train = dataset.pivot(index='Date',columns='Stock',values='Close').reset_index()



# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
train_data = sc.fit_transform(dataset_train[num_cols])


# Creating a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60,len(train_data)):
    X_train.append(train_data[i-60:i,:])
    y_train.append(train_data[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)


# Saving min max scaler file
joblib.dump(sc,'D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\Stock_predictor_BSE_2025\\model_files\\min_max_scaler.joblib')


# # Training of GRU model

regressor = Sequential()

regressor.add(GRU(units=200,activation='relu',return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.05))

regressor.add(GRU(units=200,activation='relu'))
regressor.add(Dropout(0.05))

regressor.add(Dense(units=X_train.shape[2]))

regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


history = regressor.fit(X_train,y_train,epochs=50, batch_size=100,validation_split=0.2)


#  Plotting loss and accuracy curves


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.savefig(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\loss_curves\\loss_curve_{today_date}.png")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.savefig(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\accuracy_curves\\accuracy_curve_{today_date}.png")


# # Saving the model


regressor.save('D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\Stock_predictor_BSE_2025\\model_files\\stock_predictor.keras')

