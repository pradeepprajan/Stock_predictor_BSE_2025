#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries


import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from datetime import datetime,date
from dateutil.relativedelta import relativedelta


# # Reading and preprocessing of dataset


today_date = date.today().strftime("%Y-%m-%d")


dataset = pd.read_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_dataset\\train\\Stocks_train_{today_date}.csv")


dataset_test = dataset.pivot(index='Date',columns='Stock',values='Close').reset_index()


dataset_test = dataset_test[dataset_test['Date']<today_date]


# Feature Scaling
num_cols = ['ASHOKLEY','CANBK','LICI','ONGC','SBIN']
sc = joblib.load('D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\Stock_predictor_BSE_2025\\model_files\\min_max_scaler.joblib')
test_data = sc.transform(dataset_test[num_cols])



# Creating a data structure with 60 time steps and 1 output
X_test = test_data[len(test_data)-60:len(test_data),:]


X_test = X_test.reshape(1,60,-1)


# # Load the model and predict


regressor_load = load_model('D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\Stock_predictor_BSE_2025\\model_files\\stock_predictor.keras')


today_preds = regressor_load.predict(X_test)
today_preds = sc.inverse_transform(today_preds)


today_preds_pd = pd.DataFrame(today_preds,columns=num_cols)
today_preds_pd['Date'] = today_date
today_preds_pd = today_preds_pd[['Date']+num_cols]



today_preds_pd.to_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_predictions\\Predictions_{today_date}.csv",index=False)




