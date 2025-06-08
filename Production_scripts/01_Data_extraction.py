#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import requests
from datetime import datetime,date
from dateutil.relativedelta import relativedelta


# Parameters
end_date = (date.today()+relativedelta(days=-1)).strftime("%Y-%m-%d")
start_date = (datetime.strptime(end_date,"%Y-%m-%d") + relativedelta(years=-3)).strftime("%Y-%m-%d")
stocks = ['ASHOKLEY','CANBK','LICI','ONGC','SBIN']
today_date = date.today().strftime("%Y-%m-%d")


dev_data = pd.DataFrame(columns=['Date','Stock','Close'])
candle_stick_data = pd.DataFrame(columns=['Date','Stock','Open','High','Low','Close'])
for stock in stocks:
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}.BSE&outputsize=full&apikey=OL89MS2MIR3KD58F'
        r = requests.get(url)
        data = r.json()
        dataframe1 = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        dataframe1 = dataframe1.reset_index(names='Date')
        dataframe1.rename(columns = {'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close'},inplace=True)
        dataframe1['Stock'] = stock
        dataframe1 = dataframe1[dataframe1['Date']>start_date]
        dataframe1 = dataframe1[['Date','Stock','Open','High','Low','Close']]
        candle_stick_data = pd.concat([candle_stick_data,dataframe1],ignore_index=True)
        dataframe1 = dataframe1[['Date','Stock','Close']]
        dev_data = pd.concat([dev_data,dataframe1],ignore_index=True)
    except Exception as e:
        print(e)
dev_data.to_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_dataset\\train\\Stocks_train_{today_date}.csv",index=False)
candle_stick_data.to_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_dataset\\candle_stick\\Stocks_data_{today_date}.csv",index=False)






