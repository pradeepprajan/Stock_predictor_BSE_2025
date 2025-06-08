#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries


from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime,date
from dash.dependencies import Output, Input
import subprocess
from dateutil.relativedelta import relativedelta

# # Importing the predictions dataset

today_date = date.today().strftime("%Y-%m-%d")
yesterday_date = (date.today()+relativedelta(days=-1)).strftime("%Y-%m-%d")
today_prediction = pd.read_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_predictions\\Predictions_{today_date}.csv")


past_prices = pd.read_csv(f"D:\\Projects\\Jupyter_Lab\\Stock_market_predictor\\stock_prices_dataset\\candle_stick\\Stocks_data_{today_date}.csv")
past_prices = past_prices[past_prices['Date']<today_date]



color_map = {'⯅':'green','⯆':'red'}
if float(today_prediction.loc[:,'ASHOKLEY'].values[0]) >= float(past_prices.loc[(past_prices['Stock']=='ASHOKLEY')&(past_prices['Date']==yesterday_date),'Close'].values[0]):
    default_trend = '⯅'
    default_color = 'green'
else:
    default_trend = '⯆'
    default_color = 'red'


# # Creating the dashboard


app = Dash(__name__,external_stylesheets=[dbc.themes.COSMO])
server = app.server
stocks = ['ASHOKLEY','CANBK','LICI','ONGC','SBIN']
option_list = []

for stock in stocks:
    params = {'label':stock,'value':stock}
    option_list.append(params)

dropdown = html.Div([
    html.Label('Select the stock in BSE which you want to analyze',style={'display':'block','fontSize':15,'marginLeft':10}),
    dcc.Dropdown(
        id = 'stock',
        options = option_list,
        value = 'ASHOKLEY',
        clearable = False,
        style = {'marginLeft' : 10, 'marginRight' : 20}
    )
])
button =  html.Div([
    html.Button('Recalibrate model',id='recalibrate',style={'marginLeft': 10, 'marginRight': 20})
])

app.layout = html.Div(
    [
        dbc.Row([
            dbc.Col((html.H1('STOCK MARKET PREDICTOR',
                            style={'textAlign':'center','color':'white','marginTop':90})), width=12)
                    ],style={'background-color':'indigo','marginBottom':20,'height':200}),
        html.Div(
        [
            dbc.Row([
                dbc.Col(dropdown, width=12)
            ],style={'marginBottom':20}),
            dbc.Row([
                dbc.Col(html.H3('Historical trends'))
            ],style={'marginBottom':5}),
            dbc.Row([
                dbc.Col(dcc.Graph(id='candlestick-chart',figure={},config={'displayModeBar':False}))
            ],style={'marginBottom':10,'marginTop':10}),
            dbc.Row([
                dbc.Col(html.H5('Expected price today'),width=3),
                dbc.Col(id='prediction',children = html.H3(id='predicted-price',children=f"{float(today_prediction.loc[:,'ASHOKLEY'].values[0])} {default_trend}"),width=3,style={'color':default_color})
            ],style={'marginBottom':40}),
            dbc.Row([
                dbc.Col(html.H5('Click the button to recalibrate stock predictor'),width=5),
                dbc.Col(button),
                html.Div([
                    dcc.ConfirmDialog(
                        id = 'recalib-msg',
                        message = 'Recalibration done successfully')
            ])
        ])
        ])
    ]
)

@app.callback(
    Output(component_id='predicted-price', component_property='children'),
    Output(component_id='prediction', component_property='style'),
    Output(component_id='candlestick-chart', component_property='figure'),
    [Input(component_id='stock', component_property='value')]
)
def candlestick_chart(stock):
    past_prices_stock = past_prices[past_prices['Stock']==stock]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=pd.to_datetime(past_prices_stock['Date']),open=past_prices_stock['Open'],high=past_prices_stock['High'],
                             low=past_prices_stock['Low'],close=past_prices_stock['Close'],increasing=dict(line=dict(color='blue')),
                             decreasing=dict(line=dict(color='red')),name='Past_prices'))
    fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    )
    predicted_prices_stock = float(today_prediction.loc[:,stock].values[0])
    if float(today_prediction.loc[:,stock].values[0]) >= float(past_prices.loc[(past_prices['Stock']==stock)&(past_prices['Date']==yesterday_date),'Close'].values[0]):
        trend = '⯅'
        color = 'green'
    else:
        trend = '⯆'
        color = 'red'
    return f'{predicted_prices_stock} {trend}',{'color':color},fig

@app.callback(
    Output(component_id='recalib-msg', component_property='message'),
    [Input(component_id='recalibrate', component_property='n_clicks')]
)
def retrain_model(n_clicks):
    subprocess.run(['python','02_Model_training.py'])
    subprocess.run(['python','03_Stock_price_prediction.py'])
    subprocess.run(['python','04_Dashboard.py'])
    return "Recalibration done successfully"

if __name__ == '__main__':
    app.run(debug=True,port=8051)





