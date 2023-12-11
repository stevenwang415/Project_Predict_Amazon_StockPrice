
import pandas as pd
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_monthly_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    response = requests.get(url)
    data = response.json()
    # Extract the time series data and convert it to a DataFrame
    time_series_key = 'Time Series (Daily)'
    df = pd.DataFrame(data[time_series_key]).T
    # df['Date'] = df.index
    # df.reset_index(drop = True, inplace= True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df

# Example usage
api_key = '93HDEE8UFKUO8AJ8'
df_amazon = get_monthly_stock_data('AMZN', api_key)


def get_dat():
  df_amazon.index = pd.to_datetime(df_amazon.index)
  one_month_ago = datetime.datetime.now() - datetime.timedelta(days = 1.5*365)
  dat = df_amazon[df_amazon.index > one_month_ago]
  dat = dat.copy()
  dat['Date'] = dat.index
  dat.index = dat.pop('Date')
  dat = dat[['Close']]
  dat['Close'] = dat['Close'].astype('float64')
  dat = dat[::-1]
  
  # Uncomment this if you want a physical csv copy 
  #dat.to_csv('C:/Users/chens/Documents/FinalProject/app/data.csv', index =True)
  return dat
