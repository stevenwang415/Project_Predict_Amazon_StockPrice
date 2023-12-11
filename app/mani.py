#! C:\Users\chens\Documents\FinalProject\env\Scripts\python.exe

# Here is where most of the computational power is going
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from data import get_dat
#import os


def str_dt(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year = year, month = month, day = day)

# def get_dat():
#   directory = "C:/Users/chens/Documents/FinalProject/app"
#   file_name = 'data.csv'

#   file_path = os.path.join(directory, file_name)

#   dat = pd.read_csv(file_path)


#   dat['Date'] = dat['Date'].apply(str_dt)

#   dat.index = dat.pop('Date')
#   return dat


#Creating our window function
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n = 3):

  a = len(dataframe) / 3
  b = a - np.fix(a)
  
  if b != 0:
    first_date = str_dt(first_date_str) + datetime.timedelta(days = 1)
    a = (len(dataframe) + 1) / 3
    b = a - np.fix(a)
  
    if b != 0:
      first_date = str_dt(first_date_str) + datetime.timedelta(days = 2)
      a = (len(dataframe) + 2) / 3
      b = (a) - np.fix(a)
  
      if b != 0:
        first_date = str_dt(first_date_str) + datetime.timedelta(days = 3)
        a = (len(dataframe) + 3) / 3
        b = (a) - np.fix(a)
  
        if b != 0:
          return 'Something went wrong with the date ranges'
  
  
  # Converting our data from strings to dates
  last_date = str_dt(last_date_str)

  # Setting a target date
  target_date = first_date

  dates = []
  X, Y = [], []

  # Create loop condition
  stop = False

  while True:


    # We are going to select all values till the target date, and then select the last 4
    dat_subset = dataframe.loc[:target_date].tail(4)

    # Those last 4 values are going to become an array
    values = dat_subset['Close'].to_numpy()


    # Set all the values except the last one to x (our training)
    # Set the last value to y the value of stock at the date
    x, y = values[:-1], values[-1]


    # Append our current date
    dates.append(target_date)

    # Append the values that will become our training data
    X.append(x)


    # Append the value that corresponds to our current day
    Y.append(y)


    # Look a week later from the current time
    next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days = 7)]


    # We grab the time of our next data point to look back 3 days
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day

    next_date = datetime.datetime(day = int(day), month = int(month), year = int(year))

    if stop:
      break

    target_date = next_date


    if target_date == last_date:
      stop = True



    # Create our dataframe
  ret_df = pd.DataFrame({})

    # Add in our dates
  ret_df['Target Date'] = dates

    # Transform x into an np array
  X = np.array(X)
    # We loop through each value in x and assign them as data that represents
    # value for each day until our target day
  for i in range(0, n):
    ret_df[f'Target-{n-i}'] = X[:, i]

  ret_df['Target'] = Y


  return ret_df

# Calling our window function
def get_window_df(dat = get_dat()):
    start_date = dat.index[0] + datetime.timedelta(days = 3)
    start_date = str(start_date.to_numpy()).split('T')[0]
    end_date = str(dat.index[-1].to_numpy()).split('T')[0]
    windowed_df = df_to_windowed_df(dat, start_date, end_date)
    return windowed_df


def windowed_df_to_date_X_y(df):
  df_as_np = df.to_numpy()

  dates = df_as_np[:, 0]

  input_matrix = df_as_np[:, 1:-1]
  X = input_matrix.reshape((len(dates), input_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

#Getting our date matrix so that we can begin tensor flow
def get_dates():
  dates, X, y = windowed_df_to_date_X_y(get_window_df())
  q_80 = int(len(dates) * .8)
  q_90 = int(len(dates) * .9)
  dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
  dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
  dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
  return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test

#--------------------------------------------------------------------------
# Here we are splitting our data in order to train 

# Here if you want to show the plot
# plt.plot(dates_train, y_train)
# plt.plot(dates_val, y_val)
# plt.plot(dates_test, y_test)
# plt.legend(['Trained, Val, Test'])
# plt.show()

#-------------------------------------------------------------------------
# Here we actually apply LSTM on our stock model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

def get_model(x_train, y_train, x_val, y_val, epoch, lr):
  
  model = Sequential([layers.Input((3,1)),
                  layers.LSTM(64),
                  layers.Dense(32, activation = 'relu'),
                  layers.Dense(32, activation = 'relu'),
                  layers.Dense(1)])

  model.compile(loss = 'mse',
            optimizer = Adam(learning_rate = lr),
            metrics = ['mean_absolute_error'])


  model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = epoch)
  return model
    
def get_predictions(data, model):
  return model.predict(data).flatten()


#----------------------------------------------------------------------------------



