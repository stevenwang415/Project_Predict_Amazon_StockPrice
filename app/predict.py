
import streamlit as st
from mani import get_dat
import numpy as np
from mani import get_predictions
from mani import get_model
from mani import get_dates


def get_predict_page():
    st.header('Welcome To The Predict Page')
    st.write('Here we will use the model obtained by LSTM to predict future values.\
             Start by entering the closing prices for the previous 3 days (including today)')
    a = st.number_input('Today Closing Price')
    b = st.number_input('Yesteday Closing Price')
    c = st.number_input('Day Before Yesterday Closing Price')
    
    if st.button('Calculate'):
        test = np.array([[[float(a)], [float(b)], [float(c)]]])
        d_train, x_train, y_train, d_val, x_val, y_val, d_test, x_test, y_test = get_dates()
        model = get_model(x_train, y_train, x_val, y_val, 60, .001)
        
        predict = get_predictions(test, model)[0]
        st.subheader(f'The predicted closing stock price is {predict:.2f}')
    
    
    st.write('A realtime reference table is provided for the recent days that you might\
                want to choose')
    
    dat = get_dat().tail(10)[::-1]
    dat.rename(columns = {'Close': 'Closing Prices'}, inplace = True)
    st.table(dat)