from mani import get_model
from mani import get_predictions
from mani import get_dates
from mani import get_dat
import streamlit as st
import matplotlib.pyplot as plt

dat = get_dat()


def show_stock_plot():
    st.header('Explore Page')
    st.title('Amazon Stock Price Data')
    st.write('Below are realtime Stock Prices for Amazon. We also include a chart that predicts\
             the stock prices from a LSTM.') 


    fig, ax = plt.subplots(figsize = (10, 3))
    ax.plot(dat.index, dat['Close'])
    plt.ylabel('Closing Prices')
    plt.xlabel('Time')
    plt.title('Amazon Stock Prices')

    st.pyplot(fig)


def show_predict_plot():
    d_train, x_train, y_train, d_val, x_val, y_val, d_test, x_test, y_test = get_dates()

    model = get_model(x_train, y_train, x_val, y_val, 60, .001)
    train_predictions = get_predictions(x_train, model)
    val_predictions = get_predictions(x_val, model)
    test_predictions = get_predictions(x_test, model)

    fig1, ax1 = plt.subplots(figsize = (10, 3))
    ax1.plot(d_train, train_predictions)    
    ax1.plot(d_val, val_predictions)
    ax1.plot(d_test, test_predictions)
    plt.ylabel('Closing Prices')
    plt.xlabel('Time')
    plt.title('Prediction of Amazon Prices')
    st.pyplot(fig1)