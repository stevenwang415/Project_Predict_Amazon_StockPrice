#! C:\Users\chens\Documents\FinalProject\env\Scripts\python.exe

import streamlit as st
from plots import show_stock_plot
from predict import get_predict_page
from plots import show_predict_plot

page = st.sidebar.selectbox('Explore or Predict', ('Explore', 'Predict'))

if page == 'Predict':
    get_predict_page()
    
else:
    show_stock_plot()
    st.write('If you would like to see how the LSTM performed on this set of data click the button')
    if st.button('Click Me'):
        show_predict_plot()
        if st.button('Close'):
            st.rerun

