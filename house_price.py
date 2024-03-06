import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import joblib
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('USA_Housing.csv')

model = joblib.load('HousePredictor.pkl')

st.markdown("<h1 style = 'color: #83C0C1; text-align: center; font-family: helvetica '>House Price Prediction</h1>", unsafe_allow_html =True)
st.markdown("<h4 style = 'margin: -30px; color: #6D2932; text-align: center; font-family: helvetica'>Designed and built by", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h3 style = 'margin: -30px; color: #1E1E1E; text-align: center; font-family: cursive '>kaka Tech Word", unsafe_allow_html =True)

st.markdown("<p style = 'font-family: recursive' >Model Name: Home Value Predictor.</p>", unsafe_allow_html=True)

st.markdown("<p style = 'font-family: recursive' >Objective: The HomeValue Predictor is designed to predict the market value of residential properties based on various features and attributes. Its primary goal is to assist homebuyers, sellers, and real estate professionals in estimating the fair market price of a house.</p>", unsafe_allow_html=True)
#st.image('pngwing.com (4).png', width = 1000, height= 400,  use_column_width = True, caption='Welcone to house prediction')

#import streamlit as st

st.image('pngwing.com (5).png', width=1000,use_column_width=True, caption='Welcome to house prediction')
st.markdown('<br>', unsafe_allow_html = True)

st.dataframe(data, use_container_width=True)

st.sidebar.image('pngwing.com (6).png', width = 150, use_column_width = True, caption='We love having you. ')
st.markdown('<br>', unsafe_allow_html = True)
input_choice = st.sidebar.radio('choose your input choice',['slider input', 'number input'])

if input_choice == 'slider input':
    area_income = st.sidebar.slider('Average area income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.slider('Average area house age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_number = st.sidebar.slider('Average number of Rooms', data['Avg. Area Number of Rooms'].min(),data['Avg. Area Number of Rooms'].max())
    bedroom_number = st.sidebar.slider('Average number of bed rooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population = st.sidebar.slider('Area Population', data['Area Population'].min(),data['Area Population'].max() )
else: 
    st.markdown('<br>', unsafe_allow_html = True)
    area_income = st.sidebar.number_input('Avg. Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.number_input('Avg. Area House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_number = st.sidebar.number_input('Avg. Area Number of Rooms', data['Avg. Area Number of Rooms'].min(),data['Avg. Area Number of Rooms'].max())
    bedroom_number = st.sidebar.number_input('Avg. Area Number of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population = st.sidebar.number_input('Area Population', data['Area Population'].min(),data['Area Population'].max() )


st.markdown('<br>', unsafe_allow_html = True)
input_vars = pd.DataFrame({'Avg. Area Income':[area_income], 
                          'Avg. Area House Age':[house_age],
                          'Avg. Area Number of Rooms': [room_number],
                          'Avg. Area Number of Bedrooms': [bedroom_number], 
                          'Area Population':[area_population]})
st.markdown('<br>', unsafe_allow_html = True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h5 style = 'margin: -30px;color: olive; font-family: helvetica'>User Input Variable</h5>", unsafe_allow_html = True)
st.dataframe(input_vars)

predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')


#['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population'],
