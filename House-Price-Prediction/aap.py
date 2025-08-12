import streamlit as st 
import pickle
import numpy as np

# load MOdel
with open("house_price_model.pkl","rb") as f:
    model = pickle.load(f)
    

st.title("House Price Prediction App")

# Input Field
lot_area = st.number_input("Lot Area (sq ft)",min_value=500,max_value=1000000,step= 100)

year_bulit = st.number_input("Year Built",min_value=1800,max_value=2024,step=1)

full_bath = st.number_input("Full Bathrooms" , min_value=0,max_value=5,step=1)

bedroom = st.number_input("Bethroom",min_value=1,max_value=10,step=1)

Garage = st.number_input("Garage",min_value=0,max_value=5,step=1)

# Predict button

if st.button("Predict Price"):
    features = np.array([[lot_area,year_bulit,full_bath,bedroom,Garage]])
    prediction = model.predict(features)
    st.success(f"Estimated House Price: $ {prediction[0][0]:,.2f}")
