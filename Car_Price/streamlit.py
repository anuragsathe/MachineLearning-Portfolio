# ===================== Import Required Libraries =====================
import streamlit as st
import numpy as np
import joblib
import requests

# ===================== Load Model and Scaler ==========
model = joblib.load("Hello.pkl")
scaler = joblib.load("sc.pkl")

# ===================== App Title =================

st.set_page_config(page_title= "Seller Price Predictor",layout = 'centered')
st.title("CAR PRICE PREDICTION")
st.markdown('Predict the price of the car based on its specification')


# ======================== Input Form for User Data =======

    
Year   = st.selectbox("Year",[2014, 2013, 2017, 2011, 2018, 2015, 2016, 2009, 2010, 2012, 2003,
       2008, 2006, 2005, 2004, 2007])
    
Present_Price  = st.number_input("Present Price",min_value = 0.0)         
Kms_Driven     = st.number_input("Kms Driven",min_value = 500)
Owner          = st.selectbox("Owner",[0,1,2,3,4,5])
Fuel_Type_CNG  = st.selectbox("Fuel_Type_CNG ",[0,1])
Fuel_Type_Diesel = st.selectbox("Fuel_Type_Diesel ",[0,1])
Fuel_Type_Petrol  = st.selectbox("Fuel_Type_Petrol ",[0,1])
Seller_Type_Dealer  =  st.selectbox("Seller_Type_Dealer  ",[0,1])
Seller_Type_Individual =  st.selectbox("Seller_Type_Individual ",[0,1])
Transmission_Automatic  =  st.selectbox("Transmission_Automatic  ",[0,1])
Transmission_Manual     = st.selectbox("Transmission_Manual ",[0,1])

if st.button("🔮 Predict Price"):
    
    
    input_data = np.array([[Year, Present_Price, Kms_Driven, Owner, Fuel_Type_CNG,
       Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Dealer,
       Seller_Type_Individual, Transmission_Automatic,
       Transmission_Manual]])
        
    # Scale Input
    input_scaled = scaler.transform(input_data)
    
    #Predict
    Predicted_price = model.predict(input_scaled)[0]
    
    # Show result
    st.success(f"📌 Estimated CAR Price: ₹{Predicted_price}")

    
