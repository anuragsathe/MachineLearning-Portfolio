from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("Hello.pkl")
scaler = joblib.load("sc.pkl")



# ======================== Define Input Schema ===========
class Price(BaseModel):
    Year     :                   int
    Present_Price :        float
    Kms_Driven     :          int  
    Owner        :int
    Fuel_Type_CNG : int  
    Fuel_Type_Diesel :    int  
    Fuel_Type_Petrol  :     int 
    Seller_Type_Dealer :     int  
    Seller_Type_Individual:  int  
    Transmission_Automatic : int  
    Transmission_Manual     :int
    


# ======================== Prediction Route ====================
@app.post("/predict")
def predict_price(data:Price):
    
    input_data = np.array([[data.Year, data.Present_Price, data.Kms_Driven, data.Owner, data.Fuel_Type_CNG,
       data.Fuel_Type_Diesel, data.Fuel_Type_Petrol, data.Seller_Type_Dealer,
       data.Seller_Type_Individual, data.Transmission_Automatic,
       data.Transmission_Manual]])
    

    # Apply Scaler
    scaled_data = scaler.transform(input_data)
    
    # Predict price
    prediction = model.predict(scaled_data)
    
    #Return 
    return {"Seller Price":prediction[0]}
