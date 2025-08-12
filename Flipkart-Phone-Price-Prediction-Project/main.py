# ======================== Import Required Libraries ========================
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# ======================== Initialize FastAPI ========================
app = FastAPI()

# ======================== Load Trained Model ========================
model = joblib.load("phone_price_predictor.pkl")

# ======================== Define Input Schema ========================
class Phone_price(BaseModel):
    RAM_GB: float
    Display_Size_inches: float
    Battery_mAh: float
    Back_Camera: int
    Front_Camera: int
    Processor: int
    Rating: float
    Discount: float

# ======================== Prediction Route ========================
@app.post("/predict")
def predict_price(data: Phone_price):
    # Convert input to array
    input_data = np.array([[data.RAM_GB, data.Display_Size_inches,
                            data.Battery_mAh, data.Back_Camera, data.Front_Camera,
                            data.Processor, data.Rating, data.Discount]])

    # Predict price
    prediction = model.predict(input_data)

    # Return predicted price
    return {"Predicted Phone Price (INR)": round(prediction[0], 2)}
