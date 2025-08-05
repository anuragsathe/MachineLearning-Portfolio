# ======================== Import Required Libraries ========================
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# ======================== Initialize FastAPI App ========================
app = FastAPI()

# ======================== Load Trained ML Model & Scaler ========================
# Load the Random Forest model and StandardScaler saved earlier
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================== Define Request Data Format using Pydantic ========================
class CreditApprovalRequest(BaseModel):
    Gender: int
    Age: float
    Debt: float
    Married: int
    BankCustomer: int
    Industry: int
    Ethnicity: int
    YearsEmployed: float
    PriorDefault: int
    Employed: int
    CreditScore: int
    DriversLicense: int
    Citizen: int
    ZipCode: int
    Income: int

# ======================== Define Prediction Endpoint ========================
@app.post("/predict")
def predict_approval(data: CreditApprovalRequest):
    # Convert input data to numpy array (2D)
    input_data = np.array([[data.Gender, data.Age, data.Debt, data.Married, data.BankCustomer,
                            data.Industry, data.Ethnicity, data.YearsEmployed, data.PriorDefault,
                            data.Employed, data.CreditScore, data.DriversLicense,
                            data.Citizen, data.ZipCode, data.Income]])

    # Apply the same scaling used during training
    scaled_data = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(scaled_data)

    # Format result for frontend
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    
    # Return result as JSON
    return {"prediction": result}

print("Complete")