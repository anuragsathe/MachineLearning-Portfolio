from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Create FastAPI app
app = FastAPI()

# Load trained model and scaler
model = joblib.load("SVM_model.pkl")
scaler = joblib.load("Scaler.pkl")

# Define input schema using Pydantic
class TitanicInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int
    Sex_female: int
    Sex_male: int

# API endpoint for prediction
@app.post("/predict")
def predict(data: TitanicInput):
    # Convert input to NumPy array
    input_data = np.array([[
        data.Pclass, data.Age, data.SibSp, data.Parch, data.Fare,
        data.Embarked_C, data.Embarked_Q, data.Embarked_S,
        data.Sex_female, data.Sex_male
    ]])

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Return result
    result = "Survived" if prediction[0] == 1 else "Not Survived"
    return {"Prediction": result}

print("complete")