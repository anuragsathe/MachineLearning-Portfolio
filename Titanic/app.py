import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("SVM_model.pkl")
scaler = joblib.load("Scaler.pkl")

# Set page config
st.set_page_config(page_title="Titanic Survival Prediction 🚢", page_icon="⚓", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🚢 Titanic Survival Predictor</h1>
    <p style='text-align: center; color: grey;'>Enter passenger details below to predict survival chance on the Titanic.</p>
    <hr>
""", unsafe_allow_html=True)

# Input fields in 2 columns
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    age = st.slider("Age", 0.0, 80.0, 25.0)
    sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.slider("Fare Paid", 0.0, 550.0, 50.0)

with col2:
    sex = st.radio("Sex", ["Male", "Female"])
    embarked = st.radio("Port of Embarkation", ["S", "C", "Q"])

# Convert inputs to model format
sex_female = 1 if sex == "Female" else 0
sex_male = 1 if sex == "Male" else 0

embarked_S = 1 if embarked == "S" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

input_data = np.array([[
    pclass, age, sibsp, parch, fare,
    embarked_C, embarked_Q, embarked_S,
    sex_female, sex_male
]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.success("🎉 The passenger **Survived**!")
        st.balloons()
    else:
        st.error("❌ The passenger **Did NOT survive**.")
