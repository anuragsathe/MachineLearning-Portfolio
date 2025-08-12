
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="🏥 Insurance Premium Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
    .title {
        font-size: 40px !important;
        color: #2E86C1;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<p class="title">🏥 Medical Insurance Premium Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fill in your health details below and get an instant premium estimate</p>', unsafe_allow_html=True)

# ------------------ Two-column Input Form ------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=0, max_value=120, step=1)
    diabetes = st.selectbox("💉 Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
    bp = st.selectbox("🩺 Blood Pressure Problems", [0, 1], format_func=lambda x: "Yes" if x else "No")
    transplant = st.selectbox("🧬 Any Transplants", [0, 1], format_func=lambda x: "Yes" if x else "No")
    chronic = st.selectbox("♾ Any Chronic Diseases", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    height = st.number_input("📏 Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("⚖ Weight (kg)", min_value=10, max_value=300)
    allergies = st.selectbox("🌿 Known Allergies", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cancer = st.selectbox("🎗 History of Cancer in Family", [0, 1], format_func=lambda x: "Yes" if x else "No")
    surgeries = st.slider("🔪 Number of Major Surgeries", 0, 10)

# ------------------ Prediction Button ------------------
st.markdown("---")
if st.button("🔍 Predict Premium"):
    input_data = {
        "Age": age,
        "Diabetes": diabetes,
        "BloodPressureProblems": bp,
        "AnyTransplants": transplant,
        "AnyChronicDiseases": chronic,
        "Height": height,
        "Weight": weight,
        "KnownAllergies": allergies,
        "HistoryOfCancerInFamily": cancer,
        "NumberOfMajorSurgeries": surgeries
    }

    try:
        with st.spinner("⏳ Predicting your premium..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
            result = response.json()
        st.success(f"💰 **Predicted Premium:** ₹{result['Predicted Premium Price']}")
    except:
        st.error("❌ Could not connect to the backend API. Please check your server.")

# ------------------ Footer ------------------
st.markdown("""
---
**Tip:** This is just an estimate based on the provided details.  
For an accurate premium, please consult an insurance provider.  
""")
