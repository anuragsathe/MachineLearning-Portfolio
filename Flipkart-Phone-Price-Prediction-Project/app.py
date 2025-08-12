# ===================== Import Required Libraries =====================
import streamlit as st
import numpy as np
import joblib

# ===================== Load Model =====================
model = joblib.load("phone_price_predictor.pkl")

# ===================== App Title =====================
st.set_page_config(page_title="📱 Phone Price Predictor", layout="centered")
st.title("📱 5G Phone Price Predictor")
st.markdown("Use below phone specs to predict the price instantly!")

# ===================== 🔧 Input Section =====================
st.subheader("🛠️ Phone Specifications")

# RAM
ram = st.slider("🔹 RAM (GB)", 2, 16, 8)

# Display Size
display_size = st.slider("🔹 Display Size (inches)", 5.0, 7.2, 6.5)

# Battery
battery = st.slider("🔋 Battery (mAh)", 3000, 8000, 5000)

# Back Camera
back_camera = st.number_input("📷 Back Camera (encoded value)", min_value=0)

# Front Camera
front_camera = st.number_input("🤳 Front Camera (encoded value)", min_value=0)

# Processor
processor = st.number_input("⚙️ Processor (encoded value)", min_value=0)

# Rating
rating = st.slider("⭐ User Rating", 1.0, 5.0, 4.3, step=0.1)

# Discount
discount = st.slider("🏷️ Discount (%)", 0.0, 100.0, 20.0)

# ===================== 🔮 Predict Button =====================
if st.button("🔍 Predict Phone Price"):
    input_data = np.array([[ram, display_size, battery, back_camera,
                            front_camera, processor, rating, discount]])

    # Predict using the trained model
    predicted_price = model.predict(input_data)[0]

    # Show prediction
    st.success(f"💰 **Estimated Phone Price: ₹{round(predicted_price, 2)}**")
