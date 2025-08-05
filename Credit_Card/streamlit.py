import streamlit as st
import requests

# ======================== Custom CSS for Professional Look ========================
st.markdown("""
    <style>
        body {
            background-color: #f6f6f9;
        }
        .main {
            background-color: #fff;
            border-radius: 16px;
            padding: 2rem 2rem 1rem 2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            margin-top: 2rem;
        }
        h1 {
            color: #2d3a4b;
            font-weight: 700;
            letter-spacing: 1px;
        }
        label, .stSelectbox label, .stNumberInput label {
            font-weight: 600;
            color: #3b4252;
        }
        .stButton>button {
            background-color: #4f8cff;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            margin-top: 1em;
        }
        .stButton>button:hover {
            background-color: #3761c8;
        }
        .stSuccess {
            background-color: #e6f7ee;
            color: #167d55;
            border-radius: 8px;
            padding: 1em;
        }
        .stError {
            background-color: #fff0f0;
            color: #d7263d;
            border-radius: 8px;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# ======================== Page Navigation using Session State ========================
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def go_to_result(prediction):
    st.session_state.prediction = prediction
    st.session_state.page = 'result'

# ======================== Main App ========================
def credit_form():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown("## 🏦 Credit Approval Prediction")
    st.markdown(
        "Welcome to the **Credit Approval Predictor**! "
        "Fill out the form below to check your credit approval status instantly. "
        "All data is confidential and used only for prediction purposes."
    )

    with st.form("credit_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Married = st.selectbox("Marital Status", ["Single", "Married"])
            Ethnicity = st.selectbox("Ethnicity", ["Group 1", "Group 2", "Group 3"])
            Citizen = st.selectbox("Citizenship", ["Native", "Foreign", "Other"])
            ZipCode = st.number_input("Zip Code", min_value=10000, max_value=99999, value=12345)
        with col2:
            BankCustomer = st.selectbox("Bank Customer", ["No", "Yes"])
            Industry = st.selectbox("Industry", ["Industry 1", "Industry 2", "Industry 3", "Industry 4"])
            YearsEmployed = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0)
            Employed = st.selectbox("Currently Employed", ["No", "Yes"])
            DriversLicense = st.selectbox("Driver's License", ["No", "Yes"])

        st.markdown("### 💰 Financial Information")
        col3, col4 = st.columns(2)
        with col3:
            Debt = st.number_input("Total Debt ($)", min_value=0.0, value=5000.0, step=100.0)
            PriorDefault = st.selectbox("Prior Default", ["No", "Yes"])
        with col4:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            Income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)

        submit = st.form_submit_button("🔮 Predict Credit Approval")

    if submit:
        # Convert categorical choices to model-expected values
        gender_map = {"Male": 0, "Female": 1}
        married_map = {"Single": 0, "Married": 1}
        bank_map = {"No": 0, "Yes": 1}
        industry_map = {"Industry 1": 0, "Industry 2": 1, "Industry 3": 2, "Industry 4": 3}
        ethnicity_map = {"Group 1": 0, "Group 2": 1, "Group 3": 2}
        prior_default_map = {"No": 0, "Yes": 1}
        employed_map = {"No": 0, "Yes": 1}
        drivers_map = {"No": 0, "Yes": 1}
        citizen_map = {"Native": 0, "Foreign": 1, "Other": 2}

        input_data = {
            "Gender": gender_map[Gender],
            "Age": Age,
            "Debt": Debt,
            "Married": married_map[Married],
            "BankCustomer": bank_map[BankCustomer],
            "Industry": industry_map[Industry],
            "Ethnicity": ethnicity_map[Ethnicity],
            "YearsEmployed": YearsEmployed,
            "PriorDefault": prior_default_map[PriorDefault],
            "Employed": employed_map[Employed],
            "CreditScore": CreditScore,
            "DriversLicense": drivers_map[DriversLicense],
            "Citizen": citizen_map[Citizen],
            "ZipCode": ZipCode,
            "Income": Income
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            if response.status_code == 200:
                result = response.json()
                go_to_result(result['prediction'])
            else:
                st.error("❌ Something went wrong with the prediction. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def result_page():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Result")

    prediction = st.session_state.get('prediction', None)

    if prediction is None:
        st.warning("No prediction found. Please submit the form first.")
        if st.button("⬅️ Back to Form"):
            st.session_state.page = 'form'
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if prediction == "Approved":
        # Balloons animation
        st.balloons()
        # Optional: Snow animation (Streamlit 1.18+)
        try:
            st.snow()
        except Exception:
            pass
        st.markdown(
            """
            <div style="background-color:#e6f7ee; color:#167d55; border-radius:10px; padding:2em; text-align:center;">
                <h2>🎉 Congratulations! 🎉</h2>
                <p style="font-size:1.3em;">Your credit application is <b>APPROVED</b>!</p>
                <p>Welcome to a brighter financial future. 🥳</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#fff0f0; color:#d7263d; border-radius:10px; padding:2em; text-align:center;">
                <h2>😞 Application Not Approved</h2>
                <p style="font-size:1.2em;">Unfortunately, your credit application was <b>not approved</b> at this time.</p>
                <p>We encourage you to review your details and try again in the future.</p>
            </div>
            """, unsafe_allow_html=True
        )

    if st.button("⬅️ Back to Form"):
        st.session_state.page = 'form'

    st.markdown('</div>', unsafe_allow_html=True)

# ======================== Page Router ========================
if st.session_state.page == 'form':
    credit_form()
elif st.session_state.page == 'result':
    result_page()
