import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Agriculture Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    text-align:center;
    color:#4CAF50;
}

.stButton > button {
    background-color:#2E7D32;
    color:white;
    border-radius:12px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

[data-testid="stMetric"] {
    background-color:#1E1E1E;
    border:1px solid #2E7D32;
    padding:20px;
    border-radius:15px;
}

[data-testid="stMetricLabel"] {
    color:white;
}

[data-testid="stMetricValue"] {
    color:#4CAF50;
    font-size:40px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)
   

# -------------------------------
# LOAD MODEL
# -------------------------------
artifacts = joblib.load("agriculture_complete_model.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]

# -------------------------------
# HEADER
# -------------------------------
st.title("🌾 Smart Agriculture Yield Prediction System")

st.markdown("""
<div style="
background: linear-gradient(135deg,#1b5e20,#2e7d32);
padding:20px;
border-radius:15px;
margin-bottom:20px;
">

<h3 style="color:white;margin-bottom:10px;">
🌾 AI Powered Agriculture Yield Prediction
</h3>

<p style="color:white;font-size:16px;">
Predict crop yield using an Ensemble Machine Learning Model
(Stacking Regressor) trained on agricultural data.
</p>

<p style="color:#dcedc8;">
⚡ State-Based Prediction &nbsp;&nbsp; | &nbsp;&nbsp;
🌦 Auto Season Detection &nbsp;&nbsp; | &nbsp;&nbsp;
🌱 Smart Soil Recommendation
</p>

</div>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:

    st.header("📌 Project Overview")

    st.info("""
Model Type:
Stacking Regressor

Base Models:
• Linear Regression
• Decision Tree
• Random Forest
• Gradient Boosting

Meta Model:
• Ridge Regression
""")

    st.success("Built with Python + Scikit-Learn")

# -------------------------------
# STATE-SOIL MAPPING
# -------------------------------
state_soil = {
    "Bihar": "Alluvial",
    "Punjab": "Alluvial",
    "Haryana": "Alluvial",
    "Uttar Pradesh": "Alluvial",
    "West Bengal": "Alluvial",
    "Rajasthan": "Desert",
    "Maharashtra": "Black",
    "Madhya Pradesh": "Black",
    "Gujarat": "Black",
    "Jharkhand": "Red",
    "Tamil Nadu": "Red",
    "Karnataka": "Red",
    "Odisha": "Laterite",
    "Kerala": "Laterite",
    "Himachal Pradesh": "Mountain",
    "Uttarakhand": "Mountain",
    "Jammu and Kashmir": "Mountain"
}

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:

    state = st.selectbox(
        "Select State",
        [
            'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar',
            'Chhattisgarh','Goa','Gujarat','Haryana',
            'Himachal Pradesh','Jammu and Kashmir',
            'Jharkhand','Karnataka','Kerala',
            'Madhya Pradesh','Maharashtra','Manipur',
            'Meghalaya','Mizoram','Nagaland',
            'Odisha','Punjab','Rajasthan',
            'Sikkim','Tamil Nadu','Telangana',
            'Tripura','Uttar Pradesh',
            'Uttarakhand','West Bengal'
        ]
    )

    month = st.selectbox(
        "Select Month",
        [
            "January","February","March",
            "April","May","June",
            "July","August","September",
            "October","November","December"
        ]
    )

with col2:

    rainfall = st.selectbox(
        "Rainfall Level",
        ["Low", "Medium", "High"]
    )

    farming = st.selectbox(
        "Farming Practice",
        ["Conventional", "Organic"]
    )

# -------------------------------
# AUTO SEASON
# -------------------------------
if month in [
    "October","November","December",
    "January","February","March"
]:
    season = "Rabi"

elif month in [
    "June","July","August","September"
]:
    season = "Kharif"

else:
    season = "Zaid"

st.info(f"🌦 Detected Season: {season}")

# -------------------------------
# AUTO SOIL
# -------------------------------
default_soil = state_soil.get(state, "Alluvial")

soil = st.selectbox(
    "Soil Type",
    ["Alluvial","Black","Desert","Laterite","Mountain","Red"],
    index=[
        "Alluvial","Black","Desert",
        "Laterite","Mountain","Red"
    ].index(default_soil)
)

# -------------------------------
# CROP FILTER
# -------------------------------
crop_mapping = {

    "Rabi": [
        "Wheat",
        "Pulses"
    ],

    "Kharif": [
        "Rice",
        "Maize"
    ],

    "Zaid": [
        "Sugarcane",
        "Millets"
    ]
}

crop = st.selectbox(
    "Select Crop",
    crop_mapping[season]
)

# -------------------------------
# PREDICTION
# -------------------------------
# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Yield"):

    input_df = pd.DataFrame([{
        "State": state,
        "Month": month,
        "Season": season,
        "Crop": crop,
        "Soil_Type": soil,
        "Rainfall_Level": rainfall,
        "Farming_Practice": farming
    }])

    encoded = encoder.transform(input_df)

    prediction = model.predict(encoded)

    st.markdown("---")

    st.metric(
        label="🌾 Predicted Yield",
        value=f"{prediction[0]:.2f} Q/H"
    )

    st.success("Prediction Generated Successfully")

    with st.expander("📋 View Input Summary"):

        st.dataframe(
            input_df,
            use_container_width=True
        )

    st.markdown(
        f"""
        ### 📈 Yield Insight

        Expected agricultural yield is approximately
        **{prediction[0]:.2f} Quintal per Hectare**.

        This prediction is generated using the trained
        Stacking Regressor model.
        """
    )


    st.markdown("---")

st.markdown("""
<div style='text-align:center'>

Built by Ashish Pal

Machine Learning | Data Analytics | 

</div>
""", unsafe_allow_html=True)
      

   