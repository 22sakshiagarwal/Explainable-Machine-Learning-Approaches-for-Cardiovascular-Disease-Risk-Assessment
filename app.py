import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------
# Load model
# ------------------------------------
model = joblib.load("rf_model.pkl")

st.title("Cardiovascular Disease Risk Assessment")
st.write("Prediction + Explainable AI (SHAP)")

# ------------------------------------
# SHAP Explainer (cached for stability)
# ------------------------------------
@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

explainer = load_shap_explainer()

# ------------------------------------
# User Inputs
# ------------------------------------
st.header("Enter Patient Details")

age = st.number_input("Age (in days)", min_value=5000, max_value=30000, step=1)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", 100, 250, step=1)
weight = st.number_input("Weight (kg)", 20, 200, step=1)
ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 250, step=1)
ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 180, step=1)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
gluc = st.selectbox("Glucose Level", [1, 2, 3])
smoke = st.selectbox("Smokes?", [0, 1])
alco = st.selectbox("Consumes Alcohol?", [0, 1])
active = st.selectbox("Physically Active?", [0, 1])

# ------------------------------------
# Derived Features
# ------------------------------------
bmi = weight / ((height / 100) ** 2)
gender_2 = 1 if gender == "Male" else 0

# BP Category logic
if ap_hi < 120 and ap_lo < 80:
    bp_category = "Normal"
elif (120 <= ap_hi < 130) and (ap_lo < 80):
    bp_category = "Elevated"
elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
    bp_category = "Hypertension Stage 1"
else:
    bp_category = "Hypertension Stage 2"

bp_category_encoded = bp_category

# ------------------------------------
# One-hot encode EXACT model columns
# ------------------------------------
input_dict = {
    "id": 0,
    "age": age,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "bmi": bmi,
    "gender_2": gender_2,

    "bp_category_Elevated": 0,
    "bp_category_Hypertension Stage 1": 0,
    "bp_category_Hypertension Stage 2": 0,
    "bp_category_Normal": 0,

    "bp_category_encoded_Elevated": 0,
    "bp_category_encoded_Hypertension Stage 1": 0,
    "bp_category_encoded_Hypertension Stage 2": 0,
    "bp_category_encoded_Normal": 0,
}

# Set dummy columns
input_dict[f"bp_category_{bp_category}"] = 1
input_dict[f"bp_category_encoded_{bp_category_encoded}"] = 1

model_columns = [
    'id', 'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active', 'bmi', 'gender_2',
    'bp_category_Elevated', 'bp_category_Hypertension Stage 1',
    'bp_category_Hypertension Stage 2', 'bp_category_Normal',
    'bp_category_encoded_Elevated', 'bp_category_encoded_Hypertension Stage 1',
    'bp_category_encoded_Hypertension Stage 2', 'bp_category_encoded_Normal'
]

input_data = pd.DataFrame([input_dict], columns=model_columns)

# ------------------------------------
# Prediction + SHAP
# ------------------------------------
if st.button("Predict"):

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk ({probability*100:.2f}%)")
    else:
        st.success(f"Low Risk ({probability*100:.2f}%)")

    # SHAP Explainability
    st.header("Explainability (SHAP)")

    shap_values = explainer.shap_values(input_data)

    # Stable SHAP plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
    st.pyplot(fig)

    plt.close("all")
