# app.py

import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import add_features

model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Heart Disease (CVD) Prediction – Logistic Regression Demo")
st.write("Enter patient data below to predict the risk of heart disease.")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=80, max_value=200, value=130)
chol = st.number_input("Cholesterol level (chol)", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum heart rate (thalach)", min_value=60, max_value=210, value=150)
exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of the ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0=unknown, 1=fixed defect, 2=normal, 3=reversible defect)", [0, 1, 2, 3])

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    input_df = add_features(input_df)
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= 0.5)

    st.subheader("Prediction Result:")
    st.write(f"Estimated probability of heart disease: **{prob*100:.2f}%**")

    if pred == 1:
        st.error("High risk of heart disease!")
    else:
        st.success("Low risk of heart disease.")