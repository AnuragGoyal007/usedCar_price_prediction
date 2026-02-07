import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("car_price_model.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.write("Enter car details to predict the selling price (in Lakhs)")

# Input fields
myear = st.number_input("Manufacturing Year", min_value=1995, max_value=2025, value=2018)
km = st.number_input("Kilometers Driven", min_value=0, value=40000)
engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, value=1197)
power_bhp = st.number_input("Power (bhp)", min_value=30, value=82)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, value=18.0)
tt = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "myear": myear,
        "km": km,
        "engine_cc": engine_cc,
        "power_bhp": power_bhp,
        "mileage": mileage,
        "tt": tt
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: **₹ {prediction:.2f} Lakh**")
