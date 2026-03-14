import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and features
model = joblib.load("car_price_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Predict resale value using Machine Learning")

# User Inputs
brand = st.selectbox("Brand", ["Maruti","Hyundai","Honda","Toyota","Tata","Mahindra","BMW","Audi","Mercedes"])
model_name = st.text_input("Model Name")

seller_type = st.selectbox("Seller Type", ["Individual","Dealer","Trustmark Dealer"])
fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric"])
transmission = st.selectbox("Transmission", ["Manual","Automatic"])

vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=3)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=30000)

mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=20.0)
engine = st.number_input("Engine CC", min_value=600, max_value=6000, value=1200)
max_power = st.number_input("Max Power (BHP)", min_value=20.0, max_value=500.0, value=80.0)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

# Feature Engineering
km_per_year = km_driven / (vehicle_age + 1)
power_to_engine = max_power / engine

luxury_brands = ['BMW','Audi','Mercedes','Jaguar','Volvo','Lexus','Land Rover']
is_luxury = 1 if brand in luxury_brands else 0

# Create Input DataFrame
input_dict = {
    "vehicle_age": vehicle_age,
    "km_driven": km_driven,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats,
    "km_per_year": km_per_year,
    "power_to_engine": power_to_engine,
    "is_luxury": is_luxury
}

# Add categorical dummies
for col in features:
    if col.startswith("brand_"):
        input_dict[col] = 1 if col == f"brand_{brand}" else 0
        
    if col.startswith("seller_type_"):
        input_dict[col] = 1 if col == f"seller_type_{seller_type}" else 0
        
    if col.startswith("fuel_type_"):
        input_dict[col] = 1 if col == f"fuel_type_{fuel_type}" else 0
        
    if col.startswith("transmission_type_"):
        input_dict[col] = 1 if col == f"transmission_type_{transmission}" else 0

# Fill missing columns
for col in features:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[features]

# Predict
if st.button("Predict Price"):

    pred_log = model.predict(input_df)[0]
    price = np.expm1(pred_log)

    st.success(f"Estimated Price: ₹ {price:,.0f}")

