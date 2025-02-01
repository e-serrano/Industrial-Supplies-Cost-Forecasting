import streamlit as st
import pandas as pd
import joblib
import os
from feature_engineering import feature_engineering

# Load the trained model
MODEL_PATH = "models/best_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found. Please train the model and place it in the 'models' directory.")
    st.stop()

st.title("Industrial Goods Cost Forecasting")
st.write("This application predicts the cost of industrial goods based on historical data.")

# Sample supplier and supply lists (replace with actual data source)
supplier_list = ["Supplier A", "Supplier B", "Supplier C"]
supply_list = ["Supply X", "Supply Y", "Supply Z"]

# User inputs
st.sidebar.header("Input Features")
order_date = st.sidebar.date_input("Order Date")
supplier_name = st.sidebar.selectbox("Supplier Name", supplier_list)
supply_reference = st.sidebar.selectbox("Supply Reference", supply_list)
unit_value = st.sidebar.number_input("Unit Value (€)", min_value=0.0, value=10.0)
delivery_date = st.sidebar.date_input("Delivery Date")
quantity = st.sidebar.number_input("Quantity", min_value=1, value=10)

# Create input DataFrame
input_data = pd.DataFrame({
    "order_date": [order_date],
    "supplier_name": [supplier_name],
    "supply_reference": [supply_reference],
    "unit_value": [unit_value],
    "delivery_date": [delivery_date],
    "quantity": [quantity]
})

# Preprocess input
processed_input = feature_engineering(input_data)

# Make prediction
if st.button("Predict Cost"):
    prediction = model.predict(processed_input)[0]
    st.success(f"Predicted Cost: €{prediction:.2f}")
