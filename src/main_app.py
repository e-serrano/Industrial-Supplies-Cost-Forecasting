import streamlit as st
import pandas as pd
import joblib
import os

from feature_engineering import predict_sarimax, feature_engineering_random_forest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load all models and encoders
LR_MODEL_PATH = "models/linear_regresion_model.pkl"
ARIMA_MODEL_PATH = "models/arima_model.pickle"
SES_MODEL_PATH = "models/ses_model.pkl"
SARIMAX_MODEL_PATH = "models/sarimax_model.pickle"
RF_MODEL_PATH = "models/random_forest_model.pkl"

ENCODERS_SARIMAX_PATH = "models/label_encoders_sarimax.pkl"
SCALER_SARIMAX_PATH = "models/scaler_sarimax.pkl"
META_SARIMAX_PATH = "models/sarimax_meta.pkl"

ENCODERS_RANDOM_FOREST_PATH = "models/label_encoders_random_forest.pkl"

DF_MONTHLY_PATH = "models/df_monthly.pkl"

if all([os.path.exists(LR_MODEL_PATH),
        os.path.exists(ARIMA_MODEL_PATH),
        os.path.exists(SES_MODEL_PATH),
        os.path.exists(SARIMAX_MODEL_PATH),
        os.path.exists(RF_MODEL_PATH)]):
    linear_regresion_model = joblib.load(LR_MODEL_PATH)
    # arima_model = joblib.load(ARIMA_MODEL_PATH)
    arima_model = ARIMA.load(ARIMA_MODEL_PATH)
    ses_model = joblib.load(SES_MODEL_PATH)
    # sarimax_model = joblib.load(SARIMAX_MODEL_PATH)
    sarimax_model = SARIMAXResults.load(SARIMAX_MODEL_PATH)
    random_forest_model = joblib.load(RF_MODEL_PATH)

    encoders_sarimax = joblib.load(ENCODERS_SARIMAX_PATH)
    scaler_sarimax = joblib.load(SCALER_SARIMAX_PATH)
    meta_sarimax = joblib.load(META_SARIMAX_PATH)

    encoders_random_forest = joblib.load(ENCODERS_RANDOM_FOREST_PATH)

    df_monthly = joblib.load(DF_MONTHLY_PATH)
else:
    st.error("Model file not found. Please train the model and place it in the 'models' directory.")
    st.stop()

st.title("Industrial Goods Cost Forecasting")
st.write("This application predicts the cost of industrial goods based on historical data, in particular for the supply reference 'TA 1/2\"NPT-M 316/316L'.")

# Sample supplier and supply lists (replace with actual data source)
supplier_list = ["DELCORTE", "SIDSA", "EGARENSE"]

# User inputs
st.sidebar.header("Input Features")
order_date = st.sidebar.date_input("Order Date")
supplier_name = st.sidebar.selectbox("Supplier Name", supplier_list)
delivery_date = st.sidebar.date_input("Delivery Date")
quantity = st.sidebar.number_input("Quantity")

# Create input DataFrame
input_data = pd.DataFrame({
    "order_date": [order_date],
    "supplier_name": [supplier_name],
    "delivery_date": [delivery_date],
    "quantity": [quantity]
})

X_rf, _, _, _ = feature_engineering_random_forest(input_data, label_encoders=encoders_random_forest, min_date=pd.Timestamp("2020-01-01"))

# Make prediction
if st.button("Predict Cost"):
    target_date = pd.Timestamp(input_data.iloc[0]["order_date"])
    
    # Linear Regression Prediction
    date_ordinal = [[target_date.toordinal()]]
    pred_lr = linear_regresion_model.predict(date_ordinal)[0]

    # ARIMA and SES Prediction
    steps = (target_date - df_monthly.index[-1]).days // 30

    if steps > 0:
        pred_ses = ses_model.forecast(steps).iloc[-1]
        pred_arima = arima_model.forecast(steps).iloc[-1]
    else:
        # If date is in historical data
        if target_date in df_monthly.index:
            pred_ses = df_monthly.loc[target_date, "unit_value"]
            pred_arima = df_monthly.loc[target_date, "unit_value"]
        else:
            # If date is before historical data, use first available value
            pred_ses = df_monthly.iloc[0]["unit_value"]
            pred_arima = df_monthly.iloc[0]["unit_value"]
    
    # SARIMAX Prediction
    prediction_sarimax = predict_sarimax(input_data, sarimax_model, encoders_sarimax, scaler_sarimax, meta_sarimax)

    # Random Forest Prediction
    prediction_rf = random_forest_model.predict(X_rf)[0]

    st.success(f"Predicted Cost (LINEAR REGRESSION): {pred_lr:.2f} €")
    st.success(f"Predicted Cost (ARIMA): {pred_arima:.2f} €")
    st.success(f"Predicted Cost (SES): {pred_ses:.2f} €")
    st.success(f"Predicted Cost (SARIMAX): {prediction_sarimax['prediccion']:.2f} €")
    st.success(f"Predicted Cost (RANDOM FOREST): {prediction_rf:.2f} €")
