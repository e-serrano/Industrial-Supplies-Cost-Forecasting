import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def feature_engineering_training(processed_dataframe):
    """
    This function receives a DataFrame with the processed data and performs feature engineering.

    Args:
        processed_dataframe (DataFrame): DataFrame with the processed data

    Returns:
        processed_dataframe (DataFrame): DataFrame with the new features    
    """

# Coding cathegorical variables.
# Using Target Encoding to establish to each category the mean of target variable
    supplier_avg_cost = processed_dataframe.groupby("supplier_name")["unit_value"].mean().to_dict()
    supply_ref_avg_cost = processed_dataframe.groupby("supply_reference")["unit_value"].mean().to_dict()

# Save the dictionaries to use them in the prediction
    joblib.dump(supplier_avg_cost, "models/supplier_avg_cost.joblib")
    joblib.dump(supply_ref_avg_cost, "models/supply_ref_avg_cost.joblib")

    processed_dataframe["supplier_encoded"] = processed_dataframe["supplier_name"].map(supplier_avg_cost)
    processed_dataframe["supply_ref_encoded"] = processed_dataframe["supply_reference"].map(supply_ref_avg_cost)

# Conversion of date columns to datetime
    processed_dataframe["delivery_date"] = pd.to_datetime(processed_dataframe["delivery_date"])
    processed_dataframe["order_date"] = pd.to_datetime(processed_dataframe["order_date"])

# Creation of new categories for time series
    processed_dataframe["lead_time"] = (processed_dataframe["delivery_date"] - processed_dataframe["order_date"]).dt.days  # Delivery time in days
    processed_dataframe["month"] = processed_dataframe["order_date"].dt.month  # Month of order
    processed_dataframe["year"] = processed_dataframe["order_date"].dt.year # Year of order

# Standarization of numeric columns
# Negative values can be obtained of this transformation
    scaler = StandardScaler()

    processed_dataframe['quantity'] = scaler.fit_transform(processed_dataframe[['quantity']])
    processed_dataframe['lead_time'] = scaler.fit_transform(processed_dataframe[['lead_time']])
    # if 'unit_value' in processed_dataframe.columns:
    #     processed_dataframe['unit_value'] = scaler.fit_transform(processed_dataframe[['unit_value']])

    return processed_dataframe

def feature_engineering_prediction(processed_dataframe, supplier_avg_cost, supply_ref_avg_cost):
    """
    This function receives a DataFrame with the processed data and performs feature engineering.

    Args:
        processed_dataframe (DataFrame): DataFrame with the processed data
        supplier_avg_cost (dict): Dictionary with the average cost of each supplier
        supply_ref_avg_cost (dict): Dictionary with the average cost of each supply reference

    Returns:
        processed_dataframe (DataFrame): DataFrame with the new features    
    """

# Coding cathegorical variables.
# Using Target Encoding to establish to each category the mean of target variable
    processed_dataframe["supplier_encoded"] = processed_dataframe["supplier_name"].map(supplier_avg_cost)
    processed_dataframe["supply_ref_encoded"] = processed_dataframe["supply_reference"].map(supply_ref_avg_cost)

# Conversion of date columns to datetime
    processed_dataframe["delivery_date"] = pd.to_datetime(processed_dataframe["delivery_date"])
    processed_dataframe["order_date"] = pd.to_datetime(processed_dataframe["order_date"])

# Creation of new categories for time series
    processed_dataframe["lead_time"] = (processed_dataframe["delivery_date"] - processed_dataframe["order_date"]).dt.days  # Delivery time in days
    processed_dataframe["month"] = processed_dataframe["order_date"].dt.month  # Month of order
    processed_dataframe["year"] = processed_dataframe["order_date"].dt.year # Year of order

# Standarization of numeric columns
# Negative values can be obtained of this transformation
    scaler = StandardScaler()

    processed_dataframe['quantity'] = scaler.fit_transform(processed_dataframe[['quantity']])
    processed_dataframe['lead_time'] = scaler.fit_transform(processed_dataframe[['lead_time']])

    return processed_dataframe