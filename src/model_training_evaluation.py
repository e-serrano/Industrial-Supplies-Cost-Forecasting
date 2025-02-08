from data_preprocessing import importing_company_data
from feature_engineering import feature_engineering_training
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_evaluate_model(model_to_train):
    """
    This function trains a machine learning model to predict the unit value
    of a product based on historical data.

    Args:
        model_to_train: Machine learning model to train

    Returns:
        model: Trained machine learning model
        average_metrics: Dictionary with the average metrics obtained
    """

# Data preprocessing and feature engineering
    processed_dataframe = importing_company_data()
    final_dataframe = feature_engineering_training(processed_dataframe)

# Split the data into training and test sets
    X = final_dataframe[['quantity', 'price_change_rate', 'supplier_encoded', 'supply_ref_encoded', 'lead_time', 'month', 'year']]
    y = final_dataframe['unit_value']

    n_splits = 5  # Number of divisions (folds)
    tscv = TimeSeriesSplit(n_splits=n_splits)

# Define the model
    model = model_to_train

# Initialise a dictionary to store the average metrics for each model.
    metrics = {"MAE": [], "RMSE": [], "R2": []}

# Train and evaluate the model using Time Series Cross Validation
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    # Divide data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train and evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

    # Store metrics
        metrics["MAE"].append(mae)
        metrics["RMSE"].append(rmse)
        metrics["R2"].append(r2)

# Average metrics
    avg_mae = np.mean(metrics["MAE"])
    avg_rmse = np.mean(metrics["RMSE"])
    avg_r2 = np.mean(metrics["R2"])

    average_metrics = {"MAE": [], "RMSE": [], "R2": []}
    average_metrics["MAE"] = avg_mae
    average_metrics["RMSE"] = avg_rmse
    average_metrics["R2"] = avg_r2

    return model, average_metrics