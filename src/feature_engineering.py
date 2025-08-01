import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import boxcox
from scipy.special import inv_boxcox


def feature_engineering_sarimax(initial_dataframe):
    """
    This function performs feature engineering for SARIMAX model.
    Args:
        initial_dataframe (DataFrame): DataFrame with the initial data
        
    Returns:
        daily_data (DataFrame): DataFrame with the new features for SARIMAX
    """

    df_sarimax = initial_dataframe.copy()

    start_date = df_sarimax['order_date'].min()
    end_date = df_sarimax['order_date'].max()
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')

    # Daily aggregation
    daily_data = []
    for date in df_sarimax['order_date'].unique():
        day_data = df_sarimax[df_sarimax['order_date'] == date]

        weighted_avg = np.average(day_data['unit_value'], weights=day_data['quantity'])

        daily_record = {
            'order_date': date,
            'unit_value': weighted_avg,
            'quantity': day_data['quantity'].sum(),
            'delivery_days': day_data['delivery_days'].mean(),
            'supplier_name': day_data['supplier_name'].mode()[0] if len(day_data['supplier_name'].mode()) > 0 else day_data['supplier_name'].iloc[0]
        }
        daily_data.append(daily_record)

    daily_data = pd.DataFrame(daily_data)
    daily_data = daily_data.set_index('order_date').sort_index()

    # Detect big gaps
    max_gap_days = 7
    time_diffs = daily_data.index.to_series().diff()
    large_gaps = time_diffs > pd.Timedelta(days=max_gap_days)

    if large_gaps.any():
        print(f"Detectados {large_gaps.sum()} gaps > {max_gap_days} días")
        daily_data['post_gap'] = large_gaps.astype(int)
    else:
        daily_data['post_gap'] = 0

# Fill missing dates
    daily_data['delivery_days'] = daily_data['delivery_days'].fillna(daily_data['delivery_days'].median())

# Encoding categorical variables
    le = LabelEncoder()
    daily_data['supplier_encoded'] = le.fit_transform(daily_data['supplier_name'].astype(str))

# Diagnosis and transformation of target variable
    print("\nEstadísticas de unit_value:")
    print(f"Skew: {daily_data['unit_value'].skew():.3f}")
    print(f"Kurtosis: {daily_data['unit_value'].kurtosis():.3f}")
    print(f"Rango: {daily_data['unit_value'].min():.3f} - {daily_data['unit_value'].max():.3f}")

    # Detect outliers
    Q1 = daily_data['unit_value'].quantile(0.25)
    Q3 = daily_data['unit_value'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = daily_data[(daily_data['unit_value'] < Q1 - 1.5*IQR) |
                        (daily_data['unit_value'] > Q3 + 1.5*IQR)]
    print(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(daily_data)*100:.1f}%)")

# Box-Cox transformation
    if daily_data['unit_value'].min() > 0:
        try:
            unit_value_bc, lambda_val = boxcox(daily_data['unit_value'])
            # Verify box-cox
            from scipy import stats
            _, p_normal_original = stats.jarque_bera(daily_data['unit_value'])
            _, p_normal_bc = stats.jarque_bera(unit_value_bc)

            if p_normal_bc > p_normal_original:
                print(f"Box-Cox mejora normalidad (λ={lambda_val:.3f})")
                daily_data['unit_value_bc'] = unit_value_bc
                target_var = 'unit_value_bc'
                transform_used = ('boxcox', lambda_val)
            else:
                print("Box-Cox no mejora normalidad, usando datos originales")
                target_var = 'unit_value'
                transform_used = None
        except:
            target_var = 'unit_value'
            transform_used = None
    else:
        target_var = 'unit_value'
        transform_used = None

# Exogenous variable scaling
    scaler = StandardScaler()
    exog_vars = ['quantity', 'delivery_days', 'supplier_encoded']
    daily_data[exog_vars] = scaler.fit_transform(daily_data[exog_vars])
    last_date = daily_data.index[-1]

    joblib.dump(le, "models/label_encoders_sarimax.pkl")
    joblib.dump(scaler, "models/scaler_sarimax.pkl")
    joblib.dump({"target_var": target_var, "transform_used": transform_used, "last_date": last_date}, "models/sarimax_meta.pkl")

    return df_sarimax, daily_data, target_var, exog_vars


def feature_engineering_random_forest(initial_dataframe, label_encoders=None, min_date=None):
    df_features = initial_dataframe.copy()

    if label_encoders is None:
        label_encoders = {}

    if min_date is None:
        min_date = df_features['order_date'].min()

    # Temporary Features
    df_features['year'] = df_features['order_date'].dt.year
    df_features['month'] = df_features['order_date'].dt.month
    df_features['day'] = df_features['order_date'].dt.day
    df_features['weekday'] = df_features['order_date'].dt.weekday
    df_features['quarter'] = df_features['order_date'].dt.quarter
    df_features['day_of_year'] = df_features['order_date'].dt.dayofyear

    # Cycle Features (seasonality)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)

    # Urgency Features
    df_features['delivery_days'] = (df_features['delivery_date'] - df_features['order_date']).dt.days
    df_features['urgency_score'] = 1 / df_features['delivery_days']  # more urgently = more score

    # Quantity Features
    df_features['quantity_log'] = np.log(df_features['quantity'])
    df_features['quantity_inv'] = 1 / df_features['quantity']

    # Supplier Features
    if 'supplier_name' in df_features.columns:
        if 'supplier_name' not in label_encoders:
            label_encoders['supplier_name'] = LabelEncoder()
            df_features['supplier_encoded'] = label_encoders['supplier_name'].fit_transform(df_features['supplier_name'])
        else:
            df_features['supplier_encoded'] = label_encoders['supplier_name'].transform(df_features['supplier_name'])

    # Temporary Trend Features (days since start)
    df_features['days_since_start'] = (df_features['order_date'] - min_date).dt.days
    df_features['months_since_start'] = df_features['days_since_start'] / 30.44  # Days per month average

    # Select features for model
    feature_columns = [
        'year', 'month', 'day', 'weekday', 'quarter', 'day_of_year',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'delivery_days', 'urgency_score', 'quantity', 'quantity_log', 'quantity_inv',
        'supplier_encoded', 'days_since_start', 'months_since_start'
    ]

    # Filter columns that exist
    available_features = [col for col in feature_columns if col in df_features.columns]
    feature_names = available_features

    X = df_features[available_features]

    joblib.dump(label_encoders, "models/label_encoders_random_forest.pkl")

    return X, df_features, feature_names, label_encoders


def predict_sarimax(input_df, model, le, scaler, meta):
    # Prepare input data
    row = input_df.iloc[0].copy()
    target_date = pd.Timestamp(row['order_date'])
    delivery_date = pd.Timestamp(row['delivery_date'])
    quantity = row['quantity']
    supplier_name = str(row['supplier_name'])

    # Last historical date
    last_date = meta["last_date"]
    steps = (target_date - last_date).days
    if steps <= 0:
        raise ValueError("La fecha objetivo está en el histórico o antes. Usa el valor real.")

    # Prepare future exogenous variables
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

    # Use last historical values as base
    last_exog = model.model.data.orig_exog[-1, :]
    future_exog = np.tile(last_exog, (steps, 1))
    future_exog = pd.DataFrame(future_exog, index=future_dates, columns=model.model.exog_names)

    # Change values for the target date
    delivery_days = (pd.Timestamp(delivery_date) - target_date).days
    supplier_encoded = le.transform([supplier_name])[0]
    custom_row = pd.DataFrame({
        "quantity": [quantity],
        "delivery_days": [delivery_days],
        "supplier_encoded": [supplier_encoded]
    })
    # Scale custom_row
    custom_row[["quantity", "delivery_days", "supplier_encoded"]] = scaler.transform(custom_row)
    future_exog.iloc[-1] = custom_row.iloc[0]

    # Make forecast
    forecast = model.get_forecast(steps=steps, exog=future_exog)
    pred_mean = forecast.predicted_mean.iloc[-1]

    # Revert box-cox transformation if applied
    transform_used = meta.get("transform_used", None)
    if transform_used and transform_used[0] == "boxcox":
        lambda_val = transform_used[1]
        pred_mean = inv_boxcox(pred_mean, lambda_val)

    # Confidence range
    conf_int = forecast.conf_int().iloc[-1]
    if transform_used and transform_used[0] == "boxcox":
        conf_int = conf_int.apply(lambda x: inv_boxcox(x, lambda_val))

    return {
        "Date": target_date,
        "Prediction": pred_mean,
        "Range_95": (conf_int[0], conf_int[1])
    }