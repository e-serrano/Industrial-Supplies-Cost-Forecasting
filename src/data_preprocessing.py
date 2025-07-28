import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def importing_company_data():
    """
    This function imports the data from an Excel file and processes it.

    Returns:
        df_imported (DataFrame): DataFrame with the original data
    """

# Import data from Excel original file
# This data can be obtained from the company's database
    df_imported = pd.read_excel("data/raw/raw_data.xlsx", index_col=0)

    df_preprocessed = data_preprocessing(df_imported)

    return df_preprocessed


def data_preprocessing(df_to_process):
    """
    This function reads the original Excel file with the purchases data,
    processes it and saves the processed data to a new Excel file.

    Args:
        df_to_process (DataFrame): DataFrame with the original data

    Returns:
        df_purchases (DataFrame): DataFrame with the processed data
    """
    df_purchases = df_to_process.copy()

# Select the item "TA 1/2"NPT-M 316/316L" which is the real product (not extra cargos) with the most number of samples
    df_purchases = df_purchases[df_purchases['supply_reference'] == 'TA 1/2"NPT-M 316/316L']

# Deleting column of product reference
    df_purchases = df_purchases.drop(columns=["supply_reference"])

# Setting date columns as datetime and unit_value as float
    df_purchases["delivery_date"] = pd.to_datetime(df_purchases["delivery_date"], errors='coerce')
    df_purchases["order_date"] = pd.to_datetime(df_purchases["order_date"], errors='coerce')
    df_purchases['unit_value'] = df_purchases['unit_value'].astype(str).str.replace(',', '.').astype(float)

# Change the order of columns in dataframe
    new_column_order = ["order_date", "delivery_date", "supplier_name","unit_value","quantity"]
    df_purchases = df_purchases[new_column_order]

# Sorting dataframe by order date
    df_purchases = df_purchases.sort_values(by=['order_date'])

# Fill data for items not delivered with the last day of working before christmas holidays
    df_purchases['delivery_date'] = df_purchases['delivery_date'].fillna(pd.Timestamp('2024-12-20'))

# Changing suppliers names to a simply version
    df_purchases.replace('JD DELCORTE s.a.s.', 'DELCORTE', inplace=True)
    df_purchases.replace('SIDSA-Suministros Industrialesm', 'SIDSA', inplace=True)
    df_purchases.replace('Mecánica Egarense, S.A.', 'EGARENSE', inplace=True)

# Save the dataframe to a new Excel file
    df_purchases.to_excel("data/processed/processed_data.xlsx")

    return df_purchases


def synthetic_data_creation():
    """
    This function creates synthetic data based on the real data to simulate future orders.
    
    Returns:
        df_synthetic (DataFrame): DataFrame with the synthetic data
    """

    df_raw_data = importing_company_data()
# Calculate trend with regression on monthly mean
    monthly_avg = df_raw_data.groupby(pd.Grouper(key='order_date', freq='ME'))['unit_value'].mean().interpolate()
    X = np.arange(len(monthly_avg)).reshape(-1, 1)
    y = monthly_avg.values
    model = LinearRegression().fit(X, y)

# Create 450 random order dates
    n_samples = 450
    start_date = df_raw_data['order_date'].min()
    end_date = df_raw_data['order_date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Remove real order dates from the date range
    real_dates = df_raw_data['order_date'].dt.date.unique()  # Only the unique dates
    date_range = date_range[~date_range.isin(real_dates)]  # Exclude real dates

# Generate unique random order dates
    order_dates = np.random.choice(date_range, size=n_samples, replace=False)

# Calculate temporal position to apply trend
    order_dates_sorted = np.sort(order_dates)
    relative_months = ((pd.to_datetime(order_dates_sorted) - pd.to_datetime(start_date)) / pd.Timedelta(days=30)).astype(int)
    relative_months = pd.Series(relative_months).clip(0, len(monthly_avg) - 1)
    trend = model.predict(relative_months.values.reshape(-1, 1))

# Simulate quantities (integer values multiples of 50)
    real_quantities = df_raw_data['quantity'].dropna()
    min_q = max((min(real_quantities) // 50) * 50, 50) # Ensure minimum values is 50
    max_q = (max(real_quantities) // 50) * 50
    synthetic_quantities = np.random.choice(np.arange(min_q, max_q + 1, 50), size=n_samples)

# Simulate urgency (delivery days)
    delivery_days = np.random.randint(5, 31, size=n_samples)
    delivery_dates = order_dates_sorted + pd.to_timedelta(delivery_days, unit='D')

# Simulate suppliers
    real_suppliers = df_raw_data['supplier_name'].dropna().unique()
    synthetic_suppliers = np.random.choice(real_suppliers, size=n_samples)

# Parameters of synthetic model
    alpha = 2    # Quantity impact
    beta = 3    # Urgency impact
    noise = np.random.normal(0, 0.15, size=n_samples)

# Calculate unit_value based on realistic logic
    unit_values = trend + alpha * (1 / synthetic_quantities) + beta * (1 / delivery_days) + noise

# Adjust synthetic values to be over real minimum
    scaling_factor = np.mean(df_raw_data['unit_value']) / np.mean(unit_values)
    unit_values_adjusted = unit_values * scaling_factor

# Ensure that the adjusted values do not fall below the actual minimum value
    unit_values_adjusted = np.clip(unit_values_adjusted, np.min(df_raw_data['unit_value']), np.max(df_raw_data['unit_value']))

# Create synthetic DataFrame
    df_synthetic = pd.DataFrame({
        'order_date': order_dates_sorted,
        'delivery_date': delivery_dates,
        'supplier_name': synthetic_suppliers,
        'unit_value': unit_values,
        'quantity': synthetic_quantities,
    })

    df_purchases = pd.concat([df_raw_data, df_synthetic], ignore_index=True)
    df_purchases = df_purchases.sort_values(by=['order_date'])

# Creation of new category based on difference between order date and delivery date
    df_purchases["delivery_days"] = (df_purchases["delivery_date"] - df_purchases["order_date"]).dt.days
    
    return df_purchases