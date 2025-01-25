import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime

def data_preprocessing():
    """
    This function reads the original Excel file with the purchases data,
    processes it and saves the processed data to a new Excel file.
    
    Returns:
        df_purchases (DataFrame): DataFrame with the processed data
    """

# Import data from Excel original file
# This data can be obtained from the company's database
    df_purchases = pd.read_excel("data/raw/COMPRAS.xlsx", index_col=0)

# Delete the columns that are not necessary
    df_purchases = df_purchases.drop(columns=["supplier_order_id","position_supply","supply_id","discount","pending",
                    "deliv_date_1","deliv_quant_1","deliv_note_1",
                    "deliv_date_2","deliv_quant_2","deliv_note_2",
                    "deliv_date_3","deliv_quant_3","deliv_note_3"])

# Fill the NaN values with the current date
    df_purchases['delivery_date'] = df_purchases['delivery_date'].fillna(pd.Timestamp(datetime.now().date()))

# Changing the order of the columns of the dataframe
    new_column_order = ["order_date", "delivery_date", "supplier_name", "supply_reference","unit_value","quantity"]
    df_purchases = df_purchases[new_column_order]

#Calculation of the relative change in the unit price of a product compared to previous purchases
    df_purchases = df_purchases.sort_values(by=['supply_reference', 'order_date'])

# Calculate the previous unit price for each product
    df_purchases['previous_unit_value'] = df_purchases.groupby('supply_reference')['unit_value'].shift(1)

# Calculate the rate of change in the unit price
    df_purchases['price_change_rate'] = ((df_purchases['unit_value'] - df_purchases['previous_unit_value']) / df_purchases['previous_unit_value']) * 100

# Fill the NaN values (which appear for the first purchase of each product) with 0 or an appropriate value
    df_purchases['price_change_rate'] = df_purchases['price_change_rate'].fillna(0)

# Verify if infinite or NaN values in new column and replacing them
    num_infinite_values = np.isinf(df_purchases['price_change_rate']).sum()
    num_nan_values = df_purchases['price_change_rate'].isnull().sum()

    if num_infinite_values > 0:
        df_purchases['price_change_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)

    mean_value = df_purchases['price_change_rate'].mean()
    df_purchases['price_change_rate'].fillna(mean_value, inplace=True)

# Save the dataframe to a new Excel file
    df_purchases.to_excel("data/processed/COMPRAS_processed.xlsx")

    return df_purchases