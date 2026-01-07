"""
Preprocessing functions for tabular data used in the XGBoost model.

This module contains the preprocessing logic that transforms raw tabular features
into a format suitable for training the XGBoost baseline model. The preprocessing
includes date handling, feature engineering (age, renovation), and target transformation.
"""

import pandas as pd
import numpy as np


def preprocess_data(df, is_train=True):
    """
    Preprocess tabular data for XGBoost training/inference.
    
    This function handles:
    - Date parsing and extraction (year, month, day)
    - Feature engineering (house age, renovation status, years since update)
    - Zipcode conversion
    - Target transformation (log-price for training)
    - Column cleanup (removes id, date, yr_built, yr_renovated)
    
    Args:
        df (pd.DataFrame): Raw input dataframe with columns including:
            - date, yr_built, yr_renovated, zipcode
            - price (if is_train=True)
        is_train (bool): If True, applies log transformation to 'price' column
    
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for model training/inference
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_out = df.copy()
    
    # --- A. Date Handling ---
    df_out['date'] = pd.to_datetime(df_out['date'])
    df_out['year_sold'] = df_out['date'].dt.year
    df_out['month_sold'] = df_out['date'].dt.month
    df_out['day_sold'] = df_out['date'].dt.day
    
    # --- B. Feature Engineering (Age & Renovation) ---
    ref_year = 2025
    df_out['house_age'] = ref_year - df_out['yr_built']
    df_out['was_renovated'] = (df_out['yr_renovated'] > 0).astype(int)
    last_update = df_out['yr_renovated'].where(df_out['yr_renovated'] != 0, df_out['yr_built'])
    df_out['years_since_update'] = ref_year - last_update
    
    # --- C. Zipcode ---
    df_out['zipcode'] = df_out['zipcode'].astype(int)
    
    # --- D. Target Transformation (Train Only) ---
    if is_train:
        df_out['log_price'] = np.log1p(df_out['price'])
    
    # --- E. Cleanup ---
    cols_to_drop = ['id', 'date', 'yr_built', 'yr_renovated']
    df_out = df_out.drop(columns=cols_to_drop, errors='ignore')
    
    return df_out

