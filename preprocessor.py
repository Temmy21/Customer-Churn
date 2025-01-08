import pandas as pd
import numpy as np
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    df = df.copy()
    
    # Handle missing values in numeric columns
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].mean())
    
    # Fill categorical columns with mode
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    # Convert TENURE to numeric
    tenure_mapping = {
        'F 9-12 month': 10.5,
        'G 12-15 month': 13.5,
        'H 15-18 month': 16.5,
        'I 18-21 month': 19.5,
        'J 21-24 month': 22.5,
        'K > 24 month': 26
    }
    df['TENURE_MONTHS'] = df['TENURE'].map(tenure_mapping)
    
    return df
