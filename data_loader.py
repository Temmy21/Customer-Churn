import pandas as pd
from config import DATA_PATH

def load_data(file_path=DATA_PATH):
    """Load the dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None