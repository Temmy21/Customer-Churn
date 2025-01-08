import os

# Configuration settings
DATA_PATH = 'data/Expresso_churn_dataset.csv'
MODEL_PATH = 'models/churn_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 10
MAX_DEPTH = 10

# Feature lists
NUMERIC_FEATURES = [
    'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
    'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK'
]

CATEGORICAL_FEATURES = ['REGION', 'TOP_PACK', 'MRG']

TARGET = 'CHURN'