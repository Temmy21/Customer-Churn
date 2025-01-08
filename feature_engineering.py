from sklearn.preprocessing import LabelEncoder
from config import CATEGORICAL_FEATURES

def engineer_features(df):
    """Create new features and prepare data for modeling"""
    df = df.copy()
    
    # Create new features
    df['USAGE_RATIO'] = df['DATA_VOLUME'] / (df['MONTANT'] + 1)
    df['CALL_RATIO'] = (df['ON_NET'] + df['ORANGE'] + df['TIGO']) / (df['MONTANT'] + 1)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[f'{col}_ENCODED'] = le.fit_transform(df[col].astype(str))
    
    # Select features for modeling
    feature_columns = [
        'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
        'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'TENURE_MONTHS',
        'REGULARITY', 'FREQ_TOP_PACK', 'USAGE_RATIO', 'CALL_RATIO',
        'REGION_ENCODED', 'TOP_PACK_ENCODED', 'MRG_ENCODED'
    ]
    
    # Remove any remaining missing values
    df = df.dropna(subset=feature_columns + ['CHURN'])
    
    X = df[feature_columns]
    y = df['CHURN']
    
    return X, y
