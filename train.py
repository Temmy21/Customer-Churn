from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from preprocessor import preprocess_data
from feature_engineering import engineer_features
from model import train_model, evaluate_model, save_model
from config import TEST_SIZE, RANDOM_STATE, SCALER_PATH

def run_training_pipeline():
    """Execute the complete training pipeline"""
    # Load data
    df = load_data()
    if df is None:
        return None
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Engineer features
    X, y = engineer_features(df_processed)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    save_model(scaler, SCALER_PATH)
    
    # Train and evaluate model
    model = train_model(X_train_scaled, y_train)
    report, conf_matrix = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model
    save_model(model)
    
    return model, scaler, report, conf_matrix