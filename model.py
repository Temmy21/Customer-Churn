from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from config import RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH, MODEL_PATH, SCALER_PATH

def train_model(X_train, y_train):
    """Train the Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return report, conf_matrix

def save_model(model, filename=MODEL_PATH):
    """Save the trained model to a file"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename=MODEL_PATH):
    """Load a trained model from a file"""
    with open(filename, 'rb') as file:
        return pickle.load(file)