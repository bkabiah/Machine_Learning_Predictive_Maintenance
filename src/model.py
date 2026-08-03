
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
import os

def train_model(X_train, y_train):
    """
    Trainiert einen Random Forest. class_weight='balanced' ist wichtig wegen der unbalancierten Daten!
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced', # Wichtig für Predictive Maintenance!
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluiert das Modell. F1-Score ist die wichtigste Metrik hier.
    """
    y_pred = model.predict(X_test)
    print("\n--- Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred)
    return f1

def save_model(model, preprocessor, model_path="models/rf_model.pkl", prep_path="models/preprocessor.pkl"):
    """
    Speichert Modell und Preprocessor (zwingend nötig für Inference!)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)
    print(f"Modell gespeichert unter {model_path}")