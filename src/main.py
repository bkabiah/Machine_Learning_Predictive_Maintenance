from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model, save_model

def main():
    print("Starte Predictive Maintenance Pipeline...")
    
    # 1. Daten laden
    df = load_data()
    
    # 2. Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # 3. Training
    model = train_model(X_train, y_train)
    
    # 4. Evaluation
    f1 = evaluate_model(model, X_test, y_test)
    print(f"Erreichter F1-Score für 'Machine Failure': {f1:.4f}")
    
    # 5. Speichern
    save_model(model, preprocessor)

if __name__ == "__main__":
    main()
