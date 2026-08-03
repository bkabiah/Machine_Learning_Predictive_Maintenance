
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df: pd.DataFrame):
    """
    Führt Feature Engineering und Preprocessing durch.
    """
    # Unnötige Spalten entfernen (UDI und Product ID haben keine Aussagekraft)
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # Target Variable definieren
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']
    
    # Split (Stratified, weil die Klassen extrem unbalanciert sind)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Spalten definieren
    numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    categorical_features = ['Type']
    
    # Preprocessor bauen
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Fit nur auf Trainingsdaten!
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor