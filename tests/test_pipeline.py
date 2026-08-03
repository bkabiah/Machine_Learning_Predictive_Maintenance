import pytest
import pandas as pd
import numpy as np

def test_data_shape():
    """Testet, ob der geladene Datensatz die erwartete Form hat."""
    from src.data_loader import load_data
    df = load_data()
    assert df.shape[0] == 10000, "Datensatz sollte 10.000 Zeilen haben"
    assert 'Machine failure' in df.columns, "Target Spalte fehlt"

def test_preprocessing_output_shape():
    """Testet, ob das Preprocessing die richtigen Dimensionen ausgibt."""
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    
    df = load_data().head(100) # Nur 100 Zeilen für schnellen Test
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    
    # OneHotEncoder für 'Type' (3 Kategorien - 1 = 2 Spalten) + 5 numerische = 7 Features
    assert X_train.shape[1] == 7, f"Erwartete 7 Features, got {X_train.shape[1]}"
    assert X_train.shape[0] == 80, "80% sollten Trainingsdaten sein"
