import pandas as pd
import os

def load_data(url: str = None, local_path: str = "data/predictive_maintenance.csv") -> pd.DataFrame:
    """
    Lädt den Datensatz. Wenn lokal nicht vorhanden, wird er aus dem UCI-Repo geladen.
    """
    if os.path.exists(local_path):
        print("Lade lokale Daten...")
        df = pd.read_csv(local_path)
    else:
        print("Lade Daten aus UCI Repository...")
        # Direkter Link zur CSV im UCI Repo
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        df = pd.read_csv(url)
        # Speichern für nächste Nutzung
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
    
    return df
