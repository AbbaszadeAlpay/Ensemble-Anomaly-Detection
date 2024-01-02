import gzip
from pathlib import Path

import joblib

from src.dataset_io import DatasetIO
from src.model_training import AnomalyDetection

BASE_DIR = Path(__file__).resolve(strict=True).parent


df = DatasetIO(f"{BASE_DIR}/data_source/transactions.csv").read_data()

anomaly_model = AnomalyDetection(df, contamination=0.05)

anomaly_model.fit()


with gzip.open(f"{BASE_DIR}/models/anomaly_model.gz", "wb") as file:
    joblib.dump(anomaly_model, file)
