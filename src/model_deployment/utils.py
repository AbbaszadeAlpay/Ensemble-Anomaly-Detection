from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / "models" / "anomaly_model.gz"

with open(model_path, "rb") as file:
    model = joblib.load(file)

classes = {0: "Normal", 1: "Anomaly"}


def model_predict(data):
    """Make predictions on the test data using an ensemble of models.

    Parameters:
    - test_data: Test data for making predictions.
    """

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame.from_dict(data, orient="index").T
    pred = model.predict(data)
    return classes[pred[0]]
