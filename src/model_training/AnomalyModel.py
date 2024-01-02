import logging

import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from tqdm import tqdm

from src.dataset_io import Preprocess


class AnomalyDetection:

    """Anomaly detection class using multiple models.

    Parameters:
    - train_data: Training data for model fitting.
    - contamination (float): Proportion of outliers in the data.
    """

    def __init__(self, train_data, contamination):
        self.contamination = contamination
        self.preprocessor = Preprocess()
        self.train_data = self.preprocessor.transform(train_data)
        self.models = [
            HBOS(contamination=self.contamination),
            IForest(contamination=self.contamination),
            PCA(contamination=self.contamination),
            KNN(contamination=self.contamination),
            CBLOF(contamination=self.contamination),
        ]

    def fit(self):
        """Fit anomaly detection models to the training data."""
        for model in tqdm(self.models, desc="Fitting Models", unit="model"):
            model.fit(self.train_data)
            logging.info(f"Model {type(model).__name__} fitted successfully.")

    def predict(self, test_data):
        """Make predictions on the test data using an ensemble of models.

        Parameters:
        - test_data: Test data for making predictions.

        Returns:
        - numpy.ndarray: Predictions (1 for anomaly, 0 for normal).
        """
        test_data = self.preprocessor.transform(test_data)
        predictions = np.zeros((len(test_data), len(self.models)))

        for idx, model in enumerate(
            tqdm(self.models, desc="Making Predictions", unit="model")
        ):
            predictions[:, idx] = model.predict(test_data)
            logging.info(f"Predictions using {type(model).__name__} complete.")

        ensemble_pred = np.sum(predictions, axis=1)
        final_pred = np.where(ensemble_pred >= 3, 1, 0)
        return final_pred


logging.basicConfig(level=logging.INFO)
