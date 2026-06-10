import os
import pickle
import yaml
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC

from src.logger import logging
from src.utils.utils import save_model, load_params

class ModelTrainer:
    """Train and save the machine learning model."""

    def __init__(
        self,
        params_path: str = "params.yaml",
        input_path: str = "./data/processed/train_tfidf.csv",
        model_dir: str = "./models",
        file_path: str = "./models/model.pkl"
    ):
        self.params_path = params_path
        self.input_path = input_path
        self.model_dir = model_dir
        self.file_path = file_path


    def load_train_data(self) -> pd.DataFrame:
        """Load training data."""

        try:
            df = pd.read_csv(self.input_path)

            logging.info(
                "Training data loaded from %s",
                self.input_path,
            )

            return df

        except Exception as e:
            logging.error(
                "Failed to load training data: %s",
                e,
            )
            raise

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        C: float,
        penalty: str,
    ) -> LinearSVC:
        """Train LinearSVC model."""

        try:
            model = LinearSVC(
                C=C,
                penalty=penalty,
            )

            model.fit(
                X_train,
                y_train,
            )

            logging.info(
                "Model training completed"
            )

            return model

        except Exception as e:
            logging.error(
                "Model training failed: %s",
                e,
            )
            raise

 
    def run(self) -> None:
        """Execute model training pipeline."""

        try:
            params = load_params("params.yaml")

            C = params["model_building"]["C"]
            penalty = params["model_building"]["penalty"]

            train_df = self.load_train_data()

            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values

            model = self.train_model(
                X_train=X_train,
                y_train=y_train,
                C=C,
                penalty=penalty,
            )

            save_model(model,self.model_dir,self.file_path)

            logging.info(
                "Model training pipeline completed successfully"
            )

        except Exception as e:
            logging.error(
                "Model training pipeline failed: %s",
                e,
            )
            raise


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()