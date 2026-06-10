import os
import json
import pickle
import mlflow
import numpy as np
import pandas as pd
from typing import Union
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.utils.utils import save_model


class ModelEvaluator:

    def __init__(
        self,
        model_dir: str = "./models",
        data_dir: str = "./data/processed",
        reports_dir: str = "./reports",
        file_path: str = "./models/pipeline.pkl"
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.file_path = file_path

        os.makedirs(
            self.reports_dir,
            exist_ok=True
        )

        self.setup_mlflow()

    def setup_mlflow(self) -> None:
        """Configure MLflow tracking."""

        load_dotenv()

        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        

        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "mdzaid382"
        repo_name = "Resume-Screening-Matching-System"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

    def load_model(
            self,
            file_path: str
        ) -> Union[LinearSVC, TfidfVectorizer]:
        """Load model from disk."""

        try:
            with open(file_path, "rb") as file:
                model = pickle.load(file)

            logging.info(
                "Model loaded from %s",
                file_path
            )

            return model

        except Exception as e:
            logging.error(
                "Failed to load model: %s",
                e
            )
            raise

    def load_test_data(
        self,
        file_path: str
    ) -> pd.DataFrame:
        """Load test dataset."""

        try:
            df = pd.read_csv(file_path)

            logging.info(
                "Data loaded from %s",
                file_path
            )

            return df

        except Exception as e:
            logging.error(
                "Failed to load data: %s",
                e
            )
            raise

    def evaluate_model(
        self,
        clf,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Calculate evaluation metrics."""

        try:
            y_pred = clf.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),
                "precision": precision_score(
                    y_test,
                    y_pred,
                    average="weighted"
                ),
                "recall": recall_score(
                    y_test,
                    y_pred,
                    average="weighted"
                ),
                "f1_score": f1_score(
                    y_test,
                    y_pred,
                    average="weighted"
                )
            }

            logging.info(
                "Model evaluation completed"
            )

            return metrics

        except Exception as e:
            logging.error(
                "Evaluation failed: %s",
                e
            )
            raise

    def save_metrics(
        self,
        metrics: dict
    ) -> None:
        """Save metrics."""

        try:
            metrics_path = os.path.join(
                self.reports_dir,
                "metrics.json"
            )

            with open(
                metrics_path,
                "w"
            ) as file:
                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            logging.info(
                "Metrics saved to %s",
                metrics_path
            )

        except Exception as e:
            logging.error(
                "Failed to save metrics: %s",
                e
            )
            raise

    def save_model_info(
        self,
        run_id: str,
        model_path: str
    ) -> None:
        """Save experiment information."""

        try:
            info_path = os.path.join(
                self.reports_dir,
                "experiment_info.json"
            )

            model_info = {
                "run_id": run_id,
                "model_path": model_path
            }

            with open(
                info_path,
                "w"
            ) as file:
                json.dump(
                    model_info,
                    file,
                    indent=4
                )

            logging.info(
                "Experiment info saved"
            )

        except Exception as e:
            logging.error(
                "Failed to save experiment info: %s",
                e
            )
            raise

    def log_to_mlflow(
        self,
        pipeline,
        clf,
        metrics: dict
    ) -> None:
        """Log metrics, params, and model."""

        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    metric_name,
                    metric_value
                )

            if hasattr(clf, "get_params"):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(
                        param_name,
                        param_value
                    )

            mlflow.sklearn.log_model(
                pipeline,
                "model"
            )

            mlflow.log_artifact(
                os.path.join(
                    self.reports_dir,
                    "metrics.json"
                )
            )

        except Exception as e:
            logging.error(
                "MLflow logging failed: %s",
                e
            )
            raise

    def run(self) -> None:
        """Execute evaluation pipeline."""

        try:
            mlflow.set_experiment(
                "my_dvc_pipeline"
            )

            with mlflow.start_run() as run:

                vectorizer = self.load_model(
                    os.path.join(
                        self.model_dir,
                        "vectorizer.pkl"
                    )
                )

                clf = self.load_model(
                    os.path.join(
                        self.model_dir,
                        "model.pkl"
                    )
                )

                pipeline = Pipeline([
                    ("vectorizer", vectorizer),
                    ("model", clf)
                ])

                test_data = self.load_test_data(
                    os.path.join(
                        self.data_dir,
                        "test_tfidf.csv"
                    )
                )

                X_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values

                metrics = self.evaluate_model(
                    clf,
                    X_test,
                    y_test
                )

                self.save_metrics(
                    metrics
                )

                save_model(
                    pipeline,
                    self.model_dir,
                    self.file_path                   
                    )
                

                self.log_to_mlflow(
                    pipeline,
                    clf,
                    metrics
                )

                self.save_model_info(
                    run.info.run_id,
                    "model"
                )

                logging.info(
                    "Model evaluation pipeline completed successfully"
                )

        except Exception as e:
            logging.error(
                "Model evaluation pipeline failed: %s",
                e
            )
            raise


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()