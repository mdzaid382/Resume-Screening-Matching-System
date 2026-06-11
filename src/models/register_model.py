import json
import os
import mlflow

from dotenv import load_dotenv
from src.logger import logging


class ModelRegistrar:

    def __init__(
        self,
        model_info_path: str = "reports/experiment_info.json",
        model_name: str = "resume-screening"
    ):
        self.model_info_path = model_info_path
        self.model_name = model_name

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

    def load_model_info(self) -> dict:
        """Load experiment information."""

        try:
            with open(
                self.model_info_path,
                "r"
            ) as file:

                model_info = json.load(file)

            logging.info(
                "Model info loaded from %s",
                self.model_info_path
            )

            return model_info

        except Exception as e:
            logging.error(
                "Failed to load model info: %s",
                e
            )
            raise

    def register_model(
        self,
        model_info: dict
    ) -> None:
        """Register model in MLflow Registry."""

        try:
            model_uri = (
                f"runs:/"
                f"{model_info['run_id']}/"
                f"{model_info['model_path']}"
            )

            logging.info(
                "Registering model from %s",
                model_uri
            )

            client = mlflow.tracking.MlflowClient()

            client.get_run(
                model_info["run_id"]
            )

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )

            version = model_version.version

            logging.info(
                "Model version %s registered successfully",
                version
            )

            try:
                client.get_model_version_by_alias(
                    name=self.model_name,
                    alias="champion"
                )

                client.set_registered_model_alias(
                    name=self.model_name,
                    alias="challenger",
                    version=version
                )

                logging.info(
                    "Assigned alias 'challenger' to version %s",
                    version
                )

            except Exception:

                client.set_registered_model_alias(
                    name=self.model_name,
                    alias="champion",
                    version=version
                )

                logging.info(
                    "First model detected. "
                    "Assigned aliases 'champion' and "
                    "'challenger' to version %s",
                    version
                )

        except Exception as e:
            logging.error(
                "Model registration failed: %s",
                e
            )
            raise

    def run(self) -> None:
        """Execute registration pipeline."""

        try:
            model_info = self.load_model_info()

            self.register_model(
                model_info
            )

            logging.info(
                "Model registration pipeline completed successfully"
            )

        except Exception as e:
            logging.error(
                "Model registration pipeline failed: %s",
                e
            )
            raise


if __name__ == "__main__":
    registrar = ModelRegistrar(
        model_name="my_model"
    )

    registrar.run()