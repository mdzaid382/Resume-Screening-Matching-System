import os
import mlflow
from dotenv import load_dotenv
from src.logger import logging


class ModelPromoter:

    def __init__(
        self,
        model_name: str = "resume_screening_and_matching"
    ):
        self.model_name = model_name
        self.setup_mlflow()

    def setup_mlflow(self) -> None:

        load_dotenv()

        dagshub_token = os.getenv("DAGSHUB_TOKEN")

        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/"
            "mdzaid382/"
            "Resume-Screening-Matching-System.mlflow"
        )

    def promote_challenger(self) -> None:

        client = mlflow.MlflowClient()

        challenger = client.get_model_version_by_alias(
            name=self.model_name,
            alias="challenger"
        )

        client.set_registered_model_alias(
            name=self.model_name,
            alias="champion",
            version=challenger.version
        )

        logging.info(
            "Promoted version %s to champion",
            challenger.version
        )

    def run(self) -> None:

        try:
            self.promote_challenger()

            logging.info(
                "Model promotion completed successfully"
            )

        except Exception as e:
            logging.error(
                "Model promotion failed: %s",
                e
            )
            raise


if __name__ == "__main__":
    promoter = ModelPromoter()
    promoter.run()