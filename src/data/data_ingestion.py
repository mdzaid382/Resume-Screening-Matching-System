import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.connections import s3_connection
from src.utils.utils import load_params, save_data

pd.set_option("future.no_silent_downcasting", True)
load_dotenv()


class DataIngestion:
    """Data ingestion pipeline."""

    def __init__(
        self,
        params_path: str = "params.yaml",
        output_dir: str = "./data/raw"
    ):
        self.params_path = params_path
        self.output_dir = output_dir

        self.s3 = s3_connection.s3_operations(
            bucket_name="resume-screening-and-matching",
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-1",
        )


    def load_data(
        self,
        s3_key: str
    ) -> pd.DataFrame:
        """Load dataset from S3."""
        try:
            df = self.s3.fetch_file_from_s3(s3_key)

            logging.info("Data loaded from S3: %s", s3_key)
            return df

        except Exception as e:
            logging.error("Failed to load data from S3: %s", e)
            raise
        

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
        )

        logging.info(
            "Data split completed. Train: %s, Test: %s",
            train_df.shape,
            test_df.shape,
        )

        return train_df, test_df



    def run(self) -> None:
        """Execute complete ingestion pipeline."""

        try:
            params = load_params(self.params_path)
            test_size = params["data_ingestion"]["test_size"]

            df = self.load_data("data/resume_data.csv")

            train_df, test_df = self.split_data(
                df=df,
                test_size=test_size,
            )

            save_data(
                train_df=train_df,
                test_df=test_df,
                output_dir=self.output_dir,
                train_file_name="train.csv",
                test_file_name="test.csv"
            )

            logging.info("Data ingestion completed successfully.")

        except Exception as e:
            logging.error(
                "Data ingestion pipeline failed: %s",
                e,
            )
            raise


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()