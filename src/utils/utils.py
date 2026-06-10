import os
import pickle
import yaml
import pandas as pd
from src.logger import logging


def save_model(
    model,
    model_dir,
    file_path: str,
) -> None:
    """Save model to disk."""

    try:
        os.makedirs(
            os.path.dirname(model_dir),
            exist_ok=True,
        )

        with open(
            file_path,
            "wb",
        ) as file:
            pickle.dump(
                model,
                file,
            )

        logging.info(
            "Model saved to %s",
            file_path,
        )

    except Exception as e:
        logging.error(
            "Failed to save model: %s",
            e,
        )
        raise


def load_params(params_path) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(
                params_path, "r"
            ) as file:
                params = yaml.safe_load(file)

                logging.info("Parameters loaded from %s", params_path)
                return params

    except FileNotFoundError:
        logging.error(
            "File not found: %s", params_path
        )
        raise

    except yaml.YAMLError as e:
        logging.error(
            "YAML error: %s", e
        )
        raise


def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    train_file_name: str,
    test_file_name: str
    ) -> None:
    """Save train and test datasets."""

    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(
            os.path.join(output_dir, train_file_name),
            index=False,
        )
        test_df.to_csv(
            os.path.join(output_dir, test_file_name),
            index=False,
        )
        logging.info(
            "Train and test datasets saved to %s",
            output_dir,
        )

    except Exception as e:
        logging.error("Failed to save datasets: %s", e)
        raise


def load_data(data_dir,train_file_name, test_file_name) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    train_df = pd.read_csv(
        os.path.join(data_dir, train_file_name)
    )
    train_df.fillna("", inplace=True)
    test_df = pd.read_csv(
        os.path.join(data_dir, test_file_name)
    )
    logging.info("datasets loaded successfully from %s", data_dir)
    test_df.fillna("", inplace=True)
    return train_df, test_df

    

    