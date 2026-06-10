import os
import pickle
import yaml
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging
from src.utils.utils import load_data, save_data, load_params


class FeatureEngineering:
    """Feature engineering pipeline using TF-IDF."""

    def __init__(
        self,
        params_path: str = "params.yaml",
        input_dir: str = "./data/interim",
        output_dir: str = "./data/processed",
        model_dir: str = "./models",
    ):
        self.params_path = params_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = model_dir


    def apply_tfidf(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        max_features: int,
        ngram_range: tuple[int, int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply TF-IDF vectorization."""

        try:
            logging.info("Applying TF-IDF vectorization")

            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
            )

            X_train = train_df["Resume"]
            y_train = train_df["Role"]

            X_test = test_df["Resume"]
            y_test = test_df["Role"]

            X_train_tfidf = vectorizer.fit_transform(
                X_train
            )

            X_test_tfidf = vectorizer.transform(
                X_test
            )

            train_features = pd.DataFrame(
                X_train_tfidf.toarray()
            )

            train_features["label"] = y_train.values

            test_features = pd.DataFrame(
                X_test_tfidf.toarray()
            )

            test_features["label"] = y_test.values

            self.save_vectorizer(vectorizer)

            logging.info(
                "TF-IDF transformation completed"
            )

            return train_features, test_features

        except Exception as e:
            logging.error(
                "TF-IDF transformation failed: %s",
                e,
            )
            raise

    def save_vectorizer(
        self,
        vectorizer: TfidfVectorizer,
    ) -> None:
        """Save trained vectorizer."""

        try:
            os.makedirs(
                self.model_dir,
                exist_ok=True,
            )

            vectorizer_path = os.path.join(
                self.model_dir,
                "vectorizer.pkl",
            )

            with open(
                vectorizer_path,
                "wb",
            ) as file:
                pickle.dump(
                    vectorizer,
                    file,
                )

            logging.info(
                "Vectorizer saved to %s",
                vectorizer_path,
            )

        except Exception as e:
            logging.error(
                "Failed to save vectorizer: %s",
                e,
            )
            raise

    
    def run(self) -> None:
        """Execute feature engineering pipeline."""

        try:
            params = load_params("params.yaml")

            max_features = params[
                "feature_engineering"
            ]["max_features"]

            ngram_range = tuple(
                params["feature_engineering"][
                    "ngram_range"
                ]
            )

            train_df, test_df = load_data(
                self.input_dir,
                "train_processed.csv",
                "test_processed.csv"
            )


            train_features, test_features = (
                self.apply_tfidf(
                    train_df,
                    test_df,
                    max_features,
                    ngram_range,
                )
            )

            save_data(
                train_features,
                test_features,
                self.output_dir,
                "train_tfidf.csv",
                "test_tfidf.csv"
            )



            logging.info(
                "Feature engineering pipeline completed successfully"
            )

        except Exception as e:
            logging.error(
                "Feature engineering pipeline failed: %s",
                e,
            )
            raise


if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering.run()