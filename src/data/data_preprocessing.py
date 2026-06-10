import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
from src.utils.utils import save_data, load_data

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class DataPreprocessing:
    """Preprocess resume text data."""

    def __init__(
        self,
        raw_data_dir: str = "./data/raw",
        output_dir: str = "./data/interim",
        text_column: str = "Resume",
    ):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.text_column = text_column

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean_text(
            self,
            text: str
        ) -> str:
        """Clean and preprocess resume text."""

        if pd.isna(text):
            return ""

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove URLs
        text = re.sub(
            r"http\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+",
            " ",
            text,
        )

        # Remove phone numbers
        text = re.sub(
            r"(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
            " ",
            text,
        )

        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove bullets
        text = re.sub(r"[•▪◦●■♦★]", " ", text)
        text = re.sub(r"\*", " ", text)

        # Keep useful characters
        text = re.sub(r"[^a-zA-Z0-9+#/. ]", " ", text)

        # Lowercase
        text = text.lower()

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords
        text = " ".join(
            word
            for word in text.split()
            if word not in self.stop_words
        )

        # Lemmatization
        text = " ".join(
            self.lemmatizer.lemmatize(word)
            for word in text.split()
        )

        return text

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply preprocessing to dataframe."""

        df = df.copy()

        df[self.text_column] = (
            df[self.text_column]
            .fillna("")
            .apply(self.clean_text)
        )

        df = df.dropna(subset=[self.text_column])

        logging.info("Data preprocessing completed")

        return df

    
    def run(self) -> None:
        """Execute preprocessing pipeline."""

        try:
            train_df, test_df = load_data(
                self.raw_data_dir,
                "train.csv",
                "test.csv"
            )

            train_processed = self.preprocess_dataframe(
                train_df
            )

            test_processed = self.preprocess_dataframe(
                test_df
            )

            save_data(
                train_processed,
                test_processed,
                self.output_dir,
                "train_processed.csv",
                "test_processed.csv"
            )

            logging.info(
                "Data preprocessing pipeline completed successfully."
            )

        except Exception as e:
            logging.error(
                "Data preprocessing pipeline failed: %s",
                e,
            )
            raise


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()