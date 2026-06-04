# data preprocessing

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
from sklearn.preprocessing import LabelEncoder
nltk.download('wordnet')
nltk.download('stopwords')
import pickle





def clean_text(text: str) -> str:

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs (linkedin, github, websites)
    text = re.sub(r'http\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+', ' ', text)

    # Remove phone numbers
    text = re.sub(
        r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        ' ',
        text
    )

    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bullet symbols
    text = re.sub(r'[•▪◦●■♦★]', ' ', text)

    # Remove asterisks used as bullets
    text = re.sub(r'\*', ' ', text)

    # Keep only letters, numbers, +, #, /, . and spaces
    text = re.sub(r'[^a-zA-Z0-9+#/. ]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    #remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    #lematize the text
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
   

    return text


def preprocess_dataframe(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Apply preprocessing to the specified column
    df[col_name] = df[col_name].apply(clean_text)

    # Drop rows with NaN values
    df = df.dropna(subset=[col_name])
    logging.info("Data pre-processing completed")
    return df


def labelencoder(df: pd.DataFrame, col_name: str) -> pd.DataFrame:

    le = LabelEncoder()
    df[col_name] = le.fit_transform(df[col_name])
    pickle.dump(le, open('models/le.pkl', 'wb'))

    return df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'Resume')
        test_processed_data = preprocess_dataframe(test_data, 'Resume')
        logging.info('data processed completed.')

        #labels the target column
        train_labeled_data = labelencoder(train_processed_data, 'Role')
        test_labeled_data = labelencoder(test_processed_data, 'Role')
        logging.info('target column labeled.')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_labeled_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_labeled_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()