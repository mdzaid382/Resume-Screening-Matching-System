# data ingestion
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.connections import s3_connection
from dotenv import load_dotenv

load_dotenv()


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logging.info('Data loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        
        # df = load_data(data_path=r'D:\Resume-Screening-Matching-System\data_upload_S3\data\cleaned_dataset.csv')
        s3 = s3_connection.s3_operations(bucket_name=os.getenv("S3_BUCKET_NAME"),
                                         aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                                         aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                         region_name='us-east-1'
                                         )
        df = s3.fetch_file_from_s3("data/resume_data.csv")



        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()




