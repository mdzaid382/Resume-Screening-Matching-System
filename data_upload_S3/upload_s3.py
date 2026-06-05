import boto3
from botocore.exceptions import NoCredentialsError
from src.logger import logging

def upload_csv_to_s3(local_file_path, bucket_name, s3_file_key):
    """
    Upload a CSV file to S3.

    Parameters:
    local_file_path : str
        Path to local CSV file
    bucket_name : str
        S3 bucket name
    s3_file_key : str
        Path inside S3 bucket
    """

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(
            Filename=local_file_path,
            Bucket=bucket_name,
            Key=s3_file_key
        )
        logging.info(f"File uploaded successfully to s3://{bucket_name}/{s3_file_key}")

    except FileNotFoundError as e:
        logging.info("Local file not found: %s", e)

    except NoCredentialsError:
        logging.info("Aws Credientials not found: %s", e)


if __name__ == "__main__":
    upload_csv_to_s3(
        local_file_path=r"D:\Resume-Screening-Matching-System\data_upload_S3\data\cleaned_dataset.csv",
        bucket_name="resume-screening-and-matching",
        s3_file_key="data/resume_data.csv"
    )