from dotenv import load_dotenv
import os
import mlflow

load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "mdzaid382"
repo_name = "Resume-Screening-Matching-System"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

MODEL_NAME = 'my_model'
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/latest"
)


def predict_role(resume_text):

    prediction = model.predict(
        [resume_text]
    )

    return prediction[0]


