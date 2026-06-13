import unittest
import os
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.inference.predictor import predict_role
from src.inference.similarity import calculate_similarity
from src.inference.pdf_extractor import extract_text_from_pdf
from src.logger import logging


class TestResumeScreeningSystem(unittest.TestCase):
    
    
    @classmethod
    def setUpClass(cls):
    
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
    
        cls.model_name = "resume-screening"
        client = mlflow.MlflowClient()
        
        cls.new_model = mlflow.pyfunc.load_model(
            f"models:/{cls.model_name}@challenger"
        )
    
        
    
        try:
            champion_version = client.get_model_version_by_alias(
                name=cls.model_name,
                alias="champion"
            )
    
            champion_run = client.get_run(
                champion_version.run_id
            )
         
            cls.champ_accuracy = champion_run.data.metrics.get("accuracy")
            cls.champ_precision = champion_run.data.metrics.get("precision")
            cls.champ_recall = champion_run.data.metrics.get("recall")
            cls.champ_f1 = champion_run.data.metrics.get("f1_score")

            cls.has_champion = True
    
        except Exception:
            cls.has_champion = False
            
    
        cls.sample_resume = """
        Data Scientist with experience in Python,
        Machine Learning, NLP, FastAPI, Docker
        and Scikit-learn.
        """
    
        cls.sample_jd = """
        Looking for a Data Scientist with Python,
        Machine Learning and NLP experience.
        """
            

    def test_predict_role_returns_string(self):
    
        prediction = predict_role(
            self.sample_resume,
            self.new_model
        )
    
        self.assertIsInstance(
            prediction,
            str
        )
    
        self.assertGreater(
            len(prediction),
            0
        )
    
    def test_similarity_score_range(self):
    
        score = calculate_similarity(
            self.sample_jd,
            self.sample_resume
        )
    
        self.assertGreaterEqual(
            score,
            0.0
        )
    
        self.assertLessEqual(
            score,
            100.0
        )
    
    def test_similarity_type(self):
    
        score = calculate_similarity(
            self.sample_jd,
            self.sample_resume
        )
    
        self.assertIsInstance(
            score,
            float
        )
    
    def test_resume_matches_better_than_random_text(self):
    
        good_score = calculate_similarity(
            self.sample_jd,
            self.sample_resume
        )
    
        bad_score = calculate_similarity(
            self.sample_jd,
            "I am a chef with cooking experience."
        )
    
        self.assertGreater(
            good_score,
            bad_score
        )
    
    def test_pdf_extraction(self):
    
        pdf_path = "tests/sample_resume.pdf"
    
        if os.path.exists(pdf_path):
    
            text = extract_text_from_pdf(
                pdf_path
            )
    
            self.assertIsInstance(
                text,
                str
            )
    
            self.assertGreater(
                len(text),
                0
            )
    
    def test_prediction_not_empty(self):
    
        prediction = predict_role(
            self.sample_resume,
            self.new_model
        )
    
        self.assertNotEqual(
            prediction.strip(),
            ""
        )
    
    def test_model_performance(self):

        if not self.has_champion:
            self.skipTest(
                "No champion model exists yet"
            )    
        self.holdout_data = pd.read_csv('data/interim/test_processed.csv')
        logging.info('test data loaded.')

        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(
            y_holdout,
            y_pred_new,     
        )
        precision_new = precision_score(
            y_holdout,
            y_pred_new,
            average='weighted'
        )
        recall_new = recall_score(
            y_holdout,
            y_pred_new,
            average='weighted'
        )
        f1_new = f1_score(
            y_holdout,
            y_pred_new,
            average='weighted'
        )

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, self.champ_accuracy, f'Accuracy should be at least {self.champ_accuracy}')
        self.assertGreaterEqual(precision_new, self.champ_precision, f'Precision should be at least {self.champ_precision}')
        self.assertGreaterEqual(recall_new, self.champ_recall, f'Recall should be at least {self.champ_recall}')
        self.assertGreaterEqual(f1_new, self.champ_f1, f'F1 score should be at least {self.champ_f1}')

    
if __name__ == "__main__":
    unittest.main()
    
