import unittest

from fastapi.testclient import TestClient
from webapp.app import app

client = TestClient(app)

class TestAPI(unittest.TestCase):


    def test_home_route(self):

        response = client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

    def test_docs_route(self):

        response = client.get("/docs")

        self.assertEqual(
            response.status_code,
            200
        )

    def test_predict_page(self):

        response = client.post(
            "/predict",
            data={
                "job_description": "Python ML NLP",
                "min_score": "50"
            },
            files={
                "resumes": (
                "sample.pdf",
                open("tests/sample_resume.pdf", "rb"),
                "application/pdf"
                )
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertTrue(
            "Shortlisted" in response.text or "Rejected" in response.text
        )

if __name__ == "__main__":
    unittest.main()
    
