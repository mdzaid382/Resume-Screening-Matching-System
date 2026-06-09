from dotenv import load_dotenv
from src.inference.preprocess_text import clean_text

load_dotenv()


def predict_role(resume_text: str, model) -> str :

    resume_text = clean_text(resume_text)
    prediction = model.predict(
        [resume_text]
    )

    return prediction[0]


