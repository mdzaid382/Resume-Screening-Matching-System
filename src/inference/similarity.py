from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.inference.preprocess_text import clean_text


def calculate_similarity(jd_text, resume_text):

    resume_text = clean_text(resume_text)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [jd_text.lower(), resume_text]
    )

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]
    score = min(100, 50 + score * 100)
    return round(float(score), 2)