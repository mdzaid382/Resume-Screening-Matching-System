from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(jd_text, resume_text):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    vectors = tfidf.fit_transform(
        [jd_text, resume_text]
    )

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(score * 100, 2)