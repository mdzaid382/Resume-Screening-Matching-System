from sklearn.metrics.pairwise import cosine_similarity
from src.inference.preprocess_text import clean_text



def calculate_similarity(model,jd_text, resume_text):
    


    resume_text = clean_text(resume_text)
    
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text.lower())

    score = cosine_similarity(
    [resume_embedding],
    [jd_embedding]
    )[0][0]


    return round(float(score) * 100, 2)