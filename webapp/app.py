from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import mlflow
import os
import re
import pandas as pd
from fastapi.responses import FileResponse
from src.inference.pdf_extractor import extract_text_from_pdf
from src.inference.predictor import predict_role
from src.inference.similarity import calculate_similarity

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    app.state.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.role_model = mlflow.pyfunc.load_model(
    f"models:/resume_screening_and_matching@challenger"
)

    yield


app = FastAPI(lifespan=lifespan)


app.mount(
    "/static",
    StaticFiles(directory="webapp/static"),
    name="static"
)

templates = Jinja2Templates(directory="webapp/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="resume_screening.html",
        context={"results": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def screen_resumes(
    request: Request,
    job_description: str = Form(...),
    min_score: int = Form(...),
    resumes: list[UploadFile] = File(...)
):

    

    results = []

    for resume in resumes:

        pdfbytes = await resume.read()
        resume_text = extract_text_from_pdf(pdfbytes)

        match_email = re.search(
                            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                            resume_text
                            )
        email = match_email.group(0) if match_email else None
        
        role = predict_role(resume_text,
                             request.app.state.role_model
                            )

        score = calculate_similarity(
            request.app.state.emb_model,
            job_description,
            resume_text
            )

        

        status = (
            "Shortlisted"
            if score >= min_score
            else "Rejected"
        )

        results.append(
            {
                "resume": resume.filename,
                "email" : email,
                "role": role,
                "score": score,
                "status": status
            }
        )

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    for i, item in enumerate(results, start=1):
        item["rank"] = i

    df = pd.DataFrame(results)
    df.to_csv(
        "webapp/data/results.csv",
        index=False
    )

    return templates.TemplateResponse(
        request=request,
        name="resume_screening.html",
        context={
            "results": results,
            "min_score": min_score
            }
    )

@app.get("/download-csv")
async def download_csv():

    return FileResponse(
        "webapp/data/results.csv",
        media_type="text/csv",
        filename="resume_screening_results.csv"
    )