from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from fastapi.staticfiles import StaticFiles

from src.inference.pdf_extractor import extract_text_from_pdf
from src.inference.predictor import predict_role
from src.inference.similarity import calculate_similarity

app = FastAPI()



app.mount(
    "/static",
    StaticFiles(directory="webapp/static"),
    name="static"
)

templates = Jinja2Templates(directory="webapp/templates")


@app.get("/predict", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="resume_screening.html",
        context={"results": None}
    )


@app.post("/", response_class=HTMLResponse)
async def screen_resumes(
    request: Request,
    job_description: str = Form(...),
    min_score: int = Form(...),
    resumes: list[UploadFile] = File(...)
):

    

    jd_text = job_description

    results = []

    for resume in resumes:

        pdfbytes = await resume.read()
        resume_text = extract_text_from_pdf(pdfbytes)

        role = predict_role(resume_text)

        score = calculate_similarity(
            jd_text,
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

    return templates.TemplateResponse(
        request=request,
        name="resume_screening.html",
        context={
            "results": results,
            "min_score": min_score
            }
    )