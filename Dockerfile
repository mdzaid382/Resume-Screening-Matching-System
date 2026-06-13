FROM python:3.12.6-slim

WORKDIR /app

COPY requirements/inference.txt .

RUN pip install --no-cache-dir -r inference.txt

COPY src/inference ./src/inference
COPY webapp ./webapp

EXPOSE 8000

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]