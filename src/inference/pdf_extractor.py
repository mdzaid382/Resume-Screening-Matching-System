from io import BytesIO
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_bytes):
    pdf_file = BytesIO(pdf_bytes)

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


