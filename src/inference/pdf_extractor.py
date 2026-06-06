from pypdf import PdfReader



def extract_text_from_pdf(pdf_path: str) -> str:
    #Load the PDF file
    text = ''
    reader = PdfReader(pdf_path)

    #Extract and append text from all pages
    for page in reader.pages:
        text += page.extract_text()

    return text


