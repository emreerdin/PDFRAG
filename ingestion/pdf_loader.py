from pypdf import PdfReader

def parse_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    return text