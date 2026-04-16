from pypdf import PdfReader

def parse_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "text": text,
            "page": i
        })
    
    return pages