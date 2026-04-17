from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import uuid
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from pipeline.database import insert_embedding, search_a_sentence_similarity
from embedding.embedder import create_embeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API çalışıyor"}


@app.get("/upload")
def upload_info():
    return {"detail": "Use POST /upload with a multipart PDF file."}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyası kabul edilir.")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Dosya okunurken hata oluştu.")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")

    try:
        pdf_file = io.BytesIO(contents)
        pdf_file.name = filename
        pages = parse_pdf(pdf_file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF parse hatası: {str(e)}")

    if not pages:
        raise HTTPException(status_code=422, detail="PDF'den sayfa okunamadı.")

    try:
        all_chunks = []
        for page in pages:
            chunks = text_to_chunks(
                text=page["text"],
                source=filename,
                page=page["page"]
            )
            all_chunks.extend(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metin bölme hatası: {str(e)}")

    if not all_chunks:
        raise HTTPException(status_code=422, detail="PDF boş veya metin içermiyor.")

    try:
        session_id = str(uuid.uuid4())
        embeddings = create_embeddings(all_chunks)
        insert_embedding(session_id, all_chunks, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding hatası: {str(e)}")

    return {
        "message": "Yükleme başarılı",
        "session_id": session_id,
        "chunks": len(all_chunks),
        "pages": len(pages),
        "filename": filename,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    try:
        records = search_a_sentence_similarity(body.session_id, body.question, 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arama hatası: {str(e)}")

    if not records:
        raise HTTPException(status_code=404, detail="İlgili içerik bulunamadı.")

    try:
        from pipeline.LLM import get_answer
        context_text = "\n\n".join([r[0] for r in records])
        answer = get_answer(body.question, context_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM hatası: {str(e)}")

    return AskResponse(
        question=body.question,
        answer=answer,
        context_used=[r[0] for r in records],
    )