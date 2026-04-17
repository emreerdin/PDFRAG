from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io

from fastapi.middleware.cors import CORSMiddleware
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from pipeline.database import insert_embedding, search_a_sentence_similarity
from embedding.embedder import create_embeddings
from pipeline.LLM import get_answer, check_tokens

app = FastAPI(
    title="RAG PDF API",
    description="PDF yükle, soru sor, cevap al.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm adreslere izin ver (Geliştirme aşaması için)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response modelleri ─────────────────────────────

class AskRequest(BaseModel):
    question: str
    filename: Optional[str] = None  # hangi PDF'e soru sorulduğunu filtrelemek için


class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]


class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int
    total_tokens: int


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Servisin ayakta olup olmadığını kontrol eder."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF dosyası yükler, chunk'lara böler, embedding oluşturur ve veritabanına kaydeder.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyası yüklenebilir.")

    # Streamlit'te uploaded_file doğrudan geçiliyordu;
    # burada bytes okuyup aynı parse_pdf fonksiyonuna veriyoruz.
    contents = await file.read()
    pdf_file = io.BytesIO(contents)
    pdf_file.name = file.filename  # parse_pdf name'e ihtiyaç duyuyorsa

    try:
        pages = parse_pdf(pdf_file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF okunamadı: {str(e)}")

    all_chunks = []
    for page in pages:
        chunks = text_to_chunks(
            text=page["text"],
            source=file.filename,
            page=page["page"]
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=422, detail="PDF'den metin çıkarılamadı.")

    embeddings = create_embeddings(all_chunks)
    insert_embedding(all_chunks, embeddings)
    tokens = check_tokens(all_chunks)

    return UploadResponse(
        message="PDF başarıyla işlendi.",
        filename=file.filename,
        total_chunks=len(all_chunks),
        total_tokens=tokens,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """
    Kullanıcının sorusunu alır, vektör benzerliği ile ilgili chunk'ları bulur,
    LLM'e gönderir ve cevabı döner.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # Token sayısına göre kaç chunk getireceğimizi belirle
    # (Streamlit kodundaki mantığın aynısı — burada chunk sayısı
    #  upload sırasında zaten hesaplandığından sabit limit kullanıyoruz.
    #  İsterseniz /upload cevabındaki total_tokens'ı client'tan alıp
    #  buraya da gönderebilirsiniz.)
    limit = 3  # varsayılan; gerekirse AskRequest'e token_count eklenebilir

    records = search_a_sentence_similarity(question, limit)
    if not records:
        raise HTTPException(status_code=404, detail="İlgili içerik bulunamadı.")

    context_text = "\n\n".join([r[0] for r in records])
    answer = get_answer(question, context_text)

    return AskResponse(
        question=question,
        answer=answer,
        context_used=[r[0] for r in records],
    )


@app.post("/upload-and-ask", response_model=AskResponse)
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = "",
):
    """
    PDF yükle ve aynı anda soru sor — Streamlit akışının birebir karşılığı.
    Form-data ile hem dosya hem soru gönderilebilir.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyası yüklenebilir.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    contents = await file.read()
    pdf_file = io.BytesIO(contents)
    pdf_file.name = file.filename

    try:
        pages = parse_pdf(pdf_file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF okunamadı: {str(e)}")

    all_chunks = []
    for page in pages:
        chunks = text_to_chunks(
            text=page["text"],
            source=file.filename,
            page=page["page"]
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=422, detail="PDF'den metin çıkarılamadı.")

    embeddings = create_embeddings(all_chunks)
    insert_embedding(all_chunks, embeddings)
    tokens = check_tokens(all_chunks)

    limit = 1 if tokens <= 3000 else 5
    records = search_a_sentence_similarity(question, limit)
    if not records:
        raise HTTPException(status_code=404, detail="İlgili içerik bulunamadı.")

    context_text = "\n\n".join([r[0] for r in records])
    answer = get_answer(question, context_text)

    return AskResponse(
        question=question,
        answer=answer,
        context_used=[r[0] for r in records],
    )