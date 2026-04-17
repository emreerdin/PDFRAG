from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from pipeline.database import insert_embedding, search_a_sentence_similarity
from embedding.embedder import create_embeddings
from pipeline.LLM import get_answer, check_tokens

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF kabul edilir")

    contents = await file.read()
    pdf_file = io.BytesIO(contents)
    pdf_file.name = file.filename

    pages = parse_pdf(pdf_file)

    all_chunks = []
    for page in pages:
        chunks = text_to_chunks(
            text=page["text"],
            source=file.filename,
            page=page["page"]
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=422, detail="PDF boş veya okunamadı")

    embeddings = create_embeddings(all_chunks)
    insert_embedding(all_chunks, embeddings)

    return {"message": "Yükleme başarılı", "chunks": len(all_chunks)}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz")

    records = search_a_sentence_similarity(body.question, 5)

    if not records:
        raise HTTPException(status_code=404, detail="İlgili içerik bulunamadı")

    context_text = "\n\n".join([r[0] for r in records])
    answer = get_answer(body.question, context_text)

    return AskResponse(
        question=body.question,
        answer=answer,
        context_used=[r[0] for r in records],
    )