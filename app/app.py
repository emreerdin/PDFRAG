import streamlit as st
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from pipeline.database import insert_embedding, search_a_sentence_similarity
from embedding.embedder import create_embeddings
from pipeline.LLM import get_answer, check_tokens

st.title("RAG PDF App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_input = st.text_input("Please enter your sentence")

if uploaded_file is not None:
    pages = parse_pdf(uploaded_file)

    all_chunks = []

    for page in pages:
        chunks = text_to_chunks(
            text=page["text"],
            source=uploaded_file.name,
            page=page["page"]
        )
        all_chunks.extend(chunks)

    embeddings = create_embeddings(all_chunks)
    insert_embedding(all_chunks, embeddings)

    tokens = check_tokens(all_chunks)

    if user_input and user_input.strip():
        limit = 1 if tokens <= 3000 else 5
        
        records = search_a_sentence_similarity(user_input, limit)

        if records:
            context_text = "\n\n".join([r[0] for r in records])
            answer = get_answer(user_input, context_text)

            st.subheader("Cevap:")
            st.write(answer)