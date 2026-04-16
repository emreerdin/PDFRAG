import streamlit as st
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from pipeline.database import insert_embedding, search_a_sentence_similarity
from embedding.embedder import create_embeddings

st.title("RAG PDF App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully")

    pages = parse_pdf(uploaded_file)

    all_chunks = []

    for page in pages:
        chunks = text_to_chunks(
            text=page["text"],
            source=uploaded_file.name,
            page=page["page"]
        )
        all_chunks.extend(chunks)

    st.write(all_chunks)

    embeddings = create_embeddings(all_chunks)

    st.write(embeddings)
    st.write("Database writing has started")
    insert_embedding(all_chunks, embeddings)
    st.success("Insertion completed")
    user_input = st.text_input("Please enter your sentence")

    if user_input and user_input.strip():
        records = search_a_sentence_similarity(user_input)
        st.write(records)