import streamlit as st
from ingestion.pdf_loader import parse_pdf
from ingestion.text_splitter import text_to_chunks
from embedding.embedder import create_embeddings
st.title("RAG PDF App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


if uploaded_file is not None:
    st.success("PDF uploaded successfully")
    parsed_pdf = parse_pdf(uploaded_file)
    st.write(parsed_pdf)
    chunked_text = text_to_chunks(parsed_pdf)
    st.write(chunked_text)
    embedded_text = create_embeddings(chunked_text)
    st.write(embedded_text)