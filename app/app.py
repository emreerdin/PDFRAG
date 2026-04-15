import streamlit as st
from ingestion.pdf_loader import parse_pdf
st.title("RAG PDF App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


if uploaded_file is not None:
    st.success("PDF uploaded successfully")
    parsed_pdf = parse_pdf(uploaded_file)
    st.write(parsed_pdf)