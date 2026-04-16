import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def create_embeddings(chunks):
    client = genai.Client()

    result = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = chunks
    )

    return result