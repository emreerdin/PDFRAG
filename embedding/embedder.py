import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def create_embeddings(chunks):
    client = genai.Client()

    texts = [chunk["text"] for chunk in chunks]  # sadece text al

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    return result

def create_sentence_embedding(sentence):
    client = genai.Client()
    
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence
    )

    return result
