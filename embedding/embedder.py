import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

def create_embeddings(chunks):
    client = genai.Client()
    texts = [chunk["text"] for chunk in chunks]

    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=batch
            )
            all_embeddings.extend(result.embeddings)

            if i + batch_size < len(texts):
                time.sleep(2)

        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit, 10 saniye bekleniyor... (batch {i//batch_size + 1})")
                time.sleep(10)
                result = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=batch
                )
                all_embeddings.extend(result.embeddings)
            else:
                raise e

    return all_embeddings


def create_sentence_embedding(sentence):
    client = genai.Client()
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=sentence
    )
    return result