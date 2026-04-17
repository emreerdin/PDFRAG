import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

def create_embeddings(chunks):
    client = genai.Client()
    texts = [chunk["text"] for chunk in chunks]

    all_embeddings = []
    batch_size = 20  # daha küçük batch

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0

        while retries < 5:
            try:
                result = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=batch
                )
                all_embeddings.extend(result.embeddings)
                time.sleep(3)  # her batch sonrası bekle
                break

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = (retries + 1) * 15  # 15, 30, 45, 60, 75 sn
                    print(f"Rate limit, {wait}s bekleniyor... (batch {i//batch_size + 1}, deneme {retries + 1})")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise e

        if retries == 5:
            raise Exception("5 denemede de rate limit aşılamadı")

    return all_embeddings


def create_sentence_embedding(sentence):
    client = genai.Client()
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=sentence
    )
    return result