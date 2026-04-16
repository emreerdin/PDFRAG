import psycopg2
import json
from pgvector.psycopg2 import register_vector
from embedding.embedder import create_sentence_embedding
def create_connection():
    conn = psycopg2.connect(
        dbname="mydb",
        user="admin",
        password = "secret",
        host="localhost",
        port="5433"
    )

    return conn

def insert_embedding(chunks, embeddings):

    conn = create_connection()

    register_vector(conn)
    cur = conn.cursor()

    for chunk, emb in zip(chunks, embeddings.embeddings):
        cur.execute(
            """
            INSERT INTO document (content, embedding, metadata)
            VALUES (%s, %s, %s)
            """,
            (
                chunk["text"],
                emb.values,          # vector
                json.dumps(chunk["metadata"])    # json
            )
        )

    conn.commit()
    cur.close()
    conn.close()

def search_a_sentence_similarity(sentence):
    conn = create_connection()

    register_vector(conn)
    cur = conn.cursor()

    # First sentence embedding
    sentence_embedded = create_sentence_embedding(sentence).embeddings[0].values

    # Then search

    cur.execute("""
        SELECT content, metadata
        FROM document
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """, (sentence_embedded,))
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

    