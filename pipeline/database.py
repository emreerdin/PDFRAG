import psycopg2
import json
import os
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from embedding.embedder import create_sentence_embedding

load_dotenv()

def create_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn

def delete_session(conn, session_id):
    cur = conn.cursor()
    cur.execute("DELETE FROM document WHERE session_id = %s;", (session_id,))
    conn.commit()
    cur.close()

def insert_embedding(session_id, chunks, embeddings):
    conn = create_connection()
    delete_session(conn, session_id)
    cur = conn.cursor()
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO document (session_id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (
                session_id,
                chunk["text"],
                emb.values,
                json.dumps(chunk["metadata"])
            )
        )
    conn.commit()
    cur.close()
    conn.close()

def search_a_sentence_similarity(session_id, sentence, limit):
    conn = create_connection()
    cur = conn.cursor()
    sentence_embedded = create_sentence_embedding(sentence).embeddings[0].values
    cur.execute("""
        SELECT content, metadata
        FROM document
        WHERE session_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (session_id, sentence_embedded, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results