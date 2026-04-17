import psycopg2
import json
from pgvector.psycopg2 import register_vector
from embedding.embedder import create_sentence_embedding

def create_connection():
    conn = psycopg2.connect(
        dbname="mydb",
        user="admin",
        password="secret",
        host="localhost",
        port="5433"
    )
    register_vector(conn)
    return conn


def truncate_table(conn):
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE document RESTART IDENTITY;")
    conn.commit()
    cur.close()


def insert_embedding(chunks, embeddings):
    conn = create_connection()
    cur = conn.cursor()

    truncate_table(conn)

    for chunk, emb in zip(chunks, embeddings.embeddings):
        cur.execute(
            """
            INSERT INTO document (content, embedding, metadata)
            VALUES (%s, %s, %s)
            """,
            (
                chunk["text"],
                emb.values,
                json.dumps(chunk["metadata"])
            )
        )

    conn.commit()
    cur.close()
    conn.close()


def search_a_sentence_similarity(sentence, limit):
    conn = create_connection()
    cur = conn.cursor()

    sentence_embedded = create_sentence_embedding(sentence).embeddings[0].values

    cur.execute("""
        SELECT content, metadata
        FROM document
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (sentence_embedded, limit))

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results