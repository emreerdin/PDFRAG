TOKEN_TEMPLATE = """

Return the token number of given text

"""


RAG_TEMPLATE = """
You are a QA assistant.

Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY using the provided context.
- If the answer is not in the context, say: "Not in context".
- Do NOT make assumptions or add external knowledge.
- Keep the answer concise.

Answer format:
Answer:
<final answer>

Sources:
- <source 1>
- <source 2>

"""