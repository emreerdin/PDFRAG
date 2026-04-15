def fixed_size_chunking(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += len(word.split())
        if current_tokens <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = len(word.split())
    
    # Append the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

text = "Your long text here..."
max_tokens = 100
chunks = fixed_size_chunking(text, max_tokens)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")