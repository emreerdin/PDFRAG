from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators =["\n\n", "\n", ".", " ", ""])
    texts = text_splitter.split_text(text)

    return texts


'''def text_to_chunks(text):
    if not text:
        return []

    words = text.split()

    chunk_size = 300
    overlap = 50

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        if end >= len(words):
            break

        start = end - overlap

    return chunks'''
