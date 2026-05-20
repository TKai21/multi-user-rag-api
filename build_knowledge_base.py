import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

FILE_PATH = "profile.txt"


# 從檔案讀取並切 chunks
def load_chunks(path: str) -> list[str]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)

chunks = load_chunks(FILE_PATH)

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": FILE_PATH} for _ in chunks],
)

print(f"Loaded: {len(chunks)} chunks from {FILE_PATH}")
print(f"Total stored: {collection.count()}")
