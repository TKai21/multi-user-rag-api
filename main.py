from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb



from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
app = FastAPI()

# 初始化 ChromaDB（跟 build_knowledge_base.py 一樣）
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)

class ProfileInput(BaseModel):
    text: str
    
    

@app.get("/ask/{user_id}") # this create a GET endpoint at /ask
def ask(user_id: str, question: str): # FastAPI reads "question" from the URL query string
    
    user_collection = client.get_or_create_collection(
        name = f"profile_{user_id}",
        embedding_function = ef,
    )
    
    results = user_collection.query(query_texts=[question], n_results=2)
    
    # RETRIEVE - find the 2 most relevant chunks from your knowledge base
    # results = collection.query(query_texts=[question], n_results=2)
    # combine the matching chunks into a singe string
    context = "\n\n".join(results["documents"][0])
    
    # AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""Use the following context to answer the question.

If the context doesn't contain relevant information, say so.

Context:

{context}

Question: {question}"""

    # GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
        model = "qwen2.5:0.5b",
        messages = [{"role": "user", "content": augmented_prompt}]
    )
        
    # return the answer along with the context so users can verify the source
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
    
@app.post("/documents") # post method

def upload_profile(user_id: str, profile: ProfileInput):
    # 把文字切成 chunks 
    
    chunks = [s.strip() for s in profile.text.split(".") if s.strip()]
    
    user_collection = client.get_or_create_collection(
        name = f"profile_{user_id}",
        embedding_function = ef,
    )
    
    # chunk
    user_collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"user_id": user_id} for _ in chunks],
    )

    return {"message": f"Profile uploaded for {user_id}", "chunks": len(chunks)}