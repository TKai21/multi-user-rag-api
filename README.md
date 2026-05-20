# multi-user-rag-api

A local multi-user RAG (Retrieval-Augmented Generation) API that lets multiple users upload their own profiles and query them using natural language.

## How It Works

```
POST /documents?user_id=kai   →  Upload your profile into ChromaDB
GET  /ask/kai?question=...    →  Query your profile using a local LLM
```

Each user gets their own isolated vector collection. Queries only retrieve context from the specified user's profile.

## Tech Stack

- **FastAPI** — REST API framework
- **ChromaDB** — Local vector database for storing and retrieving profile chunks
- **Ollama** — Local LLM runner
- **nomic-embed-text** — Embedding model for converting text to vectors
- **qwen2.5:0.5b** — Local LLM for generating answers

## Getting Started

### 1. Install dependencies

```bash
pip install fastapi uvicorn chromadb ollama
```

### 2. Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Start the API server

```bash
uvicorn main:app --reload
```

## Usage

### Upload a profile

```bash
curl -X POST "http://localhost:8000/documents?user_id=kai" \
     -H "Content-Type: application/json" \
     -d '{"text": "My name is Kai. I am a CS student at USC. My goal is to become a ML Engineer."}'
```

### Ask a question

```bash
curl -G "http://localhost:8000/ask/kai" \
     --data-urlencode "question=Who are you"
```

### Interactive API docs

```
http://localhost:8000/docs
```

## Project Structure

```
multi-user-rag-api/
├── main.py                  # FastAPI app with /documents and /ask endpoints
├── build_knowledge_base.py  # Script to build initial knowledge base from file
├── profile.txt              # Sample profile data
└── chroma_db/               # Local vector database (auto-generated, not tracked)
```