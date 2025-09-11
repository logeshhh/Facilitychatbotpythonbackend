import os
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

INDEX_DIR = Path("vectorstore")
DATA_DIR = Path("data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
TOP_K = int(os.getenv("TOP_K", 4))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Load index and metadata
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM model
llm = ChatOllama(model=OLLAMA_MODEL)

# Create FastAPI app
app = FastAPI()

# Define request body
class QueryRequest(BaseModel):
    query: str

# Endpoint to handle queries
@app.post("/query")
def query_bot(request: QueryRequest):
    query_vec = embedder.encode([request.query])
    D, I = index.search(query_vec, TOP_K)
    retrieved = [chunks[i] for i in I[0]]
    context = "\n\n".join([f"[Source: {r['source']}]\n{r['content']}" for r in retrieved])
    prompt = f"""You are a helpful assistant for facility management queries. Use the information below to answer the question. \
If the answer is not available, reply honestly that you don't know.
Context:
{context}
Question:
{request.query}
Answer:"""
    response = llm.invoke(prompt).content
    return {
        "answer": response,
        "sources": [r["source"] for r in retrieved]
    }

# Health check
@app.get("/")
def read_root():
    return {"message": "Facility Helpdesk API is running"}
