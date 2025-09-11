import os
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from dotenv import load_dotenv

# Load env variables
load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("vectorstore")
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Load documents from data folder
def load_documents():
    docs = []
    for file in DATA_DIR.glob("**/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({"content": f.read(), "source": str(file)})
    return docs

# Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for s in splits:
            all_chunks.append({"content": s, "source": doc["source"]})
    return all_chunks

# Create embeddings and save index
def build_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index built and saved.")

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    build_faiss_index(chunks)
