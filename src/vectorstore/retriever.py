"""
Small shared helper so src/llm/api_demo.py doesn't have to duplicate the
Chroma connection logic from query_demo.py.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(ROOT, "data", "processed", "chroma_db")
COLLECTION_NAME = "labor_code_articles"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return the top-k {source_file, text} matches for a query."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"source_file": doc_id, "text": text}
        for doc_id, text in zip(results["ids"][0], results["documents"][0])
    ]
