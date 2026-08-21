"""
Checkpoint 2 - Vector Database Setup
Philippine Labor Law Q&A Bot

Loads the cleaned corpus (data/processed/_corpus.jsonl) into a persistent
Chroma collection, using the same sentence-transformers model as Checkpoint 1
(all-MiniLM-L6-v2) so retrieval-time query embeddings and stored document
embeddings come from the same vector space. Chroma is configured for cosine
similarity - see docs/distance_metric_report.md for why.

Usage:
    python src/vectorstore/build_index.py
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_PATH = os.path.join(ROOT, "data", "processed", "_corpus.jsonl")
CHROMA_DIR = os.path.join(ROOT, "data", "processed", "chroma_db")
COLLECTION_NAME = "labor_code_articles"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    docs = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_corpus()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    # Rebuild from scratch each run so the index always matches data/processed/.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["source_file"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source_file": d["source_file"]} for d in docs],
    )

    print(f"Indexed {collection.count()} documents into Chroma collection '{COLLECTION_NAME}'")
    print(f"Persisted to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
