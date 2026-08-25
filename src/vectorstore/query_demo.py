"""
Checkpoint 2 - Vector Database Setup (similarity-search demo)
Philippine Labor Law Q&A Bot

Runs a handful of realistic queries against the Chroma index built by
build_index.py and prints/reports the top matches, as evidence the vector
database is actually working end-to-end (not just populated).

Usage:
    python src/vectorstore/build_index.py   # run first, if not already run
    python src/vectorstore/query_demo.py
"""

import os

import chromadb
from chromadb.utils import embedding_functions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(ROOT, "data", "processed", "chroma_db")
COLLECTION_NAME = "labor_code_articles"
REPORT_PATH = os.path.join(ROOT, "docs", "vector_db_demo_report.md")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_QUERIES = [
    "Can my employer fire me without giving a reason?",
    "How many hours can I be made to work in a day?",
    "Do I get paid extra for working on my rest day?",
    "What happens to my job if the company temporarily shuts down?",
    "Can a 16 year old legally be hired for a job?",
]


RULE = "=" * 70


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    print(RULE)
    print(f"VECTOR DB SIMILARITY SEARCH - {collection.count()} documents, cosine similarity")
    print(RULE)

    lines = [
        "# Vector Database Similarity-Search Demo",
        "",
        f"**Store:** Chroma (persistent, cosine similarity), `{collection.count()}` documents indexed.",
        f"**Embedding model:** `{MODEL_NAME}` (same model used in Checkpoint 1).",
        "",
    ]

    for query in SAMPLE_QUERIES:
        results = collection.query(query_texts=[query], n_results=3)
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        print(f'\nQuery: "{query}"')
        lines.append(f"## Query: \"{query}\"")
        lines.append("")
        for rank, (doc_id, dist, doc_text) in enumerate(zip(ids, distances, documents), start=1):
            similarity = 1 - dist  # Chroma cosine "distance" = 1 - cosine similarity
            preview = doc_text[:160].replace("\n", " ")
            print(f"  {rank}. {doc_id}  (cosine similarity: {similarity:.4f})")
            print(f"     {preview}...")
            lines.append(f"{rank}. `{doc_id}` (cosine similarity: {similarity:.4f}) — {preview}...")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n{RULE}")
    print(f"Ran {len(SAMPLE_QUERIES)} sample queries against the Chroma index.")
    print(f"Report -> {REPORT_PATH}")
    print(RULE)


if __name__ == "__main__":
    main()
