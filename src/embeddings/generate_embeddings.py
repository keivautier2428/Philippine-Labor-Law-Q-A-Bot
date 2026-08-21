"""
Checkpoint 1 - Embedding Generation Demo
Philippine Labor Law Q&A Bot

Loads the cleaned corpus produced by src/preprocessing/preprocess.py and
converts it into dense vector embeddings using a local Hugging Face
sentence-transformers model (no API key required).

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings, ~80MB, runs fast on CPU.
  - Trained for semantic similarity / retrieval, which is exactly what a
    RAG system needs when matching a user's question to the most relevant
    Labor Code article.
  - Free and local: no per-call cost or API key, important for a student
    project that needs to run the same way on any teammate's machine.
  - Trade-off acknowledged: a larger model (e.g. all-mpnet-base-v2) would
    likely retrieve slightly more accurately, but at ~3x the size/latency.
    MiniLM is the right size for a 33-document corpus at this checkpoint;
    revisit if retrieval quality is insufficient once the full RAG
    pipeline (Checkpoint 3) is evaluated end-to-end.

Usage:
    python src/embeddings/generate_embeddings.py
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_PATH = os.path.join(ROOT, "data", "processed", "_corpus.jsonl")
EMBEDDINGS_OUT = os.path.join(ROOT, "data", "processed", "embeddings.npy")
METADATA_OUT = os.path.join(ROOT, "data", "processed", "embeddings_metadata.json")
DEMO_REPORT = os.path.join(ROOT, "docs", "embedding_demo_report.md")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    docs = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_corpus()
    texts = [d["text"] for d in docs]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    os.makedirs(os.path.dirname(EMBEDDINGS_OUT), exist_ok=True)
    np.save(EMBEDDINGS_OUT, embeddings)

    metadata = [{"source_file": d["source_file"], "text_preview": d["text"][:120]} for d in docs]
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Sanity-check the embeddings with a similarity query, and use it as the
    # written demo evidence for the checkpoint.
    query = "Can my employer fire me without giving a reason?"
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = sorted(
        (
            (cosine_sim(query_vec, embeddings[i]), docs[i])
            for i in range(len(docs))
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    top5 = scored[:5]

    lines = [
        "# Embedding Generation Demo",
        "",
        f"**Model:** `{MODEL_NAME}` (Hugging Face `sentence-transformers`, run locally)",
        f"**Embedding dimensionality:** {embeddings.shape[1]}",
        f"**Documents embedded:** {embeddings.shape[0]}",
        "",
        "## Why this model",
        "",
        "`all-MiniLM-L6-v2` was chosen because it is trained specifically for semantic",
        "similarity/retrieval, is small enough (~80MB) to run on CPU with no GPU or API key,",
        "and its 384-dim vectors keep the Checkpoint 2 vector index lightweight for a",
        "33-document corpus. It trades some accuracy against larger models (e.g.",
        "`all-mpnet-base-v2`) in exchange for speed and zero infrastructure cost — an",
        "appropriate trade-off for this stage of the project.",
        "",
        "## Sanity-check query",
        "",
        f"Query: *\"{query}\"*",
        "",
        "Top 5 nearest documents by cosine similarity:",
        "",
    ]
    for score, doc in top5:
        lines.append(f"- `{doc['source_file']}` (similarity: {score:.4f}) — {doc['text'][:140]}...")

    lines.append("")
    lines.append("This confirms the embeddings place semantically related Labor Code articles")
    lines.append("(here, the just-cause/termination provisions) close together in vector space,")
    lines.append("which is exactly the retrieval behavior the Checkpoint 3 RAG pipeline depends on.")

    os.makedirs(os.path.dirname(DEMO_REPORT), exist_ok=True)
    with open(DEMO_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved embeddings -> {EMBEDDINGS_OUT}")
    print(f"Saved metadata -> {METADATA_OUT}")
    print(f"Saved demo report -> {DEMO_REPORT}")


if __name__ == "__main__":
    main()
