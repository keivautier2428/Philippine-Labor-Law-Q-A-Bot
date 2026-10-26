"""
Checkpoint 4 - backend wiring for the web interface.
Kept separate from the Streamlit script so tests can swap it for fakes.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_backend():
    """Return (vectorstore, llm). If the index is empty (fresh volume, first
    deploy), build it from data/raw + data/incoming instead of failing."""
    from src.ingestion.pipeline import get_vectorstore, ingest
    from src.rag.llm import make_llm

    vs = get_vectorstore()
    if vs._collection.count() == 0:
        ingest(vectorstore=vs)
    return vs, make_llm()
