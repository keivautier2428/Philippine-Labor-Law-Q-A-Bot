"""
Checkpoint 3 - shared configuration
Philippine Labor Law Q&A Bot

Single place for paths and tunables so the ingestion pipeline, the RAG chain,
the CLI and the demo all agree. Everything overridable by environment variable
(or .env) so the same code runs on a teammate's laptop and, later, in Docker.
"""

import os

from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

# --- data locations -------------------------------------------------------
RAW_DIR = os.path.join(ROOT, "data", "raw")
INCOMING_DIR = os.path.join(ROOT, "data", "incoming")  # drop NEW documents here
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(ROOT, "data", "processed", "chroma_db_v3"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "labor_code_chunks")

# --- embeddings (must match Checkpoint 1/2 so vectors stay comparable) ------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- chunking -------------------------------------------------------------
# Most Labor Code articles are < 1,000 chars, so they stay as ONE chunk (the
# article is the natural legal unit). Only long articles are split.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# --- retrieval ------------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "4"))
# Backstop only: if even the best chunk is this dissimilar, don't call the
# answer model at all. UNCALIBRATED default - run src/rag/evaluate.py and tune.
MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE", "0.20"))

# --- LLM ------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Free tier is ~5 requests/minute; the limiter paces calls instead of crashing.
LLM_REQUESTS_PER_MINUTE = float(os.environ.get("LLM_REQUESTS_PER_MINUTE", "5"))

# --- conversation memory --------------------------------------------------
HISTORY_WINDOW_TURNS = int(os.environ.get("HISTORY_WINDOW_TURNS", "4"))  # user+bot pairs sent to the model
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "1"))

DISCLAIMER = "General information based on the Labor Code text available to this bot, not legal advice."
