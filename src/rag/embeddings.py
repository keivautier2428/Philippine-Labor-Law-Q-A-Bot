"""
Checkpoint 3 - LangChain-compatible embeddings
Philippine Labor Law Q&A Bot

Thin adapter that lets LangChain / Chroma use the SAME local model as
Checkpoints 1-2 (all-MiniLM-L6-v2). Written directly against sentence-
transformers instead of depending on langchain-huggingface: one fewer package
whose API changes between versions. Vectors are L2-normalised, matching the
unit-norm finding in docs/distance_metric_report.md.
"""

from langchain_core.embeddings import Embeddings

from src.rag.config import EMBEDDING_MODEL


class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None  # lazy: importing torch is slow, only pay when needed

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
