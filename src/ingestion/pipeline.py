"""
Checkpoint 3 - Automated Ingestion Pipeline
Philippine Labor Law Q&A Bot

parse -> clean -> chunk -> embed -> store, end to end, with no manual steps.

Drop a new .txt / .md / .pdf into data/incoming/ (or data/raw/) and run:

    python -m src.ingestion.pipeline            # ingest anything new/changed
    python -m src.ingestion.pipeline --dry-run  # show what WOULD happen
    python -m src.ingestion.pipeline --reset    # wipe the collection, rebuild

Design points (each one answers a way the Checkpoint 1/2 demo was fragile):
  * Idempotent: every chunk stores a hash of its source document. Unchanged
    files are skipped, edited files are replaced, so re-running is always safe.
    State lives in Chroma itself, not a second manifest file that can drift.
  * One bad file never aborts the run; it is reported and the rest continue.
  * Reuses Checkpoint 1's strip_header/clean_text so cleaning has one source
    of truth.
  * Article-aware chunking: a Labor Code article is the natural unit of
    citation, so short articles stay whole; only long ones are split, and every
    chunk carries article number + Book/Title/Chapter metadata for citations.
"""

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from src.preprocessing.preprocess import HEADER_SEPARATOR, clean_text, strip_header  # noqa: E402
from src.rag import config  # noqa: E402

SUPPORTED = {".txt", ".md", ".pdf"}
ARTICLE_RE = re.compile(r"^Article\s+(\d+)\.\s*(.*?)\.(?:\s|$)")


# --------------------------------------------------------------------------
# 1. PARSE
# --------------------------------------------------------------------------
@dataclass
class ParsedDoc:
    source: str                      # path relative to repo root, forward slashes
    text: str                        # cleaned body text
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _parse_header(raw: str) -> dict:
    """Pull Book / Title / Chapter out of the manifest-style header that the
    Checkpoint 1 data collection step wrote above the '====' line."""
    idx = raw.find(HEADER_SEPARATOR)
    meta: dict = {}
    if idx == -1:
        return meta
    for line in raw[:idx].splitlines():
        line = line.strip()
        if re.match(r"^BOOK\b", line, re.I):
            meta["book"] = line.title()
        elif re.match(r"^Title\b", line, re.I):
            meta["title_heading"] = line
        elif re.match(r"^Chapter\b", line, re.I):
            meta["chapter"] = line
    return meta


def parse_file(path: str) -> ParsedDoc:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw = _read_pdf(path)
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read()

    meta = _parse_header(raw)
    text = clean_text(strip_header(raw))
    if not text:
        raise ValueError("document is empty after cleaning")

    m = ARTICLE_RE.match(text)
    if m:
        meta["article"] = int(m.group(1))
        meta["article_title"] = m.group(2).strip()
    else:
        fm = re.search(r"art_(\d+)", os.path.basename(path))
        if fm:
            meta["article"] = int(fm.group(1))

    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    meta["source"] = rel
    meta["doc_title"] = meta.get("article_title") or os.path.splitext(os.path.basename(path))[0]
    return ParsedDoc(
        source=rel,
        text=text,
        metadata=meta,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


# --------------------------------------------------------------------------
# 2. CHUNK
# --------------------------------------------------------------------------
def chunk_document(doc: ParsedDoc, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        # Legal prose: prefer breaking between list items (';') and sentences.
        separators=["; ", ". ", ", ", " ", ""],
        keep_separator="end",
    )
    pieces = splitter.split_text(doc.text) if len(doc.text) > size else [doc.text]
    base = os.path.splitext(os.path.basename(doc.source))[0]
    chunks = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if i > 0 and "article" in doc.metadata:
            # continuation chunks must still say which article they belong to,
            # both for embedding quality and so the model can cite correctly
            piece = f"(Article {doc.metadata['article']}, continued) {piece}"
        meta = dict(doc.metadata)
        meta.update(
            doc_hash=doc.content_hash,
            chunk_index=i,
            chunk_count=len(pieces),
            chunk_id=f"{base}::{i}",
        )
        chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


# --------------------------------------------------------------------------
# 3-5. EMBED + STORE (Chroma through LangChain)
# --------------------------------------------------------------------------
def get_vectorstore(embeddings=None, persist_directory: str | None = None, collection_name: str | None = None):
    from langchain_chroma import Chroma

    if embeddings is None:
        from src.rag.embeddings import MiniLMEmbeddings

        embeddings = MiniLMEmbeddings()
    return Chroma(
        collection_name=collection_name or config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory or config.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # same metric as Checkpoint 2's recommendation
    )


def discover(dirs: list[str]) -> list[str]:
    found = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.startswith((".", "_")):
                continue  # .gitkeep, _MANIFEST.md, _corpus.jsonl ...
            if os.path.splitext(name)[1].lower() in SUPPORTED:
                found.append(os.path.join(d, name))
    return found


@dataclass
class IngestStats:
    added: list = field(default_factory=list)
    updated: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)   # (source, reason)
    chunks_written: int = 0

    def summary(self) -> str:
        lines = [
            f"added:   {len(self.added)}",
            f"updated: {len(self.updated)}",
            f"skipped: {len(self.skipped)} (unchanged)",
            f"failed:  {len(self.failed)}",
            f"chunks written: {self.chunks_written}",
        ]
        for src, why in self.failed:
            lines.append(f"  FAILED {src}: {why}")
        return "\n".join(lines)


def ingest(
    paths: list[str] | None = None,
    vectorstore=None,
    reset: bool = False,
    dry_run: bool = False,
    log=print,
) -> IngestStats:
    stats = IngestStats()
    paths = paths if paths is not None else discover([config.RAW_DIR, config.INCOMING_DIR])
    vs = vectorstore if vectorstore is not None else get_vectorstore()

    if reset and not dry_run:
        vs.reset_collection()

    for path in paths:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        try:
            doc = parse_file(path)
            existing = vs.get(where={"source": doc.source}, include=["metadatas"])
            old_ids = existing.get("ids", [])
            old_hashes = {m.get("doc_hash") for m in existing.get("metadatas", [])}

            if old_ids and old_hashes == {doc.content_hash}:
                stats.skipped.append(rel)
                continue

            chunks = chunk_document(doc)
            (stats.updated if old_ids else stats.added).append(rel)
            if dry_run:
                continue

            if old_ids:
                vs.delete(ids=old_ids)
            ids = [f"{c.metadata['source']}::{c.metadata['chunk_index']}" for c in chunks]
            vs.add_documents(chunks, ids=ids)
            stats.chunks_written += len(chunks)
            log(f"  ingested {rel}  ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            stats.failed.append((rel, f"{type(e).__name__}: {e}"))
            log(f"  FAILED   {rel}: {e}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Ingest documents into the Labor Law vector store.")
    ap.add_argument("paths", nargs="*", help="specific files (default: everything in data/raw and data/incoming)")
    ap.add_argument("--reset", action="store_true", help="delete the collection first and rebuild")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    print(f"Vector store: {config.CHROMA_DIR}  (collection '{config.COLLECTION_NAME}')")
    stats = ingest(paths=args.paths or None, reset=args.reset, dry_run=args.dry_run)
    print("\n" + ("DRY RUN - nothing written\n" if args.dry_run else "") + stats.summary())
    if stats.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
