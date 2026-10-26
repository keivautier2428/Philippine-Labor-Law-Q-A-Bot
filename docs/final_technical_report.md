# Final Technical Report: Philippine Labor Law Q&A Bot

> Fields marked **[FILL]** need numbers from your own runs (`evaluate.py`, `demo.py`, the deployment). They are
> left empty on purpose so no result in this report is invented. Delete this note when done.

## 1. Problem and scope
Workers and small employers in the Philippines need quick, citable answers about labor rights, but the
Labor Code is long and written in legal prose. The bot answers questions in plain language using only the
Labor Code articles it holds, cites the article for every claim, and declines questions outside Philippine
labor law. It is information, not legal advice. Corpus: 33 articles of Presidential Decree No. 442 (Book One,
Book Three Arts. 82-95, Book Six Arts. 278-287; see `data/raw/_MANIFEST.md`).

## 2. Architecture
```
data/raw + data/incoming -> ingestion pipeline (parse, clean, chunk, embed) -> Chroma (cosine)
user -> Streamlit UI -> LangGraph: route -> retrieve -> generate -> verify -> (regenerate) -> finish
                                     |            |                          |
                          Gemini (scope + rewrite) |             Gemini (answer, self-check)
                                          MiniLM embeddings (local)
```
Embeddings run locally (`all-MiniLM-L6-v2`, normalised vectors); only generation uses the hosted Gemini API.

## 3. Data and ingestion (Checkpoints 1 and 3)
Articles were collected from a public source, cleaned (`src/preprocessing/`), and ingested by an automated
pipeline: parse (.txt/.md/.pdf), clean, article-aware chunking (short articles stay whole; long ones are
split with overlap and keep their article number), embed, store. Each chunk carries Book/Title/Chapter and a
document hash, which makes re-ingestion idempotent and lets edited files replace their old chunks.
Result of an ingest run: **[FILL: files / chunks / seconds]**.

## 4. Retrieval design (Checkpoints 1 and 2)
Embeddings were found to be unit-normalised, so cosine and dot-product rank identically; we use cosine in
Chroma (`docs/distance_metric_report.md`). Retrieval quality on a labelled set of 12 questions
(`src/rag/evaluate.py`): hit@1 **[FILL]**/12, hit@3 **[FILL]**/12. Chosen `MIN_RELEVANCE`: **[FILL]**, with the
similarity gap between in-scope and not-covered questions **[FILL]** (the script prints a calibration section).

## 5. Orchestration, memory and verification (Checkpoint 3)
- **Router:** one Gemini call decides in/out of scope and rewrites follow-ups into standalone queries using
  recent history. Scope check and query rewriting were merged to cut calls from 5 to about 3 per question
  because of the ~20 requests/day free quota.
- **Memory:** a LangGraph checkpointer keeps the conversation per session; only the last 4 exchanges are sent
  to the model, and earlier messages are never a source of legal facts.
- **Verification:** every cited article must appear in the retrieved context (no API call), then an LLM
  self-check; a failure triggers one regeneration, and a second failure shows the answer with a warning.
- **Framework choice:** LangChain + LangGraph (`docs/framework_comparison_note.md`).

## 6. Interface and deployment (Checkpoint 4)
Streamlit chat UI with a "Retrieved articles" panel showing similarity and which articles were cited. A
two-stage Dockerfile builds dependencies (CPU-only PyTorch), caches the embedding model and bakes the
vector index in a builder stage; the runtime image is slim, runs as non-root (UID 1000) and receives the API
key only as a run-time secret. GitHub Actions runs tests, builds and smoke-tests the image, then pushes to a
Hugging Face Space. Public-demo guards cap total daily LLM calls, questions per session and optionally require
a passcode. Deployed at **[FILL: URL]**; image size **[FILL]** MB; evidence in `docs/deployment_evidence.md`.

## 7. Testing and results
Automated: 30 offline tests (ingestion idempotence, chunking, router parsing, refusals, memory isolation and
windowing, verification loop, UI flow, budget and passcode guards). They use stand-in embeddings and a
scripted LLM, so they verify logic, not answer quality.
Live demo (`docs/rag_demo_report.md`): 6 questions in one session. Outcomes: **[FILL: for each of the 6,
pass/fail and one line why]**. Total Gemini calls: **[FILL]**.

## 8. Limitations and risks
- **Coverage:** 33 articles. Questions on provisions outside them (e.g. minimum employable age) are declined.
- **Currency:** the corpus uses the original article numbering and wording. **[FILL: after checking the
  current official text, state which articles are renumbered or amended.]**
- **Not legal advice:** every answer carries a disclaimer, but users may still over-trust it.
- **Quota and availability:** free-tier limits, free Space sleep, and in-memory state (memory and the daily
  budget reset on restart).
- **Router fails open:** unparseable router output is treated as in-scope; the relevance threshold, grounded
  prompt and verification are the remaining defences.
- **Evaluation size:** 12 labelled retrieval questions and 6 demo questions are too few for statistical claims.
- **Prompt injection / abuse:** a public text box can be used to try to override instructions; the grounded
  prompt and verification reduce but do not eliminate this.

## 9. Ethics and privacy
Questions may contain personal employment details. The app stores nothing on disk; conversations live in
process memory only. Questions are sent to Google's Gemini API, which users should be told. The key is a
run-time secret and was never committed.

## 10. Future work
Expand and update the corpus (current numbering, DOLE issuances); fine-tune the embedding model if
`evaluate.py` shows retrieval is the bottleneck (`docs/model_customization_note.md`); persistent
conversation storage; per-user rate limiting at the proxy level; larger labelled evaluation set.

## 11. Team contributions
**[FILL: who did what across Checkpoints 1-4, matching the README division of labor.]**
