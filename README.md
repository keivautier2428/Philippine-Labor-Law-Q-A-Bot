# Philippine Labor Law Q&A Bot

A Retrieval-Augmented Generation (RAG) capstone project for IS Professional
Elective #4 (Generative AI Systems). The bot grounds its answers in the
Philippine Labor Code (Presidential Decree No. 442) so workers, HR staff, and
small-business owners can ask plain-language questions about labor law and get
answers cited back to the actual statute.

## Team

| Name | Role | Pipeline Component |
|---|---|---|
| Jeziel Lea Mitch Credo | Lead Developer | Data Pipeline & Embeddings |
| Nicka Hinlayagan | AI Orchestrator | RAG Architecture & Prompting |
| Jennilyn Manatad | DevOps/UI | Deployment & Interface |

## Checkpoint 1 — Foundation & Data Pipeline (Prelim)

Status: complete. Deliverables:

1. **Project Proposal** — [`docs/Project_Proposal.docx`](docs/Project_Proposal.docx)
2. **Data Collection** — [`data/raw/`](data/raw/): 33 articles from the Labor Code
   of the Philippines (Book One: Pre-Employment, Book Three: Conditions of
   Employment, Book Six: Post-Employment), sourced from
   [lawphil.net](https://lawphil.net/statutes/presdecs/pd1974/pd_442_1974.html)
   (The LawPhil Project, CC BY-NC 3.0). See `data/raw/_MANIFEST.md` for the
   full file list.
3. **Text Preprocessing** — [`src/preprocessing/preprocess.py`](src/preprocessing/preprocess.py)
   cleans and tokenizes every raw article; before/after evidence is in
   [`docs/preprocessing_report.md`](docs/preprocessing_report.md).
4. **Embedding Generation Demo** — [`src/embeddings/generate_embeddings.py`](src/embeddings/generate_embeddings.py)
   embeds the cleaned corpus with a local Hugging Face model
   (`sentence-transformers/all-MiniLM-L6-v2`); results and model justification
   are in [`docs/embedding_demo_report.md`](docs/embedding_demo_report.md).
5. **Reflection** — [`docs/reflection.md`](docs/reflection.md)

## Checkpoint 2 — Prompt Architecture & Vector Indexing (Midterm)

Status: complete. Deliverables:

6. **Prompt Engineering Document** — [`docs/prompt_engineering_document.md`](docs/prompt_engineering_document.md):
   4 structured prompts (grounded answer, scope refusal, query reformulation,
   self-check), each with a hallucination/off-topic prevention example.
   Implemented in [`src/llm/prompts.py`](src/llm/prompts.py).
7. **API Integration Demo** — [`src/llm/api_demo.py`](src/llm/api_demo.py) runs
   the full prompt pipeline against the real Gemini Chat Completion API
   (`src/llm/gemini_client.py`), including a deliberately out-of-scope query to
   demonstrate the refusal path. Transcript in
   [`docs/api_integration_demo_report.md`](docs/api_integration_demo_report.md).
8. **Vector Database Setup** — [`src/vectorstore/build_index.py`](src/vectorstore/build_index.py)
   loads the Checkpoint 1 corpus into a persistent Chroma collection;
   [`src/vectorstore/query_demo.py`](src/vectorstore/query_demo.py) proves
   similarity search works with 5 sample queries. Results in
   [`docs/vector_db_demo_report.md`](docs/vector_db_demo_report.md).
9. **Distance Metric Comparison** — [`src/vectorstore/distance_comparison.py`](src/vectorstore/distance_comparison.py)
   compares Cosine, Euclidean, and Dot Product similarity on our own corpus;
   full write-up and recommendation in
   [`docs/distance_metric_report.md`](docs/distance_metric_report.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Checkpoint 2 also needs a Gemini API key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your-key-here
```

`.env` is git-ignored — never commit your key.

## Running the Pipeline

```bash
# Checkpoint 1: clean/tokenize the raw Labor Code articles, then embed them
python src/preprocessing/preprocess.py
python src/embeddings/generate_embeddings.py

# Checkpoint 2: build the vector index, prove retrieval works, compare
# distance metrics, then run the full prompt pipeline against Gemini
python src/vectorstore/build_index.py
python src/vectorstore/query_demo.py
python src/vectorstore/distance_comparison.py
python src/llm/api_demo.py
```

Outputs land in `data/processed/` (cleaned JSON, `_corpus.jsonl`,
`embeddings.npy`, `embeddings_metadata.json`, the `chroma_db/` index) and
`docs/` (all reports referenced above).

## Repository Structure

```
data/
  raw/          # source Labor Code articles (33 .txt files + manifest)
  processed/    # cleaned/tokenized JSON, combined corpus, embeddings, chroma_db/
src/
  preprocessing/  # text cleaning & tokenization
  embeddings/     # embedding generation
  vectorstore/    # Chroma index build/query, distance metric comparison
  llm/            # prompts, Gemini API client, API integration demo
docs/           # proposal, checkpoint reports, prompt engineering document
```

Later checkpoints (RAG orchestration, deployment) will extend this structure
with `/app` and a `Dockerfile` per the course's repository requirements.
