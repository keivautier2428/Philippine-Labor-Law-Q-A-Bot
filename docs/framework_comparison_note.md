# Framework Comparison Note: LangChain vs. LlamaIndex

**What we actually built with:** LangChain (+ LangGraph, LangChain's orchestration layer). **We did not
implement anything in LlamaIndex**; the LlamaIndex half of this note is a design comparison, not a
description of our code.

## How each framework approaches data routing and node indexing

**LlamaIndex is data-first.** Source files become `Document`s, which are split by node parsers into
`Node`s (chunks that keep a link to their parent document plus metadata). Nodes go into an index
(`VectorStoreIndex`), and an index is exposed as a retriever or a *query engine*. Its
`IngestionPipeline` chains transformations (parse -> split -> embed), caches each node/transformation
combination, and, if given a docstore, tracks a hash per document ID so unchanged documents are skipped
and edited ones are re-processed. Routing is done at the query-engine level: a router query engine picks
among several indexes/engines (e.g. "article lookup" vs. "summaries") using a selector.

**LangChain is composition-first.** Data are `Document`s (text + metadata) handled by loose pieces: a
text splitter, an embeddings class, a vector-store integration. Behaviour is built by composing
runnables (`prompt | chat model | parser`) and, for anything with branches or loops, wiring them as a
LangGraph state graph. Routing is explicit control flow (conditional edges) rather than a property of
the index. Dedupe/refresh is not built into the chain; you implement it (or use LangChain's indexing
API with a record manager).

| Concern | LlamaIndex | LangChain / LangGraph |
|---|---|---|
| Chunk unit | `Node` (parent-linked) | `Document` (flat, metadata only) |
| Ingestion | `IngestionPipeline` + docstore hashing + cache | splitter + vector store; dedupe is your code |
| Retrieval abstraction | retriever / query engine over an index | `VectorStore` (+ `as_retriever`) |
| Routing | router query engines / selectors | conditional edges in a graph |
| Memory | chat engines with memory modules | LangGraph checkpointer per thread |

## What we used, and why

- **LangChain:** `Document` + `RecursiveCharacterTextSplitter` (article-aware chunking), the Chroma
  vector-store integration (`langchain-chroma`), the Gemini chat model (`langchain-google-genai`) with its
  built-in rate limiter, and `prompt | model | parser` chains for the router, answer and self-check steps.
- **LangGraph:** the route -> retrieve -> generate -> verify -> (regenerate) -> finish graph, and
  `InMemorySaver` for multi-turn memory keyed by session ID. We chose this over
  `RunnableWithMessageHistory`/`ConversationBufferMemory` because current LangChain releases mark those
  as deprecated in favour of LangGraph persistence.
- **Not used from LangChain:** `as_retriever()`; we call the vector store directly because we need the raw
  similarity score for the relevance threshold and for the sources shown to the user.
- **Why LangChain over LlamaIndex for this project:** the hard part of our app is control flow (refuse
  out-of-scope questions, rewrite follow-ups using memory, verify, regenerate), which maps directly onto a
  graph. Our corpus is one flat collection of short articles, so LlamaIndex's multi-index routing would
  add machinery we have no use for. **What we gave up:** `IngestionPipeline`'s built-in document
  management; we wrote the equivalent ourselves (per-document content hash stored in each chunk's Chroma
  metadata) in `src/ingestion/pipeline.py`.
- **Where LlamaIndex would win:** a much larger or heterogeneous corpus (full Labor Code plus DOLE
  issuances, PDFs, tables) where hierarchical nodes, auto-merging retrieval and per-source routing pay off.
