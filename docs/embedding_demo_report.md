# Embedding Generation Demo

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face `sentence-transformers`, run locally)
**Embedding dimensionality:** 384
**Documents embedded:** 33

## Why this model

`all-MiniLM-L6-v2` was chosen because it is trained specifically for semantic
similarity/retrieval, is small enough (~80MB) to run on CPU with no GPU or API key,
and its 384-dim vectors keep the Checkpoint 2 vector index lightweight for a
33-document corpus. It trades some accuracy against larger models (e.g.
`all-mpnet-base-v2`) in exchange for speed and zero infrastructure cost — an
appropriate trade-off for this stage of the project.

## Sanity-check query

Query: *"Can my employer fire me without giving a reason?"*

Top 5 nearest documents by cosine similarity:

- `labor_code_art_282.txt` (similarity: 0.4862) — Article 282. Termination by employer. An employer may terminate an employment for any of the following causes:...
- `labor_code_art_285.txt` (similarity: 0.4226) — Article 285. Termination by employee....
- `labor_code_art_286.txt` (similarity: 0.3422) — Article 286. When employment not deemed terminated. The bona-fide suspension of the operation of a business or undertaking for a period not ...
- `labor_code_art_279.txt` (similarity: 0.3373) — Article 279. Security of tenure. In cases of regular employment, the employer shall not terminate the services of an employee except for a j...
- `labor_code_art_92.txt` (similarity: 0.3350) — Article 92. When employer may require work on a rest day. The employer may require his employees to work on any day:...

This confirms the embeddings place semantically related Labor Code articles
(here, the just-cause/termination provisions) close together in vector space,
which is exactly the retrieval behavior the Checkpoint 3 RAG pipeline depends on.
