# Distance Metric Comparison

**Corpus:** 33 embedded Labor Code articles, 384-dim vectors (`sentence-transformers/all-MiniLM-L6-v2`).
**Embeddings are unit-normalized:** True (mean norm = 1.0000, std = 0.000000).

Three metrics compared on the same embedded corpus, same queries:

| Metric | What it measures | Range | Sensitive to vector magnitude? |
|---|---|---|---|
| Cosine similarity | Angle between vectors | [-1, 1], higher = more similar | No |
| Euclidean distance | Straight-line distance | [0, inf), lower = more similar | Yes |
| Dot product | Projection / raw overlap | (-inf, inf), higher = more similar | Yes |

## Query: "Can my employer fire me without giving a reason?"

| Rank | Cosine | Euclidean | Dot Product |
|---|---|---|---|
| 1 | `labor_code_art_282.txt` (0.4862) | `labor_code_art_282.txt` (1.0137) | `labor_code_art_282.txt` (0.4862) |
| 2 | `labor_code_art_285.txt` (0.4226) | `labor_code_art_285.txt` (1.0746) | `labor_code_art_285.txt` (0.4226) |
| 3 | `labor_code_art_286.txt` (0.3422) | `labor_code_art_286.txt` (1.1470) | `labor_code_art_286.txt` (0.3422) |
| 4 | `labor_code_art_279.txt` (0.3373) | `labor_code_art_279.txt` (1.1513) | `labor_code_art_279.txt` (0.3373) |
| 5 | `labor_code_art_92.txt` (0.3350) | `labor_code_art_92.txt` (1.1532) | `labor_code_art_92.txt` (0.3350) |

Cosine and dot-product rankings match exactly for this query (expected, since the corpus is unit-normalized).

## Query: "Do I get paid extra for working on my rest day?"

| Rank | Cosine | Euclidean | Dot Product |
|---|---|---|---|
| 1 | `labor_code_art_93.txt` (0.6310) | `labor_code_art_93.txt` (0.8591) | `labor_code_art_93.txt` (0.6310) |
| 2 | `labor_code_art_92.txt` (0.6169) | `labor_code_art_92.txt` (0.8754) | `labor_code_art_92.txt` (0.6169) |
| 3 | `labor_code_art_87.txt` (0.5255) | `labor_code_art_87.txt` (0.9742) | `labor_code_art_87.txt` (0.5255) |
| 4 | `labor_code_art_88.txt` (0.4629) | `labor_code_art_88.txt` (1.0365) | `labor_code_art_88.txt` (0.4629) |
| 5 | `labor_code_art_86.txt` (0.4559) | `labor_code_art_86.txt` (1.0431) | `labor_code_art_86.txt` (0.4559) |

Cosine and dot-product rankings match exactly for this query (expected, since the corpus is unit-normalized).

## Timing (average over 20 runs, full corpus)

- cosine: 0.0614 ms
- euclidean: 0.0371 ms
- dot: 0.0057 ms

## Recommendation

**Cosine similarity** is the right default for this project. The embedding model (`all-MiniLM-L6-v2`) is trained with a cosine-similarity objective, and its output vectors are effectively unit-normalized (confirmed above), which is exactly why cosine and dot-product rankings come out identical on this corpus — cosine similarity is just the normalized form of dot product. We still recommend cosine explicitly, rather than relying on that equivalence, because it stays correct even if a future embedding step (e.g. a different model, or pooled/averaged vectors for longer DOLE documents in later checkpoints) produces vectors that aren't unit-norm. Euclidean distance is not a good fit here: at this dimensionality and corpus size it produced the same top result as cosine/dot in practice, but it conflates 'how long is this article' with 'how relevant is this article' whenever document lengths vary, which cosine similarity deliberately ignores. This is also why the Chroma index in `src/vectorstore/build_index.py` is configured with `hnsw:space: cosine`.
