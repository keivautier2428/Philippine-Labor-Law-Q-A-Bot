# Model Customization Note: Fine-Tuning and LoRA/PEFT

**Status:** analysis and plan only. **We did not run any fine-tuning or LoRA experiment**, and nothing
below should be read as a result.

## Background
Full fine-tuning updates every weight of a model. Parameter-efficient fine-tuning (PEFT) freezes the base
model and trains a small set of extra parameters. LoRA (Low-Rank Adaptation) adds small low-rank matrices
beside selected weight matrices and trains only those, typically a small fraction of the model's
parameter count. QLoRA does the same on a base model stored in 4-bit precision to cut memory further.
Both produce a small adapter file that is loaded on top of the unchanged base model.

## Decision: we use the base model as-is, with RAG
1. **Our observed problems are data problems, not behaviour problems.** The weaknesses we know about are
   coverage (33 articles; no minimum-age provision) and currency (original article numbering). Fine-tuning
   cannot add a statute the corpus lacks; adding or updating documents does, and `src/ingestion/pipeline.py`
   does that in one command.
2. **Statutory facts are a poor fine-tuning target.** Fine-tuning is unreliable at injecting precise facts
   (figures, article numbers) and gives no citation to the source. It also goes stale the day a provision
   is amended, while a re-ingested document updates immediately. A legal bot that must cite is the case
   RAG is built for.
3. **Access and cost.** Our generator is the hosted Gemini API; we do not have its weights, so LoRA is not
   available on it. Tuning an open model instead would need a GPU and a separate serving setup, which
   conflicts with a free-tier CPU deployment.

## Where customization could genuinely help (ranked by expected value)
| Candidate | Why it fits | How we would measure it |
|---|---|---|
| **1. Fine-tune the embedding model** (contrastive, on question -> article pairs) | Retrieval is the stage that decides answer quality, and the model is small enough to tune on CPU/free Colab | `src/rag/evaluate.py` hit@1 / hit@3 before vs. after, on questions NOT used for training |
| **2. Distil the router/query-rewriter into a small open model with LoRA** | It is a narrow task with structured JSON output, and it costs 1 of our ~3 calls per question and counts against the free quota | JSON validity rate, scope accuracy, and retrieval hit@k of rewritten queries vs. the Gemini router |
| **3. Style/format tuning of the answerer** | Consistent citation format and tone | Only worthwhile after 1 and 2; prompt rules already enforce the format |

## Experiment we would run first (candidate 1)
- **Data:** 150-300 realistic employee questions, each labelled with the article(s) that answer it, written by
  the team (not generated from the article text, which would leak the answer). Split 70/30 by topic so the
  test set has unseen phrasing.
- **Method:** contrastive fine-tuning of `all-MiniLM-L6-v2` with in-batch negatives, small learning rate, early
  stopping on the held-out split.
- **Success criterion, decided in advance:** hit@3 improves on the held-out set without lowering the
  similarity gap between in-scope and out-of-scope questions (the gap that `MIN_RELEVANCE` depends on).
- **Main risks:** overfitting 33 articles with a few hundred pairs; a tuned model whose scores no longer
  match the calibrated threshold (re-run `evaluate.py` and recalibrate); and having to re-embed the whole
  index after the change.

## Adapter handling if we ever did LoRA (candidate 2)
Keep the adapter separate from the base weights, version it alongside the evaluation numbers that justified
it, and load it at container start. Do not bake private user conversations into training data.
