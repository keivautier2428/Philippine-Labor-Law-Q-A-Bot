# Presentation Runbook: Checkpoint 1 & 2

**Team:** Credo · Hinlayagan · Manatad | **Target:** ~8–10 min

---

## I. Problem & Motivation (~45 sec)

**Say:**
> "Filipino workers, HR staff, and small business owners constantly need quick answers to labor law questions — how much separation pay is owed, what counts as a just cause for termination. But the Labor Code itself is dense, cross-referenced legal text, and asking a generic chatbot risks a confident-sounding wrong answer with real financial and legal consequences. We built a Retrieval-Augmented Generation assistant that only answers from the actual statutory text, and cites the article it's quoting."

---

## II. Architecture Overview (~1 min)

**Show/draw this flow (on a slide or just say it out loud):**

```
Raw articles (LawPhil, PD 442)
      -> Preprocess (clean, tokenize)
      -> Embed (MiniLM, local)
      -> Chroma index (cosine similarity)
      -> 4-prompt pipeline (scope -> reformulate -> answer -> self-check)
      -> Gemini API
      -> Cited answer
```

**Say:**
> "A question comes in, gets checked for scope, rewritten into statutory phrasing, matched against the vector index, and answered strictly from what comes back — then the answer is checked a second time against the same context before it's shown to the user."

---

## III. Checkpoint 1 — Data & Embeddings (~2 min)

**1. Clean and tokenize the corpus**

Run:
```bash
python src/preprocessing/preprocess.py
# -> Processed 33 documents.
```

Show: `docs/preprocessing_report.md` — scroll to any before/after pair. Point out the raw HTML leftovers (`&nbsp;`, source citation block) versus the cleaned single sentence.

Say:
> "We deliberately kept the stopword list short — legal text relies on small words like 'shall' and 'notwithstanding' that actually carry meaning, so an aggressive generic filter would have stripped signal we need."

**2. Generate embeddings**

Run:
```bash
python src/embeddings/generate_embeddings.py
# -> Saved embeddings -> data/processed/embeddings.npy
```

Show: `docs/embedding_demo_report.md` — the sanity-check query *"Can my employer fire me without giving a reason?"* returns Article 282 (just-cause termination) as the #1 match.

Say:
> "We chose all-MiniLM-L6-v2 — a local, free model, no API key needed — specifically because it's trained for semantic retrieval. This result is genuine: nothing in that query mentions 'Article 282' or 'just cause,' it found the right law by meaning alone."

---

## IV. Checkpoint 2 — Prompts, Vector DB & Live API (~4 min)

**1. The four prompts**

Show: `docs/prompt_engineering_document.md` — open it and name the four out loud: **scope classifier** (screens before retrieval even runs), **query reformulation** (casual phrasing → statutory terms), **grounded answer** (citation required per claim), **self-check** (re-reads the draft against the same context).

**2. Build the vector index**

Run:
```bash
python src/vectorstore/build_index.py
# -> Indexed 33 documents into Chroma collection 'labor_code_articles'
```

Say:
> "Chroma runs locally, persists to disk, and we configured it for cosine similarity directly — matching what the embedding model was actually trained on."

**3. Prove retrieval works**

Run:
```bash
python src/vectorstore/query_demo.py
```

Show: `docs/vector_db_demo_report.md` — 4 of 5 sample queries land squarely on the right article. **Proactively point out the 5th:** *"Can a 16 year old legally be hired?"* returns weaker matches, because our 33-article subset doesn't yet include the minimum-age provision. Own this — it's honest evidence of real testing, not a cherry-picked demo.

**4. Distance metric comparison**

Run:
```bash
python src/vectorstore/distance_comparison.py
```

Show: `docs/distance_metric_report.md` — cosine and dot-product rankings are *identical*, empirically, because our embeddings measure unit-norm (std ≈ 0.000000). Point at that number.

Say:
> "We recommend cosine explicitly rather than relying on that equivalence — it stays correct even if a future embedding model doesn't happen to produce unit-norm vectors."

**5. Live API integration demo**

Run:
```bash
python src/llm/api_demo.py
# -> live calls to the Gemini Chat Completion API
```

Show: `docs/api_integration_demo_report.md` — read the note at the top first. Our first live run actually hit two genuine `REVISE` verdicts from self-check, because a data-pipeline bug was silently truncating every retrieved article to its first sentence (Article 282 lost its entire list of just causes). After fixing that bug, rerun the demo: both in-scope answers now cite specific figures (30%/50% premium pay, each individual just-cause ground) and pass self-check cleanly, plus the **refusal** on the California minimum-wage query still works correctly.

Say:
> "The self-check prompt did real work in both directions here — it caught a genuine grounding gap during our own testing, which is exactly what it's for, and then confirmed clean answers once we fixed the underlying data bug that caused the gap. That before-and-after is the strongest evidence we have that the safeguard isn't decorative."

After the 3 sample queries finish, the script prompts once for an optional live question — type a real one on stage to prove it isn't scripted, or press Enter to skip. One extra question costs 5 more Gemini calls.

**Quota warning:** Gemini's free tier caps out at 20 requests/day for this model, and one full run (3 sample queries) already uses ~15. **Do not rerun this script to rehearse** — you'll have nothing left for the actual presentation. If you do run out live, fall back to walking through the already-saved `docs/api_integration_demo_report.md` instead of re-running.

---

## V. Design Decisions & Trade-offs (~1 min, say fast as a list)

- **Local embeddings, not an API** — free, fast, no key required, identical across every teammate's machine
- **Four narrow prompts, not one** — each independently testable, catches a different failure mode
- **Chroma + cosine** — matches the embedding model's training objective, no server to run
- **Gemini for chat completion** — provider-agnostic by design; swapping `gemini_client.py` wouldn't touch the prompts or retrieval logic

---

## VI. Limitations & What's Next (~45 sec)

Name these yourself before they're asked — reads as control, not weakness:

- **33-article subset** is a deliberate Checkpoint 1/2 scope, not the full Labor Code — DOLE issuances and the rest of the code arrive in Checkpoint 3
- **Self-check doesn't loop back yet** — it logs `REVISE` but doesn't auto-regenerate the answer; that's Checkpoint 3 orchestration work
- **No user-facing app yet** — Checkpoint 3 adds the working application and conversational memory

---

## VII. Anticipated Questions

**Q: Why Gemini instead of OpenAI?**
A: The team had access to a Gemini key. The four-prompt architecture is provider-agnostic — `gemini_client.py` is a thin wrapper, so swapping providers wouldn't change any prompt or retrieval logic.

**Q: Why cosine similarity specifically?**
A: all-MiniLM-L6-v2 is trained with a cosine objective. We confirmed empirically our embeddings are unit-norm (std ≈ 0.000000), so cosine is both the theoretically and empirically correct choice — and Chroma's index is configured to match.

**Q: How do you prevent hallucination?**
A: Three layers — scope classifier keeps off-topic questions from ever reaching retrieval; grounded-answer prompt requires a citation per claim; self-check re-reads the draft against the same retrieved context afterward. We have concrete evidence of that third layer working: it caught a genuine grounding gap during testing (traced to a data-pipeline bug that was truncating retrieved articles), and confirmed clean, fully-cited answers after we fixed it.

**Q: What would break first if the document collection grew 100x?**
A: Retrieval precision on ambiguous queries — our minors/hiring-age query already exposed a coverage gap at just 33 documents. At 100x we'd also move off local Chroma persistence to a hosted vector database and likely add re-ranking.

**Q: Walk us through what happens when a user submits a query.**
A: Scope check first. If in-scope, reformulate into statutory language, embed, match against Chroma. Top matches become context for the grounded-answer prompt, which drafts a cited response. Self-check then re-verifies that draft against the same context before anything is shown.
