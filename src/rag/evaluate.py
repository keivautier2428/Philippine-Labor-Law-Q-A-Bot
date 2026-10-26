"""
Checkpoint 3 - retrieval evaluation (ZERO LLM calls)
Philippine Labor Law Q&A Bot

Why this exists: the Gemini free tier allows ~20 requests/day, so you cannot
evaluate retrieval by running the full chat pipeline repeatedly. Retrieval is
local, so it can be measured as often as you like. This script:

  1. checks that each labelled question retrieves its expected article
     (hit@1 / hit@3 over the in-corpus set), and
  2. prints the best-match similarity for off-topic / uncovered questions so you
     can pick a defensible MIN_RELEVANCE instead of guessing.

Usage:
    python -m src.ingestion.pipeline      # build the index first
    python -m src.rag.evaluate            # writes docs/rag_retrieval_eval.md
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag import config  # noqa: E402

# (question, expected article number). Labels were written from the article
# text in data/raw/, not from what the retriever happens to return.
IN_CORPUS = [
    ("Can my employer fire me without giving a reason?", 282),
    ("Do I get paid extra for working on my rest day?", 93),
    ("How many hours can I be made to work in a day?", 83),
    ("What happens to my job if the company temporarily shuts down?", 286),
    ("How much separation pay do I get if the company closes down?", 283),
    ("How long can a probationary period last?", 281),
    ("Am I entitled to night shift pay?", 86),
    ("How many days of paid leave do I get after a year of work?", 95),
    ("Do I get paid for regular holidays I don't work?", 94),
    ("Can I resign without giving a reason?", 285),
    ("At what age can an employee be retired?", 287),
    ("Can an employer hire a Filipino directly for work abroad?", 18),
]

# Should NOT be answerable from the corpus. The 16-year-old question is
# in-scope legally but the minimum-age article is not in our 33-article subset.
NOT_COVERED = [
    "What's the minimum wage in California?",
    "How do I file my income tax return?",
    "Can a 16 year old legally be hired for a job?",
    "What is the best pizza place in Manila?",
]


def run(vectorstore, k: int = 3) -> dict:
    def search(q):
        raw = vectorstore.similarity_search_with_score(q, k=k)
        return sorted(((d, 1.0 - float(dist)) for d, dist in raw), key=lambda x: x[1], reverse=True)

    rows, hit1, hit3 = [], 0, 0
    for q, expected in IN_CORPUS:
        hits = search(q)
        arts = [d.metadata.get("article") for d, _ in hits]
        h1, h3 = arts[:1] == [expected], expected in arts
        hit1 += h1
        hit3 += h3
        rows.append((q, expected, arts, [round(s, 3) for _, s in hits], h1, h3))

    not_rows = []
    for q in NOT_COVERED:
        hits = search(q)
        not_rows.append((q, [d.metadata.get("article") for d, _ in hits], round(hits[0][1], 3)))

    in_tops = [r[3][0] for r in rows]
    out_tops = [r[2] for r in not_rows]
    return {
        "rows": rows,
        "not_rows": not_rows,
        "hit1": hit1,
        "hit3": hit3,
        "n": len(IN_CORPUS),
        "min_in_top": min(in_tops),
        "max_out_top": max(out_tops),
    }


def render(res: dict) -> str:
    n = res["n"]
    L = [
        "# Retrieval Evaluation (no LLM calls)",
        "",
        f"**Index:** `{config.COLLECTION_NAME}`, cosine similarity, `{config.EMBEDDING_MODEL}`.",
        "",
        f"- hit@1: **{res['hit1']}/{n}**   hit@3: **{res['hit3']}/{n}**",
        "",
        "## In-corpus questions (expected article should be in the top 3)",
        "",
        "| Question | Expected | Top-3 retrieved | Similarities | @1 | @3 |",
        "|---|---|---|---|---|---|",
    ]
    for q, exp, arts, sims, h1, h3 in res["rows"]:
        L.append(f"| {q} | Art. {exp} | {arts} | {sims} | {'yes' if h1 else 'no'} | {'yes' if h3 else 'no'} |")
    L += [
        "",
        "## Off-topic / not-covered questions (should score LOW)",
        "",
        "| Question | Top-3 retrieved | Best similarity |",
        "|---|---|---|",
    ]
    for q, arts, top in res["not_rows"]:
        L.append(f"| {q} | {arts} | {top} |")
    lo, hi = res["max_out_top"], res["min_in_top"]
    L += ["", "## Threshold calibration", ""]
    if hi > lo:
        L.append(
            f"Lowest best-match among in-corpus questions: **{hi}**. Highest best-match among not-covered "
            f"questions: **{lo}**. They separate, so a `MIN_RELEVANCE` between them (midpoint "
            f"**{round((hi + lo) / 2, 3)}**) would refuse the not-covered set without refusing the in-corpus set."
        )
    else:
        L.append(
            f"Lowest best-match among in-corpus questions (**{hi}**) is NOT above the highest among not-covered "
            f"questions (**{lo}**): similarity alone cannot cleanly separate them on this set. This is why the "
            "LLM router (scope check) stays in the pipeline and the threshold is only a backstop - keep "
            "`MIN_RELEVANCE` low."
        )
    return "\n".join(L) + "\n"


def main():
    from src.ingestion.pipeline import get_vectorstore

    vs = get_vectorstore()
    res = run(vs)
    text = render(res)
    print(text)
    out = os.path.join(ROOT, "docs", "rag_retrieval_eval.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
