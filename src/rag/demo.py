"""
Checkpoint 3 - live demo (Deliverable 14)
Philippine Labor Law Q&A Bot

Runs ONE continuous conversation of 6 realistic questions through the real
pipeline and writes the transcript to docs/rag_demo_report.md.

What the script exercises:
  Q1        grounded answer                       (Art. 282 etc.)
  Q2, Q3    FOLLOW-UPS that only make sense with memory ("that", "in that case")
  Q4        a new topic in the same session        (rest-day premium pay)
  Q5        in-scope but NOT covered by our corpus (must say so, not guess)
  Q6        out-of-scope (must be refused)

Quota: ~3 Gemini calls per answered question, 1 for the refusal => about 16
calls total. The free tier allows ~20/day, so run this ONCE per day and do not
rehearse with it (use src/rag/evaluate.py for free, local retrieval checks).
    --no-self-check  cuts it to ~11 calls.

    python -m src.ingestion.pipeline
    python -m src.rag.demo
"""

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORT = os.path.join(ROOT, "docs", "rag_demo_report.md")

QUESTIONS = [
    ("grounded answer", "Can my employer fire me without giving a reason?"),
    ("memory: follow-up #1", "What if the company is just closing down instead?"),
    ("memory: follow-up #2", "In that case, how much separation pay would I get?"),
    ("new topic, same session", "Do I get extra pay if I'm made to work on my rest day?"),
    ("in scope, NOT in corpus (must not guess)", "Can a 16 year old legally be hired for a job?"),
    ("OUT OF SCOPE (must refuse)", "What's the minimum wage in California?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    from src.rag.cli import build_app

    rag = build_app(self_check=not args.no_self_check)
    started = time.strftime("%Y-%m-%d %H:%M")
    log = [
        "# RAG Demo Transcript (Checkpoint 3)",
        "",
        f"Run: {started} | model: `{os.environ.get('GEMINI_MODEL', 'default')}` | self-check: {not args.no_self_check}",
        "",
        "One continuous session: Q2 and Q3 depend on conversation memory.",
        "",
    ]
    total_calls = 0
    for i, (purpose, q) in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 70}\nQ{i} [{purpose}]\n  {q}\n{'=' * 70}")
        r = rag.ask(q, session_id="demo")
        total_calls += r.llm_calls
        print(r.answer)
        print(f"\n  status={r.status}  rewritten query={r.search_query!r}  verified={r.verified}  "
              f"revisions={r.revisions}  llm_calls={r.llm_calls}  top_similarity={r.top_score}")
        log += [
            f"## Q{i} - {purpose}",
            "",
            f"**User:** {q}",
            "",
            f"**Bot:** {r.answer}",
            "",
            f"- status: `{r.status}`",
            f"- query sent to retriever (after memory-aware rewrite): `{r.search_query}`",
            f"- retrieved: {[(s['article'], s['score'], 'cited' if s['cited'] else '') for s in r.sources]}",
            f"- verified: {r.verified}, revisions: {r.revisions}, LLM calls: {r.llm_calls}",
            "",
        ]
    log.append(f"**Total LLM calls: {total_calls}**")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nDone. {total_calls} LLM calls. Transcript -> {REPORT}")


if __name__ == "__main__":
    main()
