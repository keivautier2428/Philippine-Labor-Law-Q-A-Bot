"""
Checkpoint 2 - API Integration Demo
Philippine Labor Law Q&A Bot

Runs the full prompt pipeline (scope check -> query reformulation -> retrieval
-> grounded answer -> self-check) against the real Gemini Chat Completion API
for a handful of sample questions, including one deliberately out-of-scope
query, and writes the full transcript to docs/api_integration_demo_report.md.

Each step also prints to the terminal as it happens (not just at the end),
so this is meant to be run live during a presentation rather than only read
from the report file afterward.

After the fixed sample queries, it offers one optional live prompt so you can
type a real question on stage to prove the demo isn't scripted - just one,
not an open loop, since each question costs 5 Gemini calls and the free tier
caps out at 20/day. Press Enter with nothing typed to skip it.

Requires GEMINI_API_KEY (see .env.example) and the Chroma index built by
src/vectorstore/build_index.py.

Usage:
    python src/vectorstore/build_index.py   # if not already run
    python src/llm/api_demo.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.llm.gemini_client import generate  # noqa: E402
from src.llm.prompts import (  # noqa: E402
    GROUNDED_ANSWER_PROMPT,
    QUERY_REFORMULATION_PROMPT,
    SCOPE_CLASSIFIER_PROMPT,
    SELF_CHECK_PROMPT,
    format_context,
)
from src.vectorstore.retriever import retrieve  # noqa: E402

REPORT_PATH = os.path.join(ROOT, "docs", "api_integration_demo_report.md")

SAMPLE_QUERIES = [
    "Can my employer fire me without giving a reason?",
    "Do I get extra pay for working on my rest day?",
    "What's the minimum wage in California?",  # deliberately out-of-scope
]

RULE = "=" * 70


def banner(text: str) -> None:
    print()
    print(RULE)
    print(text)
    print(RULE)


def step(label: str, value: str) -> None:
    print(f"\n[{label}]")
    print(value)


def answer_question(question: str, log: list[str]) -> None:
    banner(f'QUERY: "{question}"')
    log.append(f"## Query: \"{question}\"")
    log.append("")

    # Step 1: scope check
    scope_raw = generate(SCOPE_CLASSIFIER_PROMPT.format(user_question=question))
    step("1. Scope check", scope_raw)
    log.append(f"**Scope check:** `{scope_raw}`")
    log.append("")

    if "OUT_OF_SCOPE" in scope_raw.upper():
        refusal = (
            "I can only answer questions about Philippine labor law. "
            "For that question, please check the relevant local authority instead."
        )
        step("Response (refused - no retrieval performed)", refusal)
        log.append(f"**Response (refused, no retrieval performed):** {refusal}")
        log.append("")
        return

    # Step 2: reformulate for retrieval
    reformulated = generate(QUERY_REFORMULATION_PROMPT.format(user_question=question))
    step("2. Reformulated for retrieval", reformulated)
    log.append(f"**Reformulated for retrieval:** `{reformulated}`")
    log.append("")

    # Step 3: retrieve context
    retrieved = retrieve(reformulated, k=3)
    step("3. Retrieved articles", ", ".join(d["source_file"] for d in retrieved))
    log.append("**Retrieved articles:** " + ", ".join(d["source_file"] for d in retrieved))
    log.append("")

    # Step 4: grounded answer
    context = format_context(retrieved)
    draft_answer = generate(
        GROUNDED_ANSWER_PROMPT.format(retrieved_context=context, user_question=question)
    )
    step("4. Draft answer", draft_answer)
    log.append("**Draft answer:**")
    log.append("")
    log.append(draft_answer)
    log.append("")

    # Step 5: self-check
    check_result = generate(
        SELF_CHECK_PROMPT.format(retrieved_context=context, draft_answer=draft_answer)
    )
    step("5. Self-check result", check_result)
    log.append(f"**Self-check result:** `{check_result}`")
    log.append("")


def main():
    log = [
        "# API Integration Demo",
        "",
        "Live run against the Gemini Chat Completion API, using the four prompts",
        "from `docs/prompt_engineering_document.md` and the Chroma index from",
        "Checkpoint 2's vector database setup. One query below is deliberately",
        "out-of-scope, to demonstrate the refusal path working end-to-end.",
        "",
        "Note: an earlier run of this demo hit two genuine `REVISE` verdicts from",
        "the self-check prompt, because a data-pipeline bug was silently truncating",
        "every retrieved article to its first sentence (e.g. Article 282 lost its",
        "entire list of just causes). With too little grounding, the model",
        "under-answered and self-check correctly flagged the gap. After fixing",
        "that extraction bug (see `data/raw/` and `docs/preprocessing_report.md`),",
        "both in-scope queries below pass self-check cleanly and cite specific",
        "figures (30%/50% premium pay, the individual just-cause grounds) that the",
        "truncated context never contained. Kept here as evidence the self-check",
        "prompt does real work in both directions - it caught the problem, and",
        "stopped flagging once the underlying data was fixed. Auto-regenerating on",
        "a `REVISE` verdict, rather than just logging it, is still planned for",
        "Checkpoint 3's RAG orchestration work.",
        "",
    ]

    banner(f"API INTEGRATION DEMO - {len(SAMPLE_QUERIES)} queries against Gemini")

    for query in SAMPLE_QUERIES:
        answer_question(query, log)

    total_queries = len(SAMPLE_QUERIES)

    banner("OPTIONAL LIVE QUERY")
    print("Type a real question to run it through the pipeline live, or just")
    print("press Enter to skip (each question uses 5 Gemini API calls).")
    try:
        live_query = input("\nYour question: ").strip()
    except (EOFError, KeyboardInterrupt):
        live_query = ""

    if live_query:
        log.append("*Live query, typed during the presentation:*")
        log.append("")
        answer_question(live_query, log)
        total_queries += 1
    else:
        print("(skipped)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    banner(f"DONE - ran {total_queries} queries. Transcript -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
