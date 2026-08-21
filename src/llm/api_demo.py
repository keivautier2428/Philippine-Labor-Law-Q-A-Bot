"""
Checkpoint 2 - API Integration Demo
Philippine Labor Law Q&A Bot

Runs the full prompt pipeline (scope check -> query reformulation -> retrieval
-> grounded answer -> self-check) against the real Gemini Chat Completion API
for a handful of sample questions, including one deliberately out-of-scope
query, and writes the full transcript to docs/api_integration_demo_report.md.

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


def answer_question(question: str, log: list[str]) -> None:
    log.append(f"## Query: \"{question}\"")
    log.append("")

    # Step 1: scope check
    scope_raw = generate(SCOPE_CLASSIFIER_PROMPT.format(user_question=question))
    log.append(f"**Scope check:** `{scope_raw}`")
    log.append("")

    if "OUT_OF_SCOPE" in scope_raw.upper():
        refusal = (
            "I can only answer questions about Philippine labor law. "
            "For that question, please check the relevant local authority instead."
        )
        log.append(f"**Response (refused, no retrieval performed):** {refusal}")
        log.append("")
        return

    # Step 2: reformulate for retrieval
    reformulated = generate(QUERY_REFORMULATION_PROMPT.format(user_question=question))
    log.append(f"**Reformulated for retrieval:** `{reformulated}`")
    log.append("")

    # Step 3: retrieve context
    retrieved = retrieve(reformulated, k=3)
    log.append("**Retrieved articles:** " + ", ".join(d["source_file"] for d in retrieved))
    log.append("")

    # Step 4: grounded answer
    context = format_context(retrieved)
    draft_answer = generate(
        GROUNDED_ANSWER_PROMPT.format(retrieved_context=context, user_question=question)
    )
    log.append("**Draft answer:**")
    log.append("")
    log.append(draft_answer)
    log.append("")

    # Step 5: self-check
    check_result = generate(
        SELF_CHECK_PROMPT.format(retrieved_context=context, draft_answer=draft_answer)
    )
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
    ]

    for query in SAMPLE_QUERIES:
        answer_question(query, log)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print(f"Ran {len(SAMPLE_QUERIES)} queries through the full pipeline.")
    print(f"Transcript -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
