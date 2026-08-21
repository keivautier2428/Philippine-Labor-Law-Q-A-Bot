"""
Checkpoint 2 - Prompt Architecture
Philippine Labor Law Q&A Bot

The four structured prompts documented in docs/prompt_engineering_document.md.
Kept as plain format strings (not a template engine) since there are only four
and each is short enough to read end-to-end here.
"""

GROUNDED_ANSWER_PROMPT = """\
You are a Philippine labor law assistant. Answer the user's question using ONLY
the CONTEXT below, which consists of excerpts from the Labor Code of the
Philippines (Presidential Decree No. 442).

Rules:
1. Every factual claim must cite its source article, e.g. "(Article 282)".
2. If the CONTEXT does not contain enough information to answer, say so plainly
   instead of guessing or using outside knowledge.
3. Do not answer questions about non-Philippine law, or about anything other than
   Philippine labor law, even if the CONTEXT happens to contain related text.
4. Keep answers concise and in plain language a non-lawyer can understand.

CONTEXT:
{retrieved_context}

QUESTION:
{user_question}
"""

SCOPE_CLASSIFIER_PROMPT = """\
Classify the following user question as IN_SCOPE or OUT_OF_SCOPE for a bot that
only answers questions about Philippine labor law (the Labor Code of the
Philippines and related DOLE issuances).

OUT_OF_SCOPE examples: questions about labor law in other countries, criminal
law, tax law, immigration law, or anything unrelated to Philippine employment.

Respond with a single word: IN_SCOPE or OUT_OF_SCOPE.

QUESTION:
{user_question}
"""

QUERY_REFORMULATION_PROMPT = """\
Rewrite the user's question as a short search query using formal Philippine labor
law terminology. Do not answer the question. Output only the rewritten query.

Examples:
"can they just kick me out with no warning" -> "termination without due process just cause"
"do I get extra for working sunday" -> "premium pay rest day work"

QUESTION:
{user_question}
"""

SELF_CHECK_PROMPT = """\
You will check a draft answer against the CONTEXT it was supposed to be grounded
in. For each factual claim in the DRAFT ANSWER, verify it is directly supported
by the CONTEXT.

If every claim is supported and cited, respond with exactly: OK
If any claim is unsupported, fabricated, or missing a citation, respond with:
REVISE: <short explanation of what to fix>

CONTEXT:
{retrieved_context}

DRAFT ANSWER:
{draft_answer}
"""


def format_context(retrieved_docs: list[dict]) -> str:
    """Render retrieved {source_file, text} dicts as labeled context blocks."""
    blocks = []
    for doc in retrieved_docs:
        blocks.append(f"[{doc['source_file']}]\n{doc['text']}")
    return "\n\n".join(blocks)
