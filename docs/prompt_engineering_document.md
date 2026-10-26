# Prompt Engineering Document

Philippine Labor Law Q&A Bot — Checkpoint 2 (updated in Checkpoint 3, see the final section)

This document specifies the structured prompts used to control the bot's behavior.
Each prompt has a single, narrow job (retrieval query rewriting, grounded answer
generation, or out-of-scope refusal) rather than one do-everything prompt, so that
each piece can be tested and tuned independently as the RAG pipeline grows in
Checkpoint 3. All four are implemented in
[`src/llm/prompts.py`](../src/llm/prompts.py) and exercised by
[`src/llm/api_demo.py`](../src/llm/api_demo.py).

---

## 1. Grounded Answer Prompt

**Purpose:** The main answer-generation prompt. Forces the model to answer *only*
from the retrieved Labor Code excerpts passed in as context, cite the specific
article number for every claim, and explicitly say so instead of guessing when the
retrieved context doesn't cover the question.

**Prompt:**

```
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
```

**Example — how it prevents hallucination:**

Query: *"Can my employer fire me without giving a reason?"*

Retrieved context includes Article 282 (just causes) and Article 279 (security of
tenure). With the rule set above, the model answers:

> No — under Article 282, an employer may only terminate employment for specific
> just causes (e.g., serious misconduct, gross neglect of duty)... [continues with
> citations]

Without rule #1 (citation requirement) and rule #2 (say-so-if-insufficient), a
general-purpose prompt tends to pad the answer with plausible-sounding but
uncited detail (e.g., inventing a specific notice period not present in the
retrieved articles). Forcing a citation per claim makes fabricated claims visibly
un-cited, which is easy for a reviewer — or the model itself in the self-check
prompt below — to catch.

---

## 2. Out-of-Scope Refusal Prompt

**Purpose:** A lightweight classification prompt run *before* retrieval. Screens
the incoming question and flags it as in-scope or out-of-scope so the bot can
decline cleanly instead of retrieving irrelevant Labor Code articles and
answering anyway.

**Prompt:**

```
Classify the following user question as IN_SCOPE or OUT_OF_SCOPE for a bot that
only answers questions about Philippine labor law (the Labor Code of the
Philippines and related DOLE issuances).

OUT_OF_SCOPE examples: questions about labor law in other countries, criminal
law, tax law, immigration law, or anything unrelated to Philippine employment.

Respond with a single word: IN_SCOPE or OUT_OF_SCOPE.

QUESTION:
{user_question}
```

**Example — how it prevents off-topic answers:**

Query: *"What's the minimum wage in California?"*

Without this gate, the Grounded Answer Prompt above might still receive a few
weakly-related retrieved chunks (e.g., an article that happens to mention "wage")
and produce a confident-sounding but wrong or misleading answer about US law
using Philippine statutory language. The refusal prompt catches this *before*
retrieval runs, and the bot responds with a fixed decline message: "I can only
answer questions about Philippine labor law — for California minimum wage, please
check California's Department of Industrial Relations."

---

## 3. Query Reformulation Prompt

**Purpose:** Rewrites a casual, conversational user question into the kind of
statutory keyword phrasing the Labor Code actually uses, before it's embedded and
sent to the vector search. Improves retrieval recall — user phrasing and legal
phrasing often diverge a lot in this domain.

**Prompt:**

```
Rewrite the user's question as a short search query using formal Philippine labor
law terminology. Do not answer the question. Output only the rewritten query.

Examples:
"can they just kick me out with no warning" -> "termination without due process just cause"
"do I get extra for working sunday" -> "premium pay rest day work"

QUESTION:
{user_question}
```

**Example — how it prevents a retrieval miss (which otherwise causes hallucination
downstream):** Query *"do I get extra for working sunday"* shares almost no
vocabulary with Article 93's actual text ("premium pay," "rest day"). Embedding
the raw casual phrasing risks missing that article in top-k retrieval; if
retrieval misses the right article, the Grounded Answer Prompt has no correct
context to cite and is far more likely to either refuse unnecessarily or, in a
weaker prompt design, fall back on ungrounded outside knowledge. Reformulating
into statutory language first closes that vocabulary gap.

---

## 4. Self-Check / Citation-Verification Prompt

**Purpose:** A post-generation guard. Re-reads the draft answer against the same
retrieved context and checks that every claim is actually supported, catching
cases where the Grounded Answer Prompt's citation rule was followed loosely.

**Prompt:**

```
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
```

**Example — how it prevents hallucination from slipping through:** If the
Grounded Answer Prompt produces a claim like "you are entitled to 30 days notice"
without that figure appearing anywhere in the retrieved Article 283 text, the
self-check prompt flags `REVISE: the 30-day figure is not present in the
provided context` rather than letting an uncited number reach the user. This is
the same idea as a code reviewer catching an unverified claim — it costs one
extra API call per answer, but it's the cheapest lever available for reducing
hallucination in a citation-dependent domain like this one.

---

## Checkpoint 3 changes (implemented in [`src/rag/prompts.py`](../src/rag/prompts.py))

The four Checkpoint 2 prompts above are unchanged in `src/llm/prompts.py` and still run in
`src/llm/api_demo.py`. The Checkpoint 3 application uses a revised set, because it needs conversation
memory and the free Gemini tier allows only ~20 requests/day (the 4-prompt flow costs 5 calls per question).

| Checkpoint 2 prompt | Checkpoint 3 status |
|---|---|
| 2. Out-of-scope refusal / scope classifier | **Merged** with prompt 3 into one **router** prompt (one call, JSON output: `scope` + `search_query`) |
| 3. Query reformulation | **Merged** into the router. It now also sees recent chat history, so a follow-up like "in that case, how much would I get?" becomes the standalone query "separation pay closure of establishment" |
| 1. Grounded answer | **Extended**: adds a history slot plus a rule that earlier messages are only for understanding references, never a source of facts |
| 4. Self-check | **Reused unchanged**, but a failed check now triggers one regeneration (with the problem stated in a revision suffix) instead of only being logged |

New safeguard that needs no prompt: every "Article N" in a draft answer must appear in the retrieved
context, otherwise it is treated as a failed check before any LLM call is spent on it.

**Example — memory without hallucination:** after a turn about company closure, the user asks "how much
would I get?". Without the router rewrite, the retriever would receive a query with no subject and return
noise. With it, the retriever gets "separation pay closure of establishment". The answer prompt's rule 2
then stops the model from reusing a figure from the earlier answer unless the retrieved CONTEXT also
supports it.

**Known weakness:** the router fails open. If its JSON cannot be parsed, the question is treated as
in-scope and passed on, relying on the relevance threshold, the grounded prompt and verification to catch
an off-topic question.
