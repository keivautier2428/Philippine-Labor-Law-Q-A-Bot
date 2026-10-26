# Presentation Script: Philippine Labor Law Q&A Bot

Three presenters, one deck (`Labor-Law-QA-Bot-Presentation.pptx`), then a live demo on the deployed app.
**Fill in before presenting:** replace "Presenter A / B / C" on slide 1 and below with your real names.

| Block | Presenter | Slides | Time |
|---|---|---|---|
| Opening and big picture | **A** | 1-4 | ~4 min |
| How it works under the hood | **B** | 5-8 | ~5 min |
| Deployment, limits, next steps, demo hand-off | **C** | 9-12 | ~4 min |
| Live demo | **C drives, B narrates, A watches the clock and quota** | (app) | ~6-7 min |
| Questions | all | | remainder |

Total talk about 13 minutes plus demo. If you have less time, cut slides 5 and 11 to one sentence each.

**Two things to fill in before presenting (the deck and this script will otherwise be wrong):**
1. Slide 6: run `python -m src.rag.evaluate` and type the real hit@1 and hit@3 into the blanks.
2. Slide 10 / Presenter C's currency line: check the current official Labor Code numbering and say what you found.

---

## Presenter A

### Slide 1: Title (~0:30)
"Good [morning/afternoon]. We're [A], [B] and [C]. Our project is a Philippine Labor Law Q&A bot. Its one
promise is in the subtitle: it answers from the actual Labor Code and cites the article, or it tells you it
can't answer. We'll explain the problem, how we built it, how we deployed it, what its limits are, and then
we'll show it live."

### Slide 2: The problem (~1:00)
"If you're a worker asking 'can my employer fire me without a reason?', the answer is buried in long legal
prose. So people ask a general chatbot instead. The trouble is that a general chatbot will happily invent a
rule or even an article number that sounds right. In law, a confident wrong answer is worse than no answer.
So our design rule is simple: retrieve the real article, answer only from it, cite every claim, and decline
when the text doesn't support an answer."

### Slide 3: What the bot does (~1:00)
"Here's what that looks like. We indexed 33 Labor Code articles. A normal answered question takes three
model calls: one to check scope and understand the question, one to write the answer, and one to
check the answer against its sources. If that check fails, the bot retries once. And we have 30 automated
tests. We want to be upfront that 33 articles is a subset, not the whole Labor Code. The bot cites the
article for every claim, remembers the conversation for follow-ups, and for anything outside Philippine
labor law, or anything our data doesn't cover, it says so instead of guessing."

### Slide 4: Architecture (~1:30)
"The architecture has two parts. The top row happens once, offline: we take the documents, run them through
our ingestion pipeline, and store them in a vector database, Chroma. The bottom row happens on every
question: the user types in the chat interface, our LangGraph pipeline fetches the relevant chunks from
Chroma, and the hosted Gemini model writes and checks the answer. One detail that matters: the embedding
model, MiniLM, runs on our own machine, so searching costs no API quota. Only the answer-writing calls the
hosted model. [B] will now walk through the pieces."

---

## Presenter B

### Slide 5: Ingestion pipeline (~1:15)
"Ingestion is fully automated: parse, clean, chunk, embed, store. Two choices are worth mentioning. First, we
chunk by article, because an article is the natural unit of a citation. Short articles stay whole, and long
ones are split but every piece keeps its article number so citations stay correct. Second, it's safe to
re-run: each document gets a hash, so unchanged files are skipped and edited files replace their old
version. One bad file is reported but doesn't stop the batch. To add documents, you drop a file in
`data/incoming` and run one command."

### Slide 6: Retrieval we can measure (~1:15)
"For retrieval we use the MiniLM embedding model locally. Its vectors are unit length, so cosine similarity
and dot product rank identically, and we use cosine. A relevance threshold refuses weak matches before we
spend a single answer call. Because retrieval is local, we can evaluate it as often as we like with zero API
calls. We wrote 12 questions where we know which article should answer them. **[Say your real numbers:]**
the right article was first in [X] of 12 and in the top three in [Y] of 12. We also use that script to pick
the threshold instead of guessing it."

### Slide 7: Orchestration with LangGraph (~1:30)
"Here's the pipeline as a graph. **Route** decides whether the question is about Philippine labor law and
rewrites follow-ups into a standalone search. If it's out of scope, we refuse right there, after one call.
**Retrieve** fetches the chunks; if even the best match is too weak, we say it's not covered. **Generate**
writes the answer from those chunks only. **Verify** checks it. If verification fails, it loops back to
generate once, with the problem stated. We chose LangGraph because this flow has branches and a loop, and
that's exactly what its conditional edges are for. It also gives us conversation memory per session."

### Slide 8: Memory and verification (~1:15)
"Two features make it trustworthy. Memory: if you ask 'what if the company is closing down?' and then 'in
that case, how much separation pay?', the second question means nothing alone. The router uses recent
history to rewrite it into a standalone query before searching. Earlier answers are used only to understand
what 'that' refers to, never as a source of legal facts. Verification: first, every article the answer cites
must actually be in the retrieved text, and that check costs no API call. Second, a model self-check compares
the draft to its sources. If it still fails after one retry, the user sees a visible warning instead of a
confident wrong answer. [C] will cover deployment."

---

## Presenter C

### Slide 9: Interface and deployment (~1:15)
"The interface is a Streamlit chat app that also shows the retrieved articles and which ones were cited. We
package it with a two-stage Docker build: the first stage installs dependencies, downloads the model and
builds the index; the second is a slim image that runs as a non-root user. GitHub Actions runs the tests,
builds the image and smoke-tests it, then deploys to a Hugging Face Space. The API key is a run-time secret,
never in the code, the repo or the image. And because a public link could burn our private quota, the app
caps the total AI calls per day and the questions per visitor, and can require a passcode."

### Slide 10: Limits we are upfront about (~1:15)
"We want to name the limits ourselves. Coverage: 33 articles only, so something like the minimum working age
is answered as 'not covered'. Currency: our text uses the original article numbering and wording, so it has to
be checked against the current official Labor Code. **[Say what you found when you checked it.]** It's not
legal advice, and every answer carries that disclaimer. And we run on free-tier limits, with conversation
memory held in process memory, so it resets when the app restarts."

### Slide 11: What we would do next (~0:45)
"Next we'd grow and refresh the corpus, including current numbering. If our evaluation shows retrieval misses,
we'd tune the embedding model. To be clear, that is a plan; we did not run any fine-tuning. And we'd harden
it for real users with persistent conversations and per-user rate limits."

### Slide 12: Live demo (~0:30)
"Now the live demo. We'll ask five questions in one conversation. Watch three things: the sources panel
under each answer, how the follow-up questions get understood, and what happens with the out-of-scope
question." *(Switch to the deployed app. C types. B narrates. A keeps time.)*

---

## Live demo run sheet

**Before (do this early, not on the stage):**
- Open the Space URL about 10 minutes before so a sleeping free Space wakes up. Loading the page makes no AI calls.
- **Do not test questions on the demo day.** The free Gemini tier is about 20 requests per day. The demo uses
  13 (four answered questions at 3 calls, plus one refusal at 1). Test on a different day.
- Set the Space variable `APP_DAILY_CALL_BUDGET=18` (the default 15 covers exactly the five questions and leaves
  no room for an audience question).
- Keep a backup open: the same app running locally (`streamlit run src/app/streamlit_app.py`) and
  `docs/rag_demo_report.md` from an earlier full run.

**Pace:** the free tier allows about 5 requests per minute, and the bot paces itself, so each answered
question takes roughly 30-40 seconds. **That is normal.** B talks through what's happening while it waits
(filler below).

| # | C types | What B points at | If it goes differently |
|---|---|---|---|
| 1 | Can my employer fire me without giving a reason? | A plain-language answer with article citations; open "Retrieved articles" to show which were cited | Read what it actually cited. Don't claim a specific article it didn't cite. |
| 2 | What if the company is closing down instead? | The bot understood "company closing down" as a termination question without re-asking | Same. |
| 3 | In that case, how much separation pay? | "In that case" only works because of memory; the answer should cite the closure/separation-pay article(s) | If it says not covered, say so honestly: that is the guardrail working. |
| 4 | Can a 16 year old legally be hired? | In scope, but our 33 articles don't include that provision, so it should say it can't find it rather than guess | If it gives a confident answer, check its citations in the panel and say if they don't support it. |
| 5 | What's the minimum wage in California? | Refused quickly: out of scope, one call, no search | |

**Waiting-time filler for B (pick one per wait, don't read them all):**
- "Right now the router is deciding if this is a labor law question and rewriting it as a search query."
- "Next it searches our local index, then the model writes the answer from only those chunks."
- "Then the verification step checks each cited article against the retrieved text. This pacing is the free-tier limit; with a paid key it's near instant."

**If something breaks:**
- *App slow or Space asleep:* say "free-tier hosting", switch to the local copy.
- *Quota message in the app:* "That's our own daily-quota guard, working as designed." Then show `docs/rag_demo_report.md`.
- *A wrong or odd answer:* don't hide it. Show the sources panel, say what the verification step should have caught, and move on. Honest handling scores better than a rehearsed-looking demo.

**Close (A):** "To recap: the bot answers from real Labor Code text, cites it, remembers the conversation,
checks itself, and refuses when it should. Thank you. We're happy to take questions."

---

## Q&A prep: who answers what

| Likely question | Answer | Who |
|---|---|---|
| Why only 33 articles? | Scope for a semester project; the ingestion pipeline is built so adding documents is one command. | A |
| Are the article numbers current? | State what you checked. The corpus uses the original numbering. | A |
| Why RAG instead of fine-tuning? | Our known weaknesses are missing and outdated data, which fine-tuning doesn't fix; RAG gives citations and updates by re-ingesting. We did not run fine-tuning. | C |
| How do you stop hallucinations? | Grounded prompt, cited articles must appear in the retrieved text, a self-check, one retry, a visible warning if still unverified, a refusal when nothing relevant is found. It reduces but doesn't eliminate them. | B |
| Why LangGraph? | Branches (refuse vs answer) and a loop (regenerate) map to graph edges, and it provides conversation memory. | B |
| How is the API key protected? | Run-time secret; not in the repo or image; daily call budget, per-visitor limit, optional passcode. | C |
| What happens with many users? | In-memory state and free-tier quotas are the limits; next steps are persistent storage and per-user rate limits. | C |
| How did you measure quality? | 12 labelled retrieval questions plus a 5-question live conversation. Too small for statistical claims. | B |
