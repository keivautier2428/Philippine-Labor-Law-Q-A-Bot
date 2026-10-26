"""
Checkpoint 3 - RAG orchestration (LangChain + LangGraph)
Philippine Labor Law Q&A Bot

The pipeline is a small LangGraph state machine. Each node is a LangChain
runnable chain (prompt | chat model | parser); LangGraph supplies the routing
between nodes and the conversation memory.

    START -> route --(OUT_OF_SCOPE / CHITCHAT)-----------------------> finish
               |
               v IN_SCOPE + standalone, memory-aware search query
            retrieve --(best similarity < MIN_RELEVANCE)--------------> finish
               |
               v
            generate  <-----------------------------+
               |                                     | verification failed and
               v                                     | revisions left
            verify  ---------------------------------+
               |
               v
            finish -> END

verify = (a) free deterministic check: every cited "Article N" must appear in the
         retrieved context; (b) LLM self-check (Checkpoint 2's prompt).
         A failure loops back to generate once with the problem stated - this is
         the "self-check doesn't loop back" gap Checkpoint 2 listed as future work.

Memory: a LangGraph checkpointer (InMemorySaver) persists the message list per
thread_id (= session id). LangChain's older RunnableWithMessageHistory /
ConversationBufferMemory are deprecated in current releases, so they are not used.
Only the last HISTORY_WINDOW_TURNS exchanges are sent to the model.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.rag import config
from src.rag import prompts as P

CITATION_RE = re.compile(r"Art(?:icle|s?\.)\s*(\d+)", re.I)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_router(raw: str, question: str) -> dict:
    """Tolerant JSON parse of the router output. Gemini sometimes wraps JSON
    in code fences or adds a sentence. On total failure we FAIL OPEN to
    IN_SCOPE with the raw question as the query: the retrieval threshold,
    grounded prompt and verification still guard the answer."""
    m = re.search(r"\{.*\}", raw.strip(), re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            scope = str(data.get("scope", "")).upper().strip()
            if scope in {"IN_SCOPE", "OUT_OF_SCOPE", "CHITCHAT"}:
                return {"scope": scope, "search_query": str(data.get("search_query", "")).strip() or question}
        except json.JSONDecodeError:
            pass
    return {"scope": "IN_SCOPE", "search_query": question, "router_parse_failed": True}


def cited_articles(text: str) -> set[int]:
    return {int(n) for n in CITATION_RE.findall(text)}


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        label = f"Article {h['article']}" if h.get("article") is not None else (h.get("doc_title") or h.get("source") or "source")
        where = " > ".join(x for x in (h.get("book"), h.get("title_heading"), h.get("chapter")) if x)
        blocks.append(f"[{label}{' | ' + where if where else ''}]\n{h['text']}")
    return "\n\n".join(blocks)


@dataclass
class RagResponse:
    answer: str
    status: str                      # answered | answered_unverified | refused_scope | refused_no_context | chitchat | error
    search_query: str = ""
    sources: list[dict] = field(default_factory=list)   # {article, source, score, cited}
    verified: bool | None = None     # None = only the free citation check ran
    revisions: int = 0
    llm_calls: int = 0
    top_score: float | None = None


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]   # persisted conversation (memory)
    # --- per-turn working fields (all plain JSON types so the checkpointer can store them) ---
    question: str
    scope: str
    query: str
    hits: list[dict]
    top: float | None
    answer: str
    problem: str | None
    verified: bool | None
    revisions: int
    calls: int
    status: str
    final: str


# --------------------------------------------------------------------------
# the orchestrator
# --------------------------------------------------------------------------
class LaborLawRAG:
    def __init__(
        self,
        llm,
        vectorstore,
        k: int = config.TOP_K,
        min_relevance: float = config.MIN_RELEVANCE,
        self_check: bool = True,
        max_revisions: int = config.MAX_REVISIONS,
        history_window: int = config.HISTORY_WINDOW_TURNS,
    ):
        self.vs = vectorstore
        self.k = k
        self.min_relevance = min_relevance
        self.self_check = self_check
        self.max_revisions = max_revisions
        self.history_window = history_window

        self._router = P.ROUTER_PROMPT | llm | StrOutputParser()
        self._answerer = P.ANSWER_PROMPT | llm | StrOutputParser()
        self._checker = P.SELF_CHECK | llm | StrOutputParser()

        self._saver = InMemorySaver()
        self._graph = self._build_graph().compile(checkpointer=self._saver)

    # ---- graph ----------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("route", self._route)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_node("finish", self._finish)

        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: "retrieve" if s["scope"] == "IN_SCOPE" else "finish", ["retrieve", "finish"])
        g.add_conditional_edges("retrieve", lambda s: "finish" if s.get("status") == "refused_no_context" else "generate", ["generate", "finish"])
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", lambda s: "generate" if s.get("problem") else "finish", ["generate", "finish"])
        g.add_edge("finish", END)
        return g

    def _recent(self, state: State) -> list[BaseMessage]:
        msgs = state.get("messages", [])
        return msgs[-2 * self.history_window:] if self.history_window else []

    def _route(self, state: State) -> dict:
        q = state["question"]
        route = parse_router(self._router.invoke({"question": q, "history": self._recent(state)}), q)
        out = {"scope": route["scope"], "query": route["search_query"], "calls": 1, "revisions": 0,
               "problem": None, "verified": None, "hits": [], "top": None, "answer": "", "status": ""}
        if route["scope"] == "OUT_OF_SCOPE":
            out |= {"status": "refused_scope", "answer": P.REFUSAL_OUT_OF_SCOPE}
        elif route["scope"] == "CHITCHAT":
            out |= {"status": "chitchat", "answer": P.CHITCHAT_REPLY}
        return out

    def retrieve(self, query: str) -> list[dict]:
        """Top-k chunks, best first, as plain dicts. Chroma is configured for
        cosine, whose 'distance' is 1 - similarity (see distance_metric_report)."""
        raw = self.vs.similarity_search_with_score(query, k=self.k)
        hits = []
        for d, dist in raw:
            m = d.metadata
            hits.append({
                "text": d.page_content, "score": round(1.0 - float(dist), 4),
                "article": m.get("article"), "source": m.get("source"), "doc_title": m.get("doc_title"),
                "book": m.get("book"), "title_heading": m.get("title_heading"), "chapter": m.get("chapter"),
            })
        return sorted(hits, key=lambda h: h["score"], reverse=True)

    def _retrieve_node(self, state: State) -> dict:
        hits = self.retrieve(state["query"])
        top = hits[0]["score"] if hits else None
        if not hits or top < self.min_relevance:
            return {"hits": hits, "top": top, "status": "refused_no_context", "answer": P.REFUSAL_NO_CONTEXT}
        return {"hits": hits, "top": top}

    def _generate(self, state: State) -> dict:
        question = state["question"]
        revisions = state.get("revisions", 0)
        if state.get("problem"):
            question += P.REVISION_SUFFIX.format(problem=state["problem"])
            revisions += 1
        answer = self._answerer.invoke(
            {"context": format_context(state["hits"]), "question": question, "history": self._recent(state)}
        )
        return {"answer": answer.strip(), "revisions": revisions, "calls": state["calls"] + 1}

    def _verify(self, state: State) -> dict:
        calls = state["calls"]
        allowed = {h["article"] for h in state["hits"] if h.get("article") is not None}
        problem = None
        stray = cited_articles(state["answer"]) - allowed
        if stray:  # free deterministic check, no API call
            problem = f"it cites Article(s) {sorted(stray)} which are not in the CONTEXT"
        elif self.self_check:
            verdict = self._checker.invoke(
                {"retrieved_context": format_context(state["hits"]), "draft_answer": state["answer"]}
            ).strip()
            calls += 1
            if not verdict.upper().startswith("OK"):
                problem = verdict
        if problem is None:
            return {"problem": None, "verified": True if self.self_check else None, "calls": calls}
        out_of_revisions = state.get("revisions", 0) >= self.max_revisions
        # problem is cleared when we give up, so the router edge exits to 'finish'
        return {"problem": None if out_of_revisions else problem, "verified": False, "calls": calls}

    def _finish(self, state: State) -> dict:
        status = state.get("status") or ("answered" if state.get("verified") is not False else "answered_unverified")
        body = state["answer"]
        if status == "answered_unverified":
            body = ("[Warning: could not fully verify this answer against the source text - "
                    "please double-check the cited articles.]\n\n" + body)
        return {
            "status": status,
            "final": body,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=body)],
        }

    # ---- memory ---------------------------------------------------------
    def history(self, session_id: str = "default") -> list[BaseMessage]:
        snap = self._graph.get_state({"configurable": {"thread_id": session_id}})
        return list(snap.values.get("messages", [])) if snap and snap.values else []

    def reset(self, session_id: str = "default") -> None:
        self._saver.delete_thread(session_id)

    # ---- public API -----------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> RagResponse:
        cfg = {"configurable": {"thread_id": session_id}}
        try:
            s = self._graph.invoke({"question": question}, cfg)
        except Exception as e:  # noqa: BLE001 - API/network failures must not crash a chat session
            return RagResponse(
                f"Sorry - something went wrong while answering ({type(e).__name__}). Please try again in a moment.",
                "error",
            )
        cites = cited_articles(s["final"])
        sources = [
            {"article": h["article"], "source": h["source"], "score": h["score"], "cited": h["article"] in cites}
            for h in s.get("hits", [])
        ]
        answer = s["final"]
        if s["status"].startswith("answered"):
            answer += f"\n\n({config.DISCLAIMER})"  # display-only; never stored in history
        return RagResponse(
            answer=answer, status=s["status"], search_query=s.get("query", "") if s["scope"] == "IN_SCOPE" else "",
            sources=sources, verified=s.get("verified"), revisions=s.get("revisions", 0),
            llm_calls=s.get("calls", 0), top_score=s.get("top"),
        )
