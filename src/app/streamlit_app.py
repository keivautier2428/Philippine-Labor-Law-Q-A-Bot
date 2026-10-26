"""
Checkpoint 4 - web interface (Streamlit)
Philippine Labor Law Q&A Bot

Wraps the Checkpoint 3 RAG graph (src/rag/chain.py) in a chat UI. All RAG logic
stays in src/rag; this file only does presentation, session handling and the
public-demo guard.

    streamlit run src/app/streamlit_app.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

from src.app import backend, guard  # noqa: E402
from src.rag.chain import LaborLawRAG  # noqa: E402

EXAMPLES = [
    "Can my employer fire me without giving a reason?",
    "Do I get extra pay if I work on my rest day?",
    "How much separation pay if the company closes down?",
    "How long can a probationary period last?",
]

st.set_page_config(page_title="Philippine Labor Law Q&A", page_icon="⚖️")


@st.cache_resource(show_spinner="Loading the Labor Code index...")
def _backend():
    return backend.load_backend()      # shared across sessions: embedding model, Chroma, LLM + rate limiter


@st.cache_resource
def _budget() -> guard.DailyBudget:
    return guard.DailyBudget()          # one budget for the whole app process


def _rag() -> LaborLawRAG:
    """One RAG object (so one conversation memory) per browser session."""
    if "rag" not in st.session_state:
        vs, llm = _backend()
        st.session_state.rag = LaborLawRAG(llm, vs)
    return st.session_state.rag


def _gate() -> None:
    if not os.environ.get("APP_PASSCODE"):
        return
    if st.session_state.get("authed"):
        return
    code = st.text_input("Passcode", type="password")
    if code and guard.passcode_ok(code):
        st.session_state.authed = True
        st.rerun()
    elif code:
        st.error("Wrong passcode.")
    st.stop()


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Retrieved articles"):
        for s in sources:
            mark = "✅ cited" if s["cited"] else "retrieved"
            st.write(f"Article {s['article']} · similarity {s['score']} · {mark}")


def _answer(question: str) -> None:
    budget = _budget()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if st.session_state.asked >= guard.max_questions():
            text, resp = "Session question limit reached. Please reload the page to start a new session.", None
        elif not budget.can_afford(3):
            text, resp = "The demo's daily AI quota is used up. Please try again tomorrow.", None
        else:
            with st.spinner("Searching the Labor Code..."):
                resp = _rag().ask(question, session_id="web")
            st.session_state.asked += 1
            budget.consume(max(resp.llm_calls, 1))
            text = resp.answer
        st.write(text)
        if resp is not None:
            _render_sources(resp.sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": resp.sources if resp else []}
    )


def main() -> None:
    st.title("⚖️ Philippine Labor Law Q&A")
    st.caption("Answers are drawn only from the Labor Code articles loaded in this bot, with citations. "
               "Not legal advice.")
    _gate()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("asked", 0)

    with st.sidebar:
        st.subheader("Try a question")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        if st.button("Clear conversation"):
            if "rag" in st.session_state:
                st.session_state.rag.reset("web")
            st.session_state.messages, st.session_state.asked = [], 0
            st.rerun()
        st.caption(f"Demo AI calls left today: {_budget().remaining()}")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            _render_sources(m.get("sources", []))

    question = st.chat_input("Ask about Philippine labor law...") or st.session_state.pop("pending", None)
    if question:
        _answer(question)
        st.rerun()


main()
