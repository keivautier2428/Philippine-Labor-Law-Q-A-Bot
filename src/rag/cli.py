"""
Checkpoint 3 - interactive RAG application (CLI)
Philippine Labor Law Q&A Bot

Multi-turn chat in the terminal. Memory lasts for the session.

    python -m src.rag.cli                 # full pipeline (3 LLM calls / question)
    python -m src.rag.cli --no-self-check # cheaper (2 calls / question)
    python -m src.rag.cli --debug         # also show router query + scores

Commands:  /reset  clear memory     /sources  last answer's retrieved articles     /quit
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_app(self_check: bool = True):
    from src.ingestion.pipeline import get_vectorstore
    from src.rag.chain import LaborLawRAG
    from src.rag.llm import make_llm

    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise SystemExit("The vector store is empty. Run:  python -m src.ingestion.pipeline")
    return LaborLawRAG(make_llm(), vs, self_check=self_check)


def print_sources(resp) -> None:
    if not resp.sources:
        return
    print("  Sources retrieved:")
    for s in resp.sources:
        mark = "*" if s["cited"] else " "
        print(f"   {mark} Article {s['article']}  (similarity {s['score']})")
    print("  (* = cited in the answer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rag = build_app(self_check=not args.no_self_check)
    print("Philippine Labor Law Q&A Bot  (type /quit to exit, /reset to clear memory)\n")
    last = None
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in {"/quit", "/exit"}:
            break
        if q == "/reset":
            rag.reset()
            print("Bot: Memory cleared.\n")
            continue
        if q == "/sources":
            if last:
                print_sources(last)
            continue
        last = rag.ask(q)
        print(f"\nBot: {last.answer}\n")
        if args.debug:
            print(f"  [status={last.status} query={last.search_query!r} verified={last.verified} "
                  f"revisions={last.revisions} llm_calls={last.llm_calls} top={last.top_score}]")
        print_sources(last)
        print()


if __name__ == "__main__":
    main()
