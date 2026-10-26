import pytest

from src.ingestion import pipeline as pl
from src.rag.chain import LaborLawRAG, cited_articles, parse_router
from tests.fakes import HashingEmbeddings, ScriptedLLM


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    vs = pl.get_vectorstore(HashingEmbeddings(), persist_directory=str(tmp_path_factory.mktemp("db")), collection_name="chain_tests")
    pl.ingest(pl.discover([pl.config.RAW_DIR]), vectorstore=vs, log=lambda *_: None)
    return vs


def make(store, llm, **kw):
    kw.setdefault("min_relevance", 0.05)
    return LaborLawRAG(llm.runnable(), store, **kw)


# ---- router parsing ---------------------------------------------------------
@pytest.mark.parametrize(
    "raw,scope",
    [
        ('{"scope": "IN_SCOPE", "search_query": "x"}', "IN_SCOPE"),
        ('```json\n{"scope": "OUT_OF_SCOPE", "search_query": ""}\n```', "OUT_OF_SCOPE"),
        ('Sure! {"scope": "chitchat", "search_query": ""} hope that helps', "CHITCHAT"),
    ],
)
def test_parse_router_tolerates_fences_and_chatter(raw, scope):
    assert parse_router(raw, "q")["scope"] == scope


def test_parse_router_fails_open_on_garbage():
    r = parse_router("I cannot comply", "my question")
    assert r["scope"] == "IN_SCOPE" and r["search_query"] == "my question" and r["router_parse_failed"]


def test_cited_articles():
    assert cited_articles("(Article 282) and Art. 93, Arts. 87") == {282, 93, 87}


# ---- refusal paths ------------------------------------------------------------
def test_out_of_scope_refused_with_one_call_and_no_retrieval(store):
    llm = ScriptedLLM(router=lambda q: '{"scope": "OUT_OF_SCOPE", "search_query": ""}')
    r = make(store, llm).ask("What's the minimum wage in California?")
    assert r.status == "refused_scope" and r.llm_calls == 1 and llm.kinds() == ["router"]
    assert "Philippine labor law" in r.answer


def test_low_relevance_refuses_without_calling_answer_model(store):
    llm = ScriptedLLM()
    r = make(store, llm, min_relevance=0.99).ask("termination just cause")
    assert r.status == "refused_no_context" and llm.kinds() == ["router"]


def test_chitchat(store):
    llm = ScriptedLLM(router=lambda q: '{"scope": "CHITCHAT", "search_query": ""}')
    assert make(store, llm).ask("hi!").status == "chitchat"


# ---- happy path + call budget --------------------------------------------------
def test_grounded_answer_makes_three_calls_and_carries_sources(store):
    llm = ScriptedLLM(answers=["An employer may terminate for serious misconduct (Article 282)."])
    r = make(store, llm, k=4).ask("termination by employer just cause serious misconduct")
    assert llm.kinds() == ["router", "answer", "check"] and r.llm_calls == 3
    assert r.status == "answered" and r.verified is True and r.revisions == 0
    assert any(s["article"] == 282 and s["cited"] for s in r.sources)
    assert "not legal advice" in r.answer


def test_no_self_check_saves_a_call(store):
    llm = ScriptedLLM(answers=["(Article 282) applies."])
    r = make(store, llm, self_check=False).ask("termination by employer just cause serious misconduct")
    assert llm.kinds() == ["router", "answer"] and r.verified is None and r.status == "answered"


# ---- verification loop ----------------------------------------------------------
def test_hallucinated_citation_is_caught_for_free_then_regenerated(store):
    llm = ScriptedLLM(answers=["Yes, see Article 9999.", "Per (Article 282) yes."], checks=["OK"])
    r = make(store, llm).ask("termination by employer just cause serious misconduct")
    assert r.revisions == 1 and r.verified is True and "9999" not in r.answer
    # the bad draft never reached the LLM self-check (deterministic check caught it first)
    assert llm.kinds() == ["router", "answer", "answer", "check"]
    assert "[9999]" in llm.calls[2][1]  # the revision prompt states what was wrong


def test_self_check_revise_triggers_one_regeneration(store):
    llm = ScriptedLLM(answers=["Draft one (Article 282).", "Draft two (Article 282)."], checks=["REVISE: claim X unsupported", "OK"])
    r = make(store, llm).ask("termination by employer just cause serious misconduct")
    assert r.revisions == 1 and r.verified is True and "Draft two" in r.answer
    assert "claim X unsupported" in llm.calls[3][1]  # calls: router, answer, check, REVISION answer


def test_still_failing_after_revision_is_flagged_not_hidden(store):
    llm = ScriptedLLM(answers=["(Article 282) a.", "(Article 282) b."], checks=["REVISE: bad"])
    r = make(store, llm, max_revisions=1).ask("termination by employer just cause serious misconduct")
    assert r.status == "answered_unverified" and r.verified is False
    assert r.answer.startswith("[Warning") and r.llm_calls == 5  # router, answer, check, answer, check


# ---- memory --------------------------------------------------------------------
def test_followup_router_receives_prior_turn_and_pronoun_is_resolved(store):
    seen = {}

    def router(q):
        seen["last"] = q
        if "separation pay" in q and "that case" in q:
            return '{"scope": "IN_SCOPE", "search_query": "separation pay closure of establishment"}'
        return '{"scope": "IN_SCOPE", "search_query": "closure of establishment termination"}'

    llm = ScriptedLLM(router=router, answers=["(Article 283) closure."])
    rag = make(store, llm, self_check=False)
    rag.ask("What if the company is closing down?", session_id="s1")
    r2 = rag.ask("In that case, how much separation pay would I get?", session_id="s1")

    router_prompts = [t for k, t in llm.calls if k == "router"]
    assert "closing down" in router_prompts[1]            # turn 1 was visible to the router on turn 2
    assert r2.search_query == "separation pay closure of establishment"
    assert len(rag.history("s1")) == 4                     # 2 exchanges stored by LangChain


def test_sessions_are_isolated_and_reset_clears(store):
    llm = ScriptedLLM(answers=["(Article 282) ok."])
    rag = make(store, llm, self_check=False)
    rag.ask("termination just cause", session_id="a")
    rag.ask("rest day pay", session_id="b")
    assert len(rag.history("a")) == 2 and len(rag.history("b")) == 2
    rag.reset("a")
    assert rag.history("a") == [] and len(rag.history("b")) == 2


def test_history_window_limits_what_model_sees_but_not_what_is_stored(store):
    llm = ScriptedLLM(answers=["(Article 282) ok."])
    rag = make(store, llm, self_check=False, history_window=1)
    for i in range(4):
        rag.ask(f"question number {i} about termination just cause", session_id="w")
    assert len(rag.history("w")) == 8
    last_router = [t for k, t in llm.calls if k == "router"][-1]
    assert "question number 2" in last_router and "question number 0" not in last_router


def test_disclaimer_is_not_replayed_into_history(store):
    llm = ScriptedLLM(answers=["(Article 282) ok."])
    rag = make(store, llm, self_check=False)
    r = rag.ask("termination just cause", session_id="d")
    assert "not legal advice" in r.answer
    assert all("not legal advice" not in m.content for m in rag.history("d"))


# ---- robustness -----------------------------------------------------------------
def test_llm_exception_becomes_friendly_error_not_crash(store):
    def boom(_):
        raise TimeoutError("network down")

    from langchain_core.runnables import RunnableLambda

    r = LaborLawRAG(RunnableLambda(boom), store).ask("termination just cause")
    assert r.status == "error" and "TimeoutError" in r.answer
