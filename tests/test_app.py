import os

import pytest
from streamlit.testing.v1 import AppTest

from src.app import backend, guard
from src.ingestion import pipeline as pl
from tests.fakes import HashingEmbeddings, ScriptedLLM

APP = os.path.join(os.path.dirname(__file__), "..", "src", "app", "streamlit_app.py")


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    import streamlit as st

    vs = pl.get_vectorstore(HashingEmbeddings(), persist_directory=str(tmp_path / "db"), collection_name="app_tests")
    pl.ingest(pl.discover([pl.config.RAW_DIR]), vectorstore=vs, log=lambda *_: None)
    llm = ScriptedLLM(answers=["Per (Article 282) an employer needs a just cause."])
    monkeypatch.setattr(backend, "load_backend", lambda: (vs, llm.runnable()))
    monkeypatch.setattr("src.rag.chain.config.MIN_RELEVANCE", 0.0)
    st.cache_resource.clear()
    return llm


def test_chat_roundtrip_and_budget_accounting(fake_backend, monkeypatch):
    monkeypatch.setenv("APP_DAILY_CALL_BUDGET", "15")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("termination by employer just cause serious misconduct").run()
    assert not at.exception
    texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert any("Article 282" in t for t in texts)
    assert fake_backend.kinds().count("router") == 1


def test_daily_budget_exhausted_blocks_llm_calls(fake_backend, monkeypatch):
    monkeypatch.setenv("APP_DAILY_CALL_BUDGET", "2")  # less than one normal question (3)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("termination just cause").run()
    assert fake_backend.calls == []                    # the model was never called
    assert any("quota" in m.markdown[0].value.lower() for m in at.chat_message if m.markdown)


def test_passcode_gate(fake_backend, monkeypatch):
    monkeypatch.setenv("APP_PASSCODE", "s3cret")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert len(at.chat_input) == 0                     # chat hidden until the passcode is entered
    at.text_input[0].set_value("wrong").run()
    assert len(at.chat_input) == 0 and at.error
    at.text_input[0].set_value("s3cret").run()
    assert len(at.chat_input) == 1


def test_guard_units(monkeypatch):
    b = guard.DailyBudget(limit=5)
    assert b.can_afford(3)
    b.consume(3)
    assert not b.can_afford(3) and b.remaining() == 2
    assert guard.passcode_ok("anything", expected="") is True
    assert guard.passcode_ok("x", expected="y") is False and guard.passcode_ok("y", expected="y") is True
