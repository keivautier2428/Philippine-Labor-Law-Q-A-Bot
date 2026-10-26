from src.ingestion import pipeline as pl
from src.rag import evaluate
from tests.fakes import HashingEmbeddings


def test_evaluate_runs_and_renders(tmp_path):
    vs = pl.get_vectorstore(HashingEmbeddings(), persist_directory=str(tmp_path / "db"), collection_name="eval_tests")
    pl.ingest(pl.discover([pl.config.RAW_DIR]), vectorstore=vs, log=lambda *_: None)
    res = evaluate.run(vs)
    assert res["n"] == len(evaluate.IN_CORPUS) and 0 <= res["hit3"] <= res["n"]
    md = evaluate.render(res)
    assert "hit@1" in md and "Threshold calibration" in md
    # every expected article label must actually exist in the corpus (guards against typos in the labels)
    present = {d["article"] for d in vs.get(include=["metadatas"])["metadatas"] if "article" in d}
    assert {a for _, a in evaluate.IN_CORPUS} <= present
