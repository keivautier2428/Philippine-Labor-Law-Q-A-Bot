import os

import pytest

from src.ingestion import pipeline as pl
from tests.fakes import HashingEmbeddings


@pytest.fixture
def vs(tmp_path):
    return pl.get_vectorstore(HashingEmbeddings(), persist_directory=str(tmp_path / "db"), collection_name="test_collection")


def raw(name):
    return os.path.join(pl.ROOT, "data", "raw", name)


def test_parse_extracts_article_and_hierarchy():
    d = pl.parse_file(raw("labor_code_art_93.txt"))
    assert d.metadata["article"] == 93
    assert d.metadata["book"].startswith("Book Three")
    assert "WEEKLY REST" in d.metadata["chapter"].upper()
    assert d.text.startswith("Article 93.")
    assert "====" not in d.text  # header stripped (Checkpoint 1 logic reused)


def test_short_article_is_one_chunk_long_article_is_split_and_labelled():
    short = pl.chunk_document(pl.parse_file(raw("labor_code_art_282.txt")))
    assert len(short) == 1
    long_ = pl.chunk_document(pl.parse_file(raw("labor_code_art_15.txt")))  # ~2.4k chars
    assert len(long_) > 1
    assert all(c.metadata["article"] == 15 for c in long_)
    assert long_[1].page_content.startswith("(Article 15, continued)")
    assert all(len(c.page_content) <= pl.config.CHUNK_SIZE + 60 for c in long_)


def test_ingest_is_idempotent_and_detects_edits(vs, tmp_path):
    f = tmp_path / "labor_code_art_999.txt"
    f.write_text("Article 999. Test rule. An employer shall pay a bonus of ten (10) pesos.", encoding="utf-8")
    s1 = pl.ingest([str(f)], vectorstore=vs, log=lambda *_: None)
    assert len(s1.added) == 1 and not s1.skipped
    s2 = pl.ingest([str(f)], vectorstore=vs, log=lambda *_: None)
    assert len(s2.skipped) == 1 and s2.chunks_written == 0           # re-run is a no-op
    f.write_text("Article 999. Test rule. An employer shall pay a bonus of twenty (20) pesos.", encoding="utf-8")
    s3 = pl.ingest([str(f)], vectorstore=vs, log=lambda *_: None)
    assert len(s3.updated) == 1
    got = vs.get(where={"article": 999})
    assert len(got["ids"]) == 1 and "twenty" in got["documents"][0]  # old version replaced, not duplicated


def test_bad_file_does_not_abort_the_batch(vs, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    good = raw("labor_code_art_94.txt")
    s = pl.ingest([str(empty), good], vectorstore=vs, log=lambda *_: None)
    assert len(s.failed) == 1 and len(s.added) == 1


def test_dry_run_writes_nothing(vs):
    s = pl.ingest([raw("labor_code_art_94.txt")], vectorstore=vs, dry_run=True, log=lambda *_: None)
    assert len(s.added) == 1
    assert vs._collection.count() == 0


def test_full_corpus_ingests_and_discover_skips_underscore_files(vs):
    paths = pl.discover([pl.config.RAW_DIR])
    assert len(paths) == 33
    assert not any(os.path.basename(p).startswith("_") for p in paths)
    s = pl.ingest(paths, vectorstore=vs, log=lambda *_: None)
    assert not s.failed and len(s.added) == 33
    assert vs._collection.count() >= 33


def test_pdf_parsing(tmp_path):
    from pypdf import PdfWriter  # only to make a blank PDF; ensures the code path fails cleanly, not crashes the batch
    p = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(200, 200)
    with open(p, "wb") as fh:
        w.write(fh)
    with pytest.raises(ValueError):
        pl.parse_file(str(p))  # no text layer -> reported as empty, not silently indexed
