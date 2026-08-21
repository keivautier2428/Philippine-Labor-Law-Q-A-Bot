"""
Checkpoint 1 - Text Preprocessing
Philippine Labor Law Q&A Bot

Cleans and tokenizes the raw Labor Code article files in data/raw/ and writes
the processed output to data/processed/. Also writes a before/after report to
docs/preprocessing_report.md as evidence for the checkpoint submission.

Usage:
    python src/preprocessing/preprocess.py
"""

import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
REPORT_PATH = os.path.join(ROOT, "docs", "preprocessing_report.md")

# Files whose metadata header block we strip before the "====" separator.
HEADER_SEPARATOR = "=" * 60

# A minimal legal/English stopword list. Kept short and domain-appropriate
# rather than pulling in a large generic list, since legal text relies on
# small function words ("shall", "any") that we deliberately keep.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "and",
    "or", "is", "are", "was", "were", "be", "been", "being", "as", "it",
    "this", "that", "these", "those", "with", "from",
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*|\d+(?:\.\d+)?%?")


def strip_header(raw_text: str) -> str:
    """Remove the manifest-style metadata block written by the data
    collection step, keeping only the legal article text itself."""
    idx = raw_text.find(HEADER_SEPARATOR)
    if idx == -1:
        return raw_text.strip()
    return raw_text[idx + len(HEADER_SEPARATOR):].strip()


def clean_text(text: str) -> str:
    """Normalize whitespace and strip characters that don't carry meaning
    for embedding/retrieval (stray HTML entities the source left behind,
    repeated punctuation, non-breaking spaces)."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer with stopword removal, tuned for
    legal text (keeps hyphenated terms like 'thirteenth-month' and
    percentages like '10%' intact instead of splitting them)."""
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def process_file(filename: str) -> dict:
    raw_path = os.path.join(RAW_DIR, filename)
    with open(raw_path, encoding="utf-8") as f:
        raw_text = f.read()

    body = strip_header(raw_text)
    cleaned = clean_text(body)
    tokens = tokenize(cleaned)

    return {
        "source_file": filename,
        "raw_char_count": len(raw_text),
        "cleaned_char_count": len(cleaned),
        "token_count": len(tokens),
        "cleaned_text": cleaned,
        "tokens": tokens,
    }


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    raw_files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".txt") and not f.startswith("_")
    )

    if not raw_files:
        raise SystemExit(f"No .txt files found in {RAW_DIR}")

    results = []
    for filename in raw_files:
        result = process_file(filename)
        results.append(result)

        out_name = os.path.splitext(filename)[0] + ".json"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write one combined corpus file — this is what the embedding step
    # consumes so it doesn't need to re-walk data/processed/*.json itself.
    corpus_path = os.path.join(PROCESSED_DIR, "_corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "source_file": r["source_file"],
                "text": r["cleaned_text"],
            }, ensure_ascii=False) + "\n")

    # Before/after report for 5 sample documents, spread across the corpus.
    sample_indices = sorted(set([0, len(results) // 3, 2 * len(results) // 3, len(results) - 1]))
    lines = [
        "# Preprocessing Report — Before / After Examples",
        "",
        f"Processed **{len(results)}** documents from `data/raw/` into `data/processed/`.",
        "",
        "Pipeline steps applied to each document:",
        "1. Strip the manifest-style metadata header (source citation block).",
        "2. Normalize whitespace, collapse HTML entity leftovers (`&nbsp;`), fix spacing before punctuation.",
        "3. Tokenize into lowercase word/number tokens, preserving hyphenated legal terms (e.g. `thirteenth-month`) and removing a small stopword list.",
        "",
        "## Corpus Statistics",
        "",
        f"- Total documents: {len(results)}",
        f"- Total tokens across corpus: {sum(r['token_count'] for r in results)}",
        f"- Average tokens per document: {sum(r['token_count'] for r in results) / len(results):.1f}",
        "",
        "## Sample Before/After",
        "",
    ]

    for i in sample_indices:
        r = results[i]
        raw_path = os.path.join(RAW_DIR, r["source_file"])
        with open(raw_path, encoding="utf-8") as f:
            raw_full = f.read()
        before_snippet = raw_full.strip().replace("\n", " ")[:400]
        after_snippet = r["cleaned_text"][:400]
        token_preview = r["tokens"][:20]

        lines.append(f"### {r['source_file']}")
        lines.append("")
        lines.append(f"**Before (raw, {r['raw_char_count']} chars):**")
        lines.append("```")
        lines.append(before_snippet + ("..." if len(before_snippet) == 400 else ""))
        lines.append("```")
        lines.append("")
        lines.append(f"**After (cleaned, {r['cleaned_char_count']} chars):**")
        lines.append("```")
        lines.append(after_snippet + ("..." if len(after_snippet) == 400 else ""))
        lines.append("```")
        lines.append("")
        lines.append(f"**Tokens ({r['token_count']} total, first 20 shown):** `{token_preview}`")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Processed {len(results)} documents.")
    print(f"  -> {PROCESSED_DIR} (per-file JSON + _corpus.jsonl)")
    print(f"  -> {REPORT_PATH} (before/after report)")


if __name__ == "__main__":
    main()
