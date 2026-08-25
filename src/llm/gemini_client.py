"""
Checkpoint 2 - API Integration
Philippine Labor Law Q&A Bot

Thin wrapper around the Gemini Chat Completion API (google-genai SDK). Reads
the API key from the GEMINI_API_KEY environment variable / .env file - never
hardcode it here.

Setup:
    1. Copy .env.example to .env
    2. Put your key in .env:  GEMINI_API_KEY=your-key-here
    3. pip install -r requirements.txt
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _retry_delay_seconds(err: genai_errors.APIError, default: float = 20.0) -> float:
    """Gemini's 429 response includes a suggested RetryInfo.retryDelay (e.g.
    "16s") - use it when present instead of guessing a wait time."""
    try:
        for detail in err.details.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay = detail.get("retryDelay", "")
                if delay.endswith("s"):
                    return float(delay[:-1])
    except Exception:
        pass
    return default


def generate(prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 4) -> str:
    """Send a single prompt to the Gemini Chat Completion API and return the
    text response. No conversation history is kept - each call is independent,
    matching how the four Checkpoint 2 prompts are used (classify, reformulate,
    answer, self-check are each a fresh, isolated call).

    Retries on 429 (rate limit) using the API's own suggested wait time - the
    free tier caps at 5 requests/minute, and a single query in api_demo.py
    makes 4 back-to-back calls, so this is expected to trigger occasionally
    rather than being an error condition."""
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = _retry_delay_seconds(e) + 2  # small buffer past the suggested delay
                print(f"  (Gemini free-tier rate limit hit - waiting {wait:.0f}s before retrying...)")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")  # loop always returns or raises
