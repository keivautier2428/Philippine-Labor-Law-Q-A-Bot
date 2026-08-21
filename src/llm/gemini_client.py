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

from dotenv import load_dotenv
from google import genai

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

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


def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a single prompt to the Gemini Chat Completion API and return the
    text response. No conversation history is kept - each call is independent,
    matching how the four Checkpoint 2 prompts are used (classify, reformulate,
    answer, self-check are each a fresh, isolated call)."""
    client = get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()
