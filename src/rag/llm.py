"""
Checkpoint 3 - chat model factory
Philippine Labor Law Q&A Bot

Builds the LangChain chat model (Gemini). Replaces the hand-rolled retry loop
in src/llm/gemini_client.py (Checkpoint 2, left untouched) with LangChain's
built-in rate limiter + retries.
"""

import os

from langchain_core.rate_limiters import InMemoryRateLimiter

from src.rag import config


def make_llm(temperature: float = 0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

    limiter = InMemoryRateLimiter(
        requests_per_second=config.LLM_REQUESTS_PER_MINUTE / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=1,  # no bursts: free tier punishes them
    )
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=3,
        rate_limiter=limiter,
    )
