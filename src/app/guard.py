"""
Checkpoint 4 - public-demo protections
Philippine Labor Law Q&A Bot

A public URL plus a private API key means strangers can spend your quota. The
Gemini free tier is only ~20 requests/day, so without a guard one curious
visitor locks out your own defense. Three independent, env-configurable limits:

  APP_DAILY_CALL_BUDGET        total LLM calls the whole app may make per UTC day (default 15)
  APP_MAX_QUESTIONS_PER_SESSION questions one browser session may ask (default 10)
  APP_PASSCODE                 if set, visitors must enter it before chatting (default: open)

State is in process memory, so it resets when the container restarts. That is an
accepted limitation: the cost of a reset is a few extra calls, not a security hole.
"""

import hmac
import os
import threading
from datetime import datetime, timezone


class DailyBudget:
    def __init__(self, limit: int | None = None):
        self.limit = int(os.environ.get("APP_DAILY_CALL_BUDGET", "15")) if limit is None else limit
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day, self._used = self._today(), 0

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)

    def can_afford(self, calls: int = 3) -> bool:
        """3 = router + answer + self-check, the normal cost of one answered question."""
        return self.remaining() >= calls

    def consume(self, calls: int) -> None:
        with self._lock:
            self._roll()
            self._used += max(0, calls)


def passcode_ok(supplied: str, expected: str | None = None) -> bool:
    expected = os.environ.get("APP_PASSCODE", "") if expected is None else expected
    if not expected:
        return True
    return hmac.compare_digest(supplied.encode(), expected.encode())  # constant-time compare


def max_questions() -> int:
    return int(os.environ.get("APP_MAX_QUESTIONS_PER_SESSION", "10"))
