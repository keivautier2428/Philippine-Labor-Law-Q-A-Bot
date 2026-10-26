"""Offline test doubles: no network, no Gemini key, no model download."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "can", "do", "i", "my", "me", "if", "be", "what", "how"}


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedding, L2-normalised. Related
    texts share words so similarity is meaningful enough to test retrieval
    plumbing. It is NOT a substitute for MiniLM when judging retrieval quality."""

    DIM = 512

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in STOP:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ScriptedLLM:
    """Stands in for the chat model. Decides what to return by looking at which
    prompt it was handed (router / answer / self-check), and records every call."""

    def __init__(self, router=None, answers=None, checks=None):
        self.router = router or (lambda q: '{"scope": "IN_SCOPE", "search_query": "%s"}' % q)
        self.answers = list(answers or ["Per (Article 282), just causes exist."])
        self.checks = list(checks or ["OK"])
        self.calls = []        # (kind, prompt_text)

    def _text(self, prompt_value):
        return "\n".join(m.content for m in prompt_value.to_messages())

    def __call__(self, prompt_value):
        text = self._text(prompt_value)
        last_human = prompt_value.to_messages()[-1].content
        if "routing step" in text:
            self.calls.append(("router", text))
            return AIMessage(content=self.router(last_human))
        if "DRAFT ANSWER" in text:
            self.calls.append(("check", text))
            return AIMessage(content=self.checks.pop(0) if len(self.checks) > 1 else self.checks[0])
        self.calls.append(("answer", text))
        return AIMessage(content=self.answers.pop(0) if len(self.answers) > 1 else self.answers[0])

    def runnable(self):
        return RunnableLambda(self)

    def kinds(self):
        return [k for k, _ in self.calls]
