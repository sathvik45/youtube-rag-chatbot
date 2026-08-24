"""A DeepEval judge backed by Groq, paced to a token budget.

Why this exists: `FaithfulnessMetric` and `AnswerRelevancyMetric` are LLM-judge
metrics, and DeepEval's default judge is OpenAI. This project has one provider
and one key, so the judge is wrapped rather than a second provider added.

The judge is NOT the generator. `GRADER_MODEL` is deliberately a different
checkpoint from `LLM_MODEL`: a model grading its own output rates it generously
(self-preference bias). Note the trade-off in the direction we picked --
gpt-oss-20b is a *smaller* model than the gpt-oss-120b generator, so it buys
independence at the cost of a noisier, more agreeable judge.

Contract, verified against deepeval 4.1.x:

    metric -> model.generate_with_schema(prompt, schema=SomePydanticModel)
           -> base class calls our generate(prompt, schema=...)
           -> we must return EITHER an instance of that model
              OR a JSON string deepeval can parse itself

Structured output modes
-----------------------
Not every Groq checkpoint supports every way of getting JSON out, and which one
works is a property of the MODEL, not of the call:

    tools  with_structured_output(schema)                      needs tool calling
    json   with_structured_output(schema, method="json_mode")  needs JSON mode
    raw    plain invoke, hand the text back to deepeval        always available

Discovery order is tools -> json -> raw, and the first working mode is cached.

A rate limit must NEVER advance that discovery. An earlier version of this file
retried transient errors inside a mode and then fell through to the next mode
when the retries ran out -- so a TPM-limited run downgraded a perfectly capable
model to `raw` on its first minute and then produced "Evaluation LLM outputted
an invalid JSON" for the rest of the run. Capability failures advance the mode;
transient failures raise. They are different things and must not share a path.

Token budget
------------
Groq's on-demand tier meters TOKENS per minute, not requests, and the limit is
low: 8,000 TPM for gpt-oss-20b. One faithfulness evaluation sends the full
retrieval context twice and costs roughly 5-6k tokens, so a naive run exhausts
a minute's budget on the first row and spends the rest of its life in backoff --
until DeepEval's per-task timeout kills the metric mid-retry, which is what
"Timed out/cancelled while evaluating metric" means.

Retrying after the fact cannot fix a budget problem. So the judge paces itself:
a token bucket refills at TPM/60 per second and every call waits for its
estimated cost before it is made. Runs get slower and stop failing. This is the
standard client-side rate limiter and it is worth understanding -- server-side
429s are the fallback, not the plan.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq.chat_models import ChatGroq

from src.core.config import settings
from src.core.logging import get_logger
from src.llm.client import get_grader

log = get_logger(__name__)

MODES = ("tools", "json", "raw")

# Substrings that mean "try again", as opposed to "this model cannot do that".
_TRANSIENT = (
    "rate limit", "rate_limit", "429", "too many requests",
    "timeout", "timed out", "502", "503", "504",
    "overloaded", "service unavailable", "connection",
)

# Groq says exactly how long to wait. Guessing when you have been told is silly.
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# Rough chars-per-token for English. Overestimating is the safe direction: the
# bucket then paces slightly conservatively rather than tripping a real 429.
CHARS_PER_TOKEN = 3.5
# Judge replies are short (verdict lists, one reason). Reserved so the bucket
# accounts for output tokens, which count against TPM too.
OUTPUT_RESERVE = 800


def _is_transient(e: Exception) -> bool:
    return any(m in f"{type(e).__name__} {e}".lower() for m in _TRANSIENT)


def _retry_after(e: Exception) -> float | None:
    m = _RETRY_AFTER.search(str(e))
    return float(m.group(1)) if m else None


def _clean(text: str) -> str:
    """Peel markdown fences off a raw-mode response."""
    return _FENCE.sub("", text).strip()


class TokenBucket:
    """Classic token bucket over a tokens-per-minute allowance.

    Shared by every call the judge makes, sync and async, so concurrency does
    not multiply the spend rate. Thread-safe on the accounting; the waiting
    happens outside the lock so callers do not serialise on each other.
    """

    def __init__(self, tpm: int):
        self.tpm = max(1, tpm)
        self.tokens = float(self.tpm)
        self.updated = time.monotonic()
        self.waited = 0.0
        self.spent = 0
        self._lock = threading.Lock()

    def _take(self, n: int) -> float:
        """Deduct n if affordable; otherwise return seconds until it is."""
        with self._lock:
            now = time.monotonic()
            self.tokens = min(
                float(self.tpm),
                self.tokens + (now - self.updated) * self.tpm / 60.0,
            )
            self.updated = now
            # A single request larger than the whole bucket can never be
            # affordable; clamp so it waits for a full bucket instead of
            # spinning forever.
            need = min(n, self.tpm)
            if self.tokens >= need:
                self.tokens -= need
                self.spent += n
                return 0.0
            return (need - self.tokens) * 60.0 / self.tpm

    def acquire(self, n: int) -> None:
        while (wait := self._take(n)) > 0:
            self.waited += wait
            time.sleep(wait)

    async def a_acquire(self, n: int) -> None:
        while (wait := self._take(n)) > 0:
            self.waited += wait
            await asyncio.sleep(wait)


def _build_llm(model_name: str | None, temperature: float | None) -> ChatGroq:
    """Reuse the project's grader when nothing is overridden."""
    if model_name is None and temperature is None:
        return get_grader()

    if settings.groq_api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.groq_api_key

    return ChatGroq(
        model=model_name or settings.grader_model_name,
        temperature=(
            settings.grader_temperature if temperature is None else temperature
        ),
    )


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge over Groq chat models, paced to a TPM budget."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        tpm: int = 8000,
        retries: int = 5,
        base_delay: float = 2.0,
    ):
        # Set before super().__init__ -- the base class calls load_model()
        # from inside its own constructor.
        self._model_name = model_name or settings.grader_model_name
        self._temperature = temperature
        self._retries = retries
        self._base_delay = base_delay
        self._runnables: dict[Any, Any] = {}
        self._mode: str | None = None
        self.bucket = TokenBucket(tpm)
        self.stats = {"calls": 0, "retries": 0, "failures": 0}
        super().__init__(model=self._model_name)

        if self._model_name == settings.llm_model:
            log.warning(
                f"judge and generator are both {self._model_name}. Scores will "
                f"be optimistic; set GRADER_MODEL to a different checkpoint."
            )

    def load_model(self) -> ChatGroq:
        return _build_llm(self._model_name, self._temperature)

    def get_model_name(self) -> str:
        return self._model_name

    # -----------------------------------------------------------
    # plumbing
    # -----------------------------------------------------------

    @staticmethod
    def estimate(prompt: str) -> int:
        return int(len(prompt) / CHARS_PER_TOKEN) + OUTPUT_RESERVE

    def _runnable(self, schema, mode: str):
        key = (schema, mode)
        if key not in self._runnables:
            if mode == "tools":
                self._runnables[key] = self.model.with_structured_output(schema)
            elif mode == "json":
                self._runnables[key] = self.model.with_structured_output(
                    schema, method="json_mode"
                )
            else:
                self._runnables[key] = None
        return self._runnables[key]

    def _modes_to_try(self, schema) -> list[str]:
        if schema is None:
            return ["raw"]
        if self._mode is not None:
            return [self._mode] if self._mode == "raw" else [self._mode, "raw"]
        return list(MODES)

    def _adopt(self, mode: str) -> None:
        if self._mode != mode:
            log.info(f"judge {self._model_name}: '{mode}' structured-output mode")
            self._mode = mode

    def _delay(self, e: Exception, attempt: int) -> float:
        # Honour the server's own number when it gives one, with a little
        # headroom; otherwise back off exponentially.
        told = _retry_after(e)
        return told + 0.5 if told is not None else self._base_delay * (2 ** attempt)

    def _fail(self, e: Exception) -> None:
        self.stats["failures"] += 1
        log.error(f"judge call failed: {type(e).__name__}: {str(e)[:200]}")

    # -----------------------------------------------------------
    # generation
    # -----------------------------------------------------------

    def generate(self, prompt: str, schema=None, **kwargs):
        self.stats["calls"] += 1
        cost = self.estimate(prompt)

        for mode in self._modes_to_try(schema):
            for attempt in range(self._retries):
                self.bucket.acquire(cost)
                try:
                    out = (
                        _clean(self.model.invoke(prompt).content)
                        if mode == "raw"
                        else self._runnable(schema, mode).invoke(prompt)
                    )
                    self._adopt(mode)
                    return out
                except Exception as e:
                    if not _is_transient(e):
                        # A capability failure. Try the next mode.
                        log.warning(
                            f"judge mode '{mode}' unusable "
                            f"({type(e).__name__}: {str(e)[:160]})"
                        )
                        break
                    if attempt == self._retries - 1:
                        # Transient and out of retries: raise. Falling through
                        # to the next mode here would mistake a rate limit for
                        # a missing capability and cripple the judge.
                        self._fail(e)
                        raise
                    self.stats["retries"] += 1
                    delay = self._delay(e, attempt)
                    log.warning(
                        f"judge transient error ({type(e).__name__}: "
                        f"{str(e)[:100]}); retry in {delay:.1f}s"
                    )
                    time.sleep(delay)

        err = RuntimeError(f"no usable structured-output mode for {self._model_name}")
        self._fail(err)
        raise err

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        self.stats["calls"] += 1
        cost = self.estimate(prompt)

        for mode in self._modes_to_try(schema):
            for attempt in range(self._retries):
                await self.bucket.a_acquire(cost)
                try:
                    out = (
                        _clean((await self.model.ainvoke(prompt)).content)
                        if mode == "raw"
                        else await self._runnable(schema, mode).ainvoke(prompt)
                    )
                    self._adopt(mode)
                    return out
                except Exception as e:
                    if not _is_transient(e):
                        log.warning(
                            f"judge mode '{mode}' unusable "
                            f"({type(e).__name__}: {str(e)[:160]})"
                        )
                        break
                    if attempt == self._retries - 1:
                        self._fail(e)
                        raise
                    self.stats["retries"] += 1
                    delay = self._delay(e, attempt)
                    log.warning(
                        f"judge transient error ({type(e).__name__}: "
                        f"{str(e)[:100]}); retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        err = RuntimeError(f"no usable structured-output mode for {self._model_name}")
        self._fail(err)
        raise err
