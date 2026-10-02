"""
llm.py — one place that creates the chat model and wraps calls.

Every agent uses two helpers:
  structured_call(Schema, system, user) -> Pydantic object
  text_call(system, user)               -> plain string
"""
import json
import re
import time
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel

from core.config import LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, provider_ready

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=4)
def get_llm(temperature: float = LLM_TEMPERATURE):
    ok, msg = provider_ready()
    if not ok:
        raise RuntimeError(msg)

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=LLM_MODEL, temperature=temperature, max_retries=2)

    raise RuntimeError(f"Unsupported provider: {LLM_PROVIDER}")


MAX_WAIT_SECONDS = 90   # longer than this = daily limit: give up and report
MAX_RATE_RETRIES = 3


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "rate_limit" in msg or "429" in msg


def _retry_after(e: Exception) -> float | None:
    """Parse Groq's 'Please try again in 1m2.5s' / '12.3s' / '850ms'. None if not found."""
    msg = str(e)
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?", msg)
    if not m or not any(m.groups()):
        return None
    h, mins, secs, ms = m.groups()
    return (int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs or 0) + float(ms or 0) / 1000)


def _wait_if_short(e: Exception, attempt: int) -> bool:
    """
    Per-minute limits clear quickly: wait and let the caller retry.
    Daily limits do not: return False so the error is reported.
    """
    if not _is_rate_limit(e) or attempt >= MAX_RATE_RETRIES:
        return False
    wait = _retry_after(e)
    if wait is None:
        wait = 20 * (attempt + 1)  # no hint given: gentle backoff
    if wait > MAX_WAIT_SECONDS:
        return False
    time.sleep(wait + 1)
    return True


def _content_text(content) -> str:
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    return str(content).strip()


def _json_fallback(schema: Type[T], system: str, user: str, temperature: float) -> T:
    """Plan B for models that fail at tool calling: ask for raw JSON and validate it."""
    instructions = (
        system + "\n\nRespond with ONLY one JSON object that matches this JSON schema. "
        "No prose, no markdown fences. Use an empty list [] where nothing applies.\n"
        + json.dumps(schema.model_json_schema())
    )
    raw = _content_text(get_llm(temperature).invoke([("system", instructions), ("human", user)]).content)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model returned no JSON object")
    return schema.model_validate_json(raw[start:end + 1])


def structured_call(schema: Type[T], system: str, user: str,
                    retries: int = 1, temperature: float = LLM_TEMPERATURE) -> T:
    """
    Ask the LLM to answer in the shape of a Pydantic schema.
    1) native structured output (tool calling), with a retry
    2) per-minute rate limits: wait as long as the provider asks, then retry
    3) if output keeps failing, plain-JSON fallback validated by Pydantic
    """
    llm = get_llm(temperature).with_structured_output(schema)
    messages = [("system", system), ("human", user)]
    last_error, attempt, waits = None, 0, 0
    while attempt <= retries:
        try:
            result = llm.invoke(messages)
            if result is not None:
                return result
            attempt += 1
        except Exception as e:
            last_error = e
            if _is_rate_limit(e):
                if _wait_if_short(e, waits):
                    waits += 1
                    continue          # rate-limit waits don't use up a retry
                break                 # daily limit: retrying won't help
            attempt += 1
    if last_error is None or not _is_rate_limit(last_error):
        for _ in range(MAX_RATE_RETRIES + 1):
            try:
                return _json_fallback(schema, system, user, temperature)
            except Exception as e:
                last_error = e
                if not _wait_if_short(e, waits):
                    break
                waits += 1
    raise RuntimeError(f"Structured call for {schema.__name__} failed: {last_error}")


def text_call(system: str, user: str, temperature: float = LLM_TEMPERATURE) -> str:
    waits = 0
    while True:
        try:
            response = get_llm(temperature).invoke([("system", system), ("human", user)])
            return _content_text(response.content)
        except Exception as e:
            if not _wait_if_short(e, waits):
                raise
            waits += 1