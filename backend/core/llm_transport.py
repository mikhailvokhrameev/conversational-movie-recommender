"""Provider HTTP protocols for the common LLM client.

No model loading or network requests happen during adapter construction.
"""
import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import Literal

import httpx
from django.conf import settings

from core import openrouter_client
from core.openrouter_client import InferenceError

logger = logging.getLogger(__name__)

def _ollama_options(kind: Literal["structured", "generation"]) -> dict:
    """Ollama `options` sent with EVERY request. All calls must use the same
    num_ctx: a differing value makes Ollama reload the model.

    `kind` picks the temperature: "structured" for schema-constrained calls whose output
    is parsed (deterministic, so intent parsing and the eval judge are
    reproducible), "generation" for user-facing free text. The seed is fixed
    for both.
    """
    return {
        "num_ctx": settings.LLM_NUM_CTX,
        "seed": settings.LLM_SEED,
        "temperature": settings.LLM_TEMPERATURE[kind],
    }


# --- Call logging: tokens, latency, model identity ---

# Warn when a prompt fills this share of llm.num_ctx or more. History is capped
# by characters, not tokens (see session.history_max_*), so this is the only
# signal that a prompt is close to being truncated by Ollama.
_NUM_CTX_WARN_SHARE = 0.9


def _ns_to_ms(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value / 1_000_000


def _fmt(value, spec: str = "") -> str:
    return "n/a" if value is None else format(value, spec)


def _log_call_stats(call_type: str, stats, ttft_ms: float | None = None) -> None:
    """One INFO line per Ollama call: token counts and Ollama's own timings.

    `stats` is the non-streamed response body, or the final `done: true` chunk
    of a stream; any field may be missing and shows as n/a. `ttft_ms` is the
    client-side time to first token and only exists for streamed calls. Never
    raises: logging must not be able to fail a request.
    """
    if not isinstance(stats, dict):
        stats = {}
    # prompt_eval_count may cover only the newly evaluated tokens when Ollama
    # reuses a cached prompt prefix, so read it from a cold call.
    prompt_tokens = stats.get("prompt_eval_count")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        prompt_tokens = None
    eval_tokens = stats.get("eval_count")
    if isinstance(eval_tokens, bool) or not isinstance(eval_tokens, int):
        eval_tokens = None
    logger.info(
        "Ollama call=%s prompt_tokens=%s eval_tokens=%s prompt_eval_ms=%s eval_ms=%s ttft_ms=%s",
        call_type,
        _fmt(prompt_tokens),
        _fmt(eval_tokens),
        _fmt(_ns_to_ms(stats.get("prompt_eval_duration")), ".0f"),
        _fmt(_ns_to_ms(stats.get("eval_duration")), ".0f"),
        _fmt(ttft_ms, ".0f"),
    )
    if prompt_tokens is not None and prompt_tokens >= _NUM_CTX_WARN_SHARE * settings.LLM_NUM_CTX:
        logger.warning(
            "Ollama call=%s prompt used %d of %d context tokens (llm.num_ctx); "
            "Ollama truncates beyond it. Lower session.history_max_turns or raise llm.num_ctx.",
            call_type, prompt_tokens, settings.LLM_NUM_CTX,
        )


class _StreamStats:
    """Collects what a streamed call reports: the final `done: true` chunk
    carries the token counts, and the first non-empty content chunk gives the
    client-side time to first token (measured from construction, so create it
    just before sending the request)."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.ttft_ms: float | None = None
        self.final: dict = {}

    def observe(self, data: dict) -> None:
        if self.ttft_ms is None and data.get("message", {}).get("content"):
            self.ttft_ms = (time.perf_counter() - self._started) * 1000
        if data.get("done"):
            self.final = data

    def log(self, call_type: str) -> None:
        _log_call_stats(call_type, self.final, self.ttft_ms)


# The tag behind settings.LLM_MODEL (e.g. "qwen3:8b") is mutable and cannot
# be pinned by digest in `ollama pull`, so the digest actually serving requests
# is logged once per process to make a run traceable. One attempt only: a
# failed lookup costs at most one short timeout, never one per request.
_digest_lookup_attempted = False
_DIGEST_LOOKUP_TIMEOUT_SECONDS = 5.0


def _model_digest(tags) -> str | None:
    """The digest of the configured model in an `/api/tags` body, if listed.

    Type-defensive: the body comes from the network, and this runs on the first
    request of a process, so a malformed shape must yield None, not an exception.
    """
    if not isinstance(tags, dict):
        return None
    models = tags.get("models")
    if not isinstance(models, list):
        return None
    wanted = {settings.LLM_MODEL, f"{settings.LLM_MODEL}:latest"}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name in wanted:
            digest = entry.get("digest")
            return digest if isinstance(digest, str) else None
    return None


def _report_model_digest(tags) -> None:
    digest = _model_digest(tags)
    if digest:
        logger.info("Ollama model %s digest %s", settings.LLM_MODEL, digest)
    else:
        logger.warning("Ollama model %s not listed by /api/tags; digest unknown", settings.LLM_MODEL)


def _log_model_digest_once() -> None:
    global _digest_lookup_attempted
    if _digest_lookup_attempted:
        return
    _digest_lookup_attempted = True
    try:
        response = httpx.get(
            f"{settings.LLM_BASE_URL}/api/tags", timeout=_DIGEST_LOOKUP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        _report_model_digest(response.json())
    except Exception as e:  # logging must never fail a request
        logger.warning("Could not read the Ollama model digest: %s", e)


async def _alog_model_digest_once(client: httpx.AsyncClient) -> None:
    """Async variant of `_log_model_digest_once`, reusing the caller's client."""
    global _digest_lookup_attempted
    if _digest_lookup_attempted:
        return
    _digest_lookup_attempted = True
    try:
        response = await client.get(
            f"{settings.LLM_BASE_URL}/api/tags", timeout=_DIGEST_LOOKUP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        _report_model_digest(response.json())
    except Exception as e:  # logging must never fail a request
        logger.warning("Could not read the Ollama model digest: %s", e)


class OllamaLLMAdapter:
    """Ollama chat/NDJSON wire format, with the existing local diagnostics."""

    provider = "ollama"

    @staticmethod
    def _payload(messages, schema, kind, stream=False):
        payload = {
            "model": settings.LLM_MODEL, "messages": messages, "stream": stream,
            "think": settings.LLM_THINKING, "options": _ollama_options(kind),
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    def complete(self, messages, *, schema, kind, timeout, call_type):
        _log_model_digest_once()
        response = httpx.post(
            f"{settings.LLM_BASE_URL}/api/chat",
            json=self._payload(messages, schema, kind), timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        _log_call_stats(call_type, body)
        return body["message"]["content"]

    async def acomplete(self, messages, *, schema, kind, timeout, call_type):
        async with httpx.AsyncClient() as client:
            await _alog_model_digest_once(client)
            response = await client.post(
                f"{settings.LLM_BASE_URL}/api/chat",
                json=self._payload(messages, schema, kind), timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            _log_call_stats(call_type, body)
            return body["message"]["content"]

    def stream(self, messages, *, schema, kind, timeout, call_type):
        stats = _StreamStats()
        with httpx.stream("POST", f"{settings.LLM_BASE_URL}/api/chat",
                          json=self._payload(messages, schema, kind, True), timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    stats.observe(data)
                    if data.get("error"):
                        raise InferenceError("ollama_stream")
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        stats.log(call_type)

    async def astream(self, messages, *, schema, kind, timeout, call_type):
        stats = _StreamStats()
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{settings.LLM_BASE_URL}/api/chat",
                                     json=self._payload(messages, schema, kind, True), timeout=timeout) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        stats.observe(data)
                        if data.get("error"):
                            raise InferenceError("ollama_stream")
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
        stats.log(call_type)

    def is_available(self):
        try:
            return httpx.get(f"{settings.LLM_BASE_URL}/", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    async def ais_available(self):
        try:
            async with httpx.AsyncClient() as client:
                return (await client.get(f"{settings.LLM_BASE_URL}/", timeout=5.0)).status_code == 200
        except httpx.HTTPError:
            return False


def _strict_schema(schema):
    """Strict schemas require every object key, including nullable fields."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


def _content(body, *, streaming=False):
    """Extract only visible text; provider reasoning fields never reach callers."""
    error = None
    try:
        choices = body.get("choices", []) if streaming else body["choices"]
        if streaming and not choices:
            return ""
        message = choices[0]["delta" if streaming else "message"]
        content = message.get("content")
        if streaming and content is None:
            return ""
        if isinstance(content, str):
            return content
    except (KeyError, IndexError, TypeError, AttributeError):
        error = InferenceError("chat_response")
    # Raise outside an exception handler: upstream bodies/exception contexts stay private.
    raise error or InferenceError("chat_response")


class _SSEEvents:
    """Assemble SSE data fields across lines (httpx handles byte boundaries)."""

    def __init__(self):
        self.data = []
        self.done = False

    def feed(self, line):
        if line:
            if line.startswith("data:"):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(" ") else value)
            return None
        if not self.data:
            return None
        value = "\n".join(self.data)
        self.data.clear()
        if value == "[DONE]":
            self.done = True
            return None
        error = None
        try:
            event = json.loads(value)
        except ValueError:
            error = InferenceError("stream_response")
        if error:
            raise error
        if not isinstance(event, dict):
            raise InferenceError("stream_response")
        if "error" in event:
            upstream = event["error"]
            code = upstream.get("code") if isinstance(upstream, dict) else None
            if not isinstance(code, int) or isinstance(code, bool):
                code = None
            raise InferenceError("stream_response", code, code == 429 or (code is not None and 500 <= code < 600))
        return event


class OpenRouterLLMAdapter:
    """OpenAI-compatible completions/SSE without retrying an emitted token."""

    provider = "openrouter"

    @staticmethod
    def _payload(messages, schema, kind, stream=False):
        payload = {
            "model": settings.LLM_MODEL, "messages": messages, "stream": stream,
            "temperature": settings.LLM_TEMPERATURE[kind], "seed": settings.LLM_SEED,
            "max_tokens": settings.LLM_MAX_OUTPUT_TOKENS[kind],
            "reasoning": {"enabled": settings.LLM_THINKING},
            "provider": {"require_parameters": True},
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "strict": True, "schema": _strict_schema(schema),
            }}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _log(call_type, body):
        usage = body.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        def count(name):
            value = usage.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else "n/a"
        # Log counts only; never log model output, prompts, provider errors or credentials.
        logger.info("OpenRouter call=%s prompt_tokens=%s eval_tokens=%s", call_type,
                    count("prompt_tokens"), count("completion_tokens"))

    def complete(self, messages, *, schema, kind, timeout, call_type):
        body = openrouter_client.post_json("chat/completions", self._payload(messages, schema, kind), timeout=timeout)
        self._log(call_type, body)
        return _content(body)

    async def acomplete(self, messages, *, schema, kind, timeout, call_type):
        body = await openrouter_client.apost_json("chat/completions", self._payload(messages, schema, kind), timeout=timeout)
        self._log(call_type, body)
        return _content(body)

    def stream(self, messages, *, schema, kind, timeout, call_type):
        url, headers, policy = openrouter_client._request("chat/completions")
        payload = self._payload(messages, schema, kind, True)
        emitted = False
        with httpx.Client(follow_redirects=False) as client:
            for attempt in range(policy["max_attempts"]):
                response = None
                error = None
                try:
                    with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
                        if not 200 <= response.status_code < 300:
                            code = response.status_code
                            error = InferenceError("stream", code, code == 429 or 500 <= code < 600)
                        else:
                            events = _SSEEvents()
                            for line in response.iter_lines():
                                event = events.feed(line)
                                if events.done:
                                    return
                                if event is not None:
                                    if "usage" in event:
                                        self._log(call_type, event)
                                    content = _content(event, streaming=True)
                                    if content:
                                        emitted = True
                                        yield content
                            error = InferenceError("stream_incomplete", retryable=True)
                except httpx.RequestError:
                    error = InferenceError("stream", retryable=True)
                except InferenceError as failure:
                    error = failure
                if emitted or not error.retryable or attempt + 1 >= policy["max_attempts"]:
                    raise error
                time.sleep(openrouter_client._delay(policy, attempt, response))

    async def astream(self, messages, *, schema, kind, timeout, call_type):
        url, headers, policy = openrouter_client._request("chat/completions")
        payload = self._payload(messages, schema, kind, True)
        emitted = False
        async with httpx.AsyncClient(follow_redirects=False) as client:
            for attempt in range(policy["max_attempts"]):
                response = None
                error = None
                try:
                    async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
                        if not 200 <= response.status_code < 300:
                            code = response.status_code
                            error = InferenceError("stream", code, code == 429 or 500 <= code < 600)
                        else:
                            events = _SSEEvents()
                            async for line in response.aiter_lines():
                                event = events.feed(line)
                                if events.done:
                                    return
                                if event is not None:
                                    if "usage" in event:
                                        self._log(call_type, event)
                                    content = _content(event, streaming=True)
                                    if content:
                                        emitted = True
                                        yield content
                            error = InferenceError("stream_incomplete", retryable=True)
                except httpx.RequestError:
                    error = InferenceError("stream", retryable=True)
                except InferenceError as failure:
                    error = failure
                if emitted or not error.retryable or attempt + 1 >= policy["max_attempts"]:
                    raise error
                await asyncio.sleep(openrouter_client._delay(policy, attempt, response))

    def is_available(self):
        try:
            url, headers, _ = openrouter_client._request("models")
            return httpx.get(url, headers=headers, timeout=settings.LLM_TIMEOUTS["intent"],
                             follow_redirects=False).status_code == 200
        except (httpx.HTTPError, InferenceError):
            return False

    async def ais_available(self):
        try:
            url, headers, _ = openrouter_client._request("models")
            async with httpx.AsyncClient(follow_redirects=False) as client:
                return (await client.get(url, headers=headers,
                                         timeout=settings.LLM_TIMEOUTS["intent"])).status_code == 200
        except (httpx.HTTPError, InferenceError):
            return False
